// MobileGlues - gl/shader.cpp
// Copyright (c) 2025-2026 MobileGL-Dev
// Licensed under the GNU Lesser General Public License v2.1
// SPDX-License-Identifier: LGPL-2.1-only

#include <cctype>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "shader.h"

#include <spirv_cross/spirv.h>

#ifndef GL_SHADER_BINARY_FORMAT_SPIR_V
#define GL_SHADER_BINARY_FORMAT_SPIR_V 0x9551
#endif

#include <GL/gl.h>

#include "log.h"
#include "program.h"
#include "../gles/loader.h"
#include "../includes.h"
#include "glsl/glsl_for_es.h"
#include "../config/settings.h"
#include "FSR1/FSR1.h"

#define DEBUG 0

struct shader_t shaderInfo;

UnorderedMap<GLuint, bool> shader_map_is_sampler_buffer_emulated;

/*
 * Minecraft 26.3 can provide SPIR-V through glShaderBinary()
 * and then call glSpecializeShader().
 *
 * GLES drivers used by MobileGlues generally do not provide the
 * desktop OpenGL SPIR-V path directly, so MobileGlues keeps the
 * SPIR-V module and translates it to ESSL through SPIRV-Cross.
 */
static UnorderedMap<GLuint, std::vector<uint32_t>> shader_spirv;


/* -------------------------------------------------------------
 * GLSL / ESSL helpers
 * ------------------------------------------------------------- */

bool can_run_essl3(unsigned int esversion, const char* glsl) {
    if (!glsl)
        return false;

    if (strncmp(glsl, "#version 100", 12) == 0)
        return true;

    unsigned int glsl_version = 0;

    if (strncmp(glsl, "#version 300 es", 15) == 0) {
        glsl_version = 300;
    } else if (strncmp(glsl, "#version 310 es", 15) == 0) {
        glsl_version = 310;
    } else if (strncmp(glsl, "#version 320 es", 15) == 0) {
        glsl_version = 320;
    } else {
        return false;
    }

    return esversion >= glsl_version;
}

bool is_direct_shader(const char* glsl) {
    if (!glsl || !hardware)
        return false;

    return can_run_essl3(hardware->es_version, glsl);
}

bool check_if_sampler_buffer_used(std::string str) {
    return str.find("samplerBuffer") != std::string::npos;
}


/* -------------------------------------------------------------
 * SPIR-V specialization constants
 * ------------------------------------------------------------- */

static bool apply_spirv_specialization(
    spvc_compiler compiler,
    GLuint count,
    const GLuint* ids,
    const GLuint* values
) {
    if (count == 0)
        return true;

    if (!ids || !values) {
        LOG_E(
            "glSpecializeShader: specialization arrays are null"
        )
        return false;
    }

    for (GLuint i = 0; i < count; ++i) {

        spvc_constant constant =
            spvc_compiler_get_constant_handle(
                compiler,
                ids[i]
            );

        if (!constant) {
            LOG_W_FORCE(
                "glSpecializeShader: specialization constant %u "
                "was not found",
                ids[i]
            );

            /*
             * A missing specialization constant is not necessarily
             * fatal. Some SPIR-V producers may provide constants
             * that are optimized away.
             */
            continue;
        }

        spvc_type type =
            spvc_compiler_get_type_handle(
                compiler,
                spvc_constant_get_type(constant)
            );

        if (!type) {
            LOG_E(
                "glSpecializeShader: could not inspect "
                "specialization constant %u",
                ids[i]
            );

            return false;
        }

        switch (spvc_type_get_basetype(type)) {

        case SPVC_BASETYPE_BOOLEAN:
            spvc_constant_set_scalar_u32(
                constant,
                0,
                0,
                values[i] ? 1u : 0u
            );
            break;

        case SPVC_BASETYPE_INT8:
        case SPVC_BASETYPE_INT16:
        case SPVC_BASETYPE_INT32:
            spvc_constant_set_scalar_i32(
                constant,
                0,
                0,
                static_cast<int32_t>(values[i])
            );
            break;

        case SPVC_BASETYPE_UINT8:
        case SPVC_BASETYPE_UINT16:
        case SPVC_BASETYPE_UINT32:
            spvc_constant_set_scalar_u32(
                constant,
                0,
                0,
                values[i]
            );
            break;

        case SPVC_BASETYPE_INT64:
            spvc_constant_set_scalar_i64(
                constant,
                0,
                0,
                static_cast<int64_t>(
                    static_cast<int32_t>(values[i])
                )
            );
            break;

        case SPVC_BASETYPE_UINT64:
            spvc_constant_set_scalar_u64(
                constant,
                0,
                0,
                static_cast<uint64_t>(values[i])
            );
            break;

        case SPVC_BASETYPE_FP16:
            spvc_constant_set_scalar_u16(
                constant,
                0,
                0,
                static_cast<uint16_t>(
                    values[i] & 0xffffu
                )
            );
            break;

        case SPVC_BASETYPE_FP32: {
            float value = 0.0f;
            uint32_t bits = values[i];

            std::memcpy(
                &value,
                &bits,
                sizeof(value)
            );

            spvc_constant_set_scalar_fp32(
                constant,
                0,
                0,
                value
            );

            break;
        }

        case SPVC_BASETYPE_FP64:
            /*
             * OpenGL provides a GLuint here. Do not reinterpret
             * unrelated memory as a 64-bit floating point value.
             */
            spvc_constant_set_scalar_fp64(
                constant,
                0,
                0,
                static_cast<double>(values[i])
            );
            break;

        default:
            LOG_W_FORCE(
                "glSpecializeShader: unsupported specialization "
                "type %d for constant %u",
                static_cast<int>(
                    spvc_type_get_basetype(type)
                ),
                ids[i]
            );

            return false;
        }
    }

    return true;
}


/* -------------------------------------------------------------
 * Determine SPIR-V execution model from shader type
 * ------------------------------------------------------------- */

static bool get_shader_execution_model(
    GLuint shader,
    SpvExecutionModel& model
) {
    if (!hardware)
        return false;

    GLint shader_type = 0;

    GLES.glGetShaderiv(
        shader,
        GL_SHADER_TYPE,
        &shader_type
    );

    switch (shader_type) {

    case GL_VERTEX_SHADER:
        model = SpvExecutionModelVertex;
        return true;

    case GL_FRAGMENT_SHADER:
        model = SpvExecutionModelFragment;
        return true;

    case GL_COMPUTE_SHADER:
        model = SpvExecutionModelGLCompute;
        return true;

    default:
        LOG_E(
            "glSpecializeShader: unsupported shader type 0x%x "
            "for shader %u",
            shader_type,
            shader
        );

        return false;
    }
}


/* -------------------------------------------------------------
 * SPIR-V -> ESSL
 * ------------------------------------------------------------- */

void glShaderBinary(
    GLsizei count,
    const GLuint* shaders,
    GLenum binaryformat,
    const void* binary,
    GLsizei length
) {
    LOG()

    if (count <= 0 || !shaders || !binary || length <= 0) {
        LOG_E("glShaderBinary: invalid arguments")
        return;
    }

    /*
     * Anything other than SPIR-V should continue to the real
     * GLES implementation.
     */
    if (binaryformat != GL_SHADER_BINARY_FORMAT_SPIR_V) {

        GLES.glShaderBinary(
            count,
            shaders,
            binaryformat,
            binary,
            length
        );

        CHECK_GL_ERROR
        return;
    }

    if ((length % static_cast<GLsizei>(
            sizeof(uint32_t))) != 0) {

        LOG_E(
            "glShaderBinary: SPIR-V length %d is not "
            "4-byte aligned",
            length
        );

        return;
    }

    const uint32_t* words =
        static_cast<const uint32_t*>(binary);

    const size_t word_count =
        static_cast<size_t>(length) /
        sizeof(uint32_t);

    /*
     * SPIR-V minimum module header is 5 words.
     */
    if (word_count < 5 ||
        words[0] != 0x07230203u) {

        LOG_E(
            "glShaderBinary: invalid SPIR-V module "
            "(magic=0x%08x)",
            word_count ? words[0] : 0u
        );

        return;
    }

    for (GLsizei i = 0; i < count; ++i) {

        if (shaders[i] == 0)
            continue;

        shader_spirv[shaders[i]] =
            std::vector<uint32_t>(
                words,
                words + word_count
            );
    }

    LOG_D(
        "glShaderBinary: captured %d SPIR-V shader(s), "
        "%zu words",
        count,
        word_count
    )

    /*
     * We intentionally do NOT submit SPIR-V to the physical GLES
     * driver here. It is translated during glSpecializeShader().
     */
    CHECK_GL_ERROR
}


/* -------------------------------------------------------------
 * glSpecializeShader
 * ------------------------------------------------------------- */

void glSpecializeShader(
    GLuint shader,
    const GLchar* pEntryPoint,
    GLuint numSpecializationConstants,
    const GLuint* pConstantIndex,
    const GLuint* pConstantValue
) {
    LOG()

    auto it = shader_spirv.find(shader);

    if (it == shader_spirv.end() ||
        it->second.empty()) {

        LOG_E(
            "glSpecializeShader: no SPIR-V binary stored "
            "for shader %u",
            shader
        );

        return;
    }

    const char* entry =
        (pEntryPoint && *pEntryPoint)
            ? pEntryPoint
            : "main";

    SpvExecutionModel execution_model;

    /*
     * IMPORTANT:
     *
     * Do not initially assume Vertex here.
     * Fragment shaders were previously failing because the
     * compiler was first configured with Vertex execution model.
     */
    if (!get_shader_execution_model(
            shader,
            execution_model
        )) {

        shader_spirv.erase(shader);
        return;
    }

    spvc_context context = nullptr;
    spvc_parsed_ir ir = nullptr;
    spvc_compiler compiler = nullptr;
    spvc_compiler_options options = nullptr;

    const auto& spirv = it->second;

    /* Create SPIRV-Cross context */
    if (spvc_context_create(&context) != SPVC_SUCCESS ||
        !context) {

        LOG_E(
            "glSpecializeShader: could not create "
            "SPIRV-Cross context"
        );

        shader_spirv.erase(shader);
        return;
    }

    /* Parse SPIR-V */
    if (spvc_context_parse_spirv(
            context,
            spirv.data(),
            spirv.size(),
            &ir
        ) != SPVC_SUCCESS ||
        !ir) {

        LOG_E(
            "glSpecializeShader: SPIR-V parsing failed: %s",
            spvc_context_get_last_error_string(context)
        );

        spvc_context_destroy(context);
        shader_spirv.erase(shader);
        return;
    }

    /*
     * Generate GLSL/ESSL source.
     *
     * SPVC_BACKEND_GLSL is used because SPIRV-Cross supports
     * GLSL ES output through SPVC_COMPILER_OPTION_GLSL_ES.
     */
    if (spvc_context_create_compiler(
            context,
            SPVC_BACKEND_GLSL,
            ir,
            SPVC_CAPTURE_MODE_TAKE_OWNERSHIP,
            &compiler
        ) != SPVC_SUCCESS ||
        !compiler) {

        LOG_E(
            "glSpecializeShader: could not create "
            "SPIRV-Cross compiler: %s",
            spvc_context_get_last_error_string(context)
        );

        spvc_context_destroy(context);
        shader_spirv.erase(shader);
        return;
    }

    /*
     * Set the REAL shader execution model exactly once.
     *
     * This fixes the previous:
     *
     *   Vertex -> Fragment/Compute
     *
     * mismatch.
     */
    if (spvc_compiler_set_entry_point(
            compiler,
            entry,
            execution_model
        ) != SPVC_SUCCESS) {

        LOG_E(
            "glSpecializeShader: could not set entry point "
            "'%s' for shader %u: %s",
            entry,
            shader,
            spvc_context_get_last_error_string(context)
        );

        spvc_context_destroy(context);
        shader_spirv.erase(shader);
        return;
    }

    /*
     * Apply specialization constants BEFORE compilation.
     */
    if (!apply_spirv_specialization(
            compiler,
            numSpecializationConstants,
            pConstantIndex,
            pConstantValue
        )) {

        LOG_E(
            "glSpecializeShader: failed to apply "
            "specialization constants"
        );

        spvc_context_destroy(context);
        shader_spirv.erase(shader);
        return;
    }

    /* Create compiler options */
    if (spvc_compiler_create_compiler_options(
            compiler,
            &options
        ) != SPVC_SUCCESS ||
        !options) {

        LOG_E(
            "glSpecializeShader: could not create "
            "SPIRV-Cross compiler options: %s",
            spvc_context_get_last_error_string(context)
        );

        spvc_context_destroy(context);
        shader_spirv.erase(shader);
        return;
    }

    /*
     * Select an ESSL version that the actual GLES device can run.
     */
    unsigned int essl_version = 300;

    if (hardware) {
        if (hardware->es_version >= 320)
            essl_version = 320;
        else if (hardware->es_version >= 310)
            essl_version = 310;
        else
            essl_version = 300;
    }

    if (spvc_compiler_options_set_uint(
            options,
            SPVC_COMPILER_OPTION_GLSL_VERSION,
            essl_version
        ) != SPVC_SUCCESS) {

        LOG_E(
            "glSpecializeShader: failed to set GLSL ES "
            "version %u",
            essl_version
        );

        spvc_context_destroy(context);
        shader_spirv.erase(shader);
        return;
    }

    if (spvc_compiler_options_set_bool(
            options,
            SPVC_COMPILER_OPTION_GLSL_ES,
            SPVC_TRUE
        ) != SPVC_SUCCESS) {

        LOG_E(
            "glSpecializeShader: failed to enable GLSL ES "
            "output"
        );

        spvc_context_destroy(context);
        shader_spirv.erase(shader);
        return;
    }

    if (spvc_compiler_install_compiler_options(
            compiler,
            options
        ) != SPVC_SUCCESS) {

        LOG_E(
            "glSpecializeShader: failed to install "
            "SPIRV-Cross compiler options: %s",
            spvc_context_get_last_error_string(context)
        );

        spvc_context_destroy(context);
        shader_spirv.erase(shader);
        return;
    }

    /* Compile SPIR-V -> ESSL */
    const char* result = nullptr;

    if (spvc_compiler_compile(
            compiler,
            &result
        ) != SPVC_SUCCESS ||
        !result ||
        !*result) {

        LOG_E(
            "glSpecializeShader: SPIRV-Cross failed: %s",
            spvc_context_get_last_error_string(context)
        );

        spvc_context_destroy(context);
        shader_spirv.erase(shader);
        return;
    }

    std::string essl(result);

    /*
     * Apply MobileGlues compatibility transformations.
     */
    essl = removeLayoutBinding(essl);
    essl = processOutColorLocations(essl);
    essl = forceSupporterOutput(essl);

    if (essl.empty()) {

        LOG_E(
            "glSpecializeShader: translated ESSL is empty "
            "for shader %u",
            shader
        );

        spvc_context_destroy(context);
        shader_spirv.erase(shader);
        return;
    }

    LOG_D(
        "[SPIR-V] Shader %u translated to ESSL %u",
        shader,
        essl_version
    );

    LOG_D(
        "[SPIR-V] Generated ESSL:\n%s",
        essl.c_str()
    );

    /*
     * Replace the SPIR-V shader object with the generated
     * ESSL source and compile it through the real GLES driver.
     */
    const char* source = essl.c_str();

    GLES.glShaderSource(
        shader,
        1,
        &source,
        nullptr
    );

    GLES.glCompileShader(shader);

    /*
     * Check the actual driver compilation result immediately.
     * Do not silently turn this into success here.
     */
    GLint compile_status = GL_FALSE;

    GLES.glGetShaderiv(
        shader,
        GL_COMPILE_STATUS,
        &compile_status
    );

    if (!compile_status) {

        GLchar info_log[4096] = {};
        GLsizei log_length = 0;

        GLES.glGetShaderInfoLog(
            shader,
            sizeof(info_log) - 1,
            &log_length,
            info_log
        );

        info_log[
            (log_length > 0 &&
             log_length < static_cast<GLsizei>(
                 sizeof(info_log)))
                ? log_length
                : sizeof(info_log) - 1
        ] = '\0';

        LOG_E(
            "glSpecializeShader: generated ESSL failed "
            "to compile for shader %u:\n%s",
            shader,
            info_log
        );

        /*
         * Keep the shader as failed here.
         *
         * glGetShaderiv() may later apply MobileGlues'
         * configured IgnoreError behavior if explicitly enabled.
         */
    } else {

        LOG_D(
            "glSpecializeShader: shader %u successfully "
            "translated and compiled as ESSL",
            shader
        );
    }

    spvc_context_destroy(context);

    /*
     * The SPIR-V module is no longer needed.
     */
    shader_spirv.erase(shader);

    CHECK_GL_ERROR
}


/* -------------------------------------------------------------
 * Traditional GLSL shader path
 * ------------------------------------------------------------- */

void glShaderSource(
    GLuint shader,
    GLsizei count,
    const GLchar* const* string,
    const GLint* length
) {
    LOG()

    shaderInfo.id = 0;
    shaderInfo.converted = "";
    shaderInfo.frag_data_changed_converted.clear();
    shaderInfo.frag_data_changed = 0;

    if (count <= 0 || !string) {
        LOG_E(
            "glShaderSource: invalid source arguments"
        );
        return;
    }

    size_t l = 0;

    for (int i = 0; i < count; ++i) {
        if (!string[i])
            continue;

        l += (length && length[i] >= 0)
            ? static_cast<size_t>(length[i])
            : strlen(string[i]);
    }

    std::string glsl_src;
    std::string essl_src;

    glsl_src.reserve(l + 1);

    if (length) {

        for (int i = 0; i < count; ++i) {

            if (!string[i])
                continue;

            if (length[i] >= 0)
                glsl_src += std::string_view(
                    string[i],
                    static_cast<size_t>(length[i])
                );
            else
                glsl_src += string[i];
        }

    } else {

        for (int i = 0; i < count; ++i) {

            if (!string[i])
                continue;

            glsl_src += string[i];
        }
    }

    bool is_sampler_buffer_emulated =
        hardware &&
        hardware->emulate_texture_buffer &&
        check_if_sampler_buffer_used(glsl_src);

    if (is_direct_shader(glsl_src.c_str())) {

        LOG_D(
            "[INFO] [Shader] Direct shader source:"
        );

        LOG_D(
            "%s",
            glsl_src.c_str()
        );

        essl_src = glsl_src;

    } else {

        int glsl_version =
            getGLSLVersion(glsl_src.c_str());

        LOG_D(
            "[INFO] [Shader] Shader source:"
        );

        LOG_D(
            "%s",
            glsl_src.c_str()
        );

        GLint shaderType = 0;

        GLES.glGetShaderiv(
            shader,
            GL_SHADER_TYPE,
            &shaderType
        );

        int return_code = 0;

        essl_src = GLSLtoGLSLES(
            glsl_src.c_str(),
            shaderType,
            hardware->es_version,
            glsl_version,
            return_code
        );

        if (essl_src.empty()) {

            LOG_E(
                "Failed to convert shader %d.",
                shader
            );

            return;
        }

        LOG_D(
            "\n[INFO] [Shader] Converted Shader source:\n%s",
            essl_src.c_str()
        );
    }

    if (!essl_src.empty()) {

        shaderInfo.id = shader;
        shaderInfo.converted = essl_src;

        const char* source =
            essl_src.c_str();

        GLES.glShaderSource(
            shader,
            1,
            &source,
            nullptr
        );

        if (hardware->emulate_texture_buffer) {

            shader_map_is_sampler_buffer_emulated[shader] =
                is_sampler_buffer_emulated;
        }

    } else {

        LOG_E(
            "Failed to convert GLSL."
        );
    }

    CHECK_GL_ERROR
}


/* -------------------------------------------------------------
 * Shader status
 * ------------------------------------------------------------- */

void glGetShaderiv(
    GLuint shader,
    GLenum pname,
    GLint* params
) {
    LOG()

    if (!params) {
        LOG_E(
            "glGetShaderiv: params is null"
        );
        return;
    }

    GLES.glGetShaderiv(
        shader,
        pname,
        params
    );

    /*
     * Keep the existing MobileGlues compatibility behavior,
     * but only for shader compilation status.
     */
    if (global_settings.ignore_error >=
            IgnoreErrorLevel::Partial &&
        pname == GL_COMPILE_STATUS &&
        !*params) {

        GLchar infoLog[4096] = {};
        GLsizei logLength = 0;

        GLES.glGetShaderInfoLog(
            shader,
            sizeof(infoLog) - 1,
            &logLength,
            infoLog
        );

        LOG_W_FORCE(
            "Shader %d compilation failed:\n%s",
            shader,
            infoLog
        );

        LOG_W_FORCE(
            "MobileGlues IgnoreError compatibility mode "
            "is active."
        );

        *params = GL_TRUE;
    }

    CHECK_GL_ERROR
}


/* -------------------------------------------------------------
 * Shader creation
 * ------------------------------------------------------------- */

GLuint glCreateShader(GLenum shaderType) {

    if (global_settings.fsr1_setting !=
            FSR1_Quality_Preset::Disabled &&
        !fsrInitialized) {

        InitFSRResources();
    }

    LOG()

    LOG_D(
        "glCreateShader(%s)",
        glEnumToString(shaderType)
    );

    GLuint shader =
        GLES.glCreateShader(shaderType);

    if (shader != 0 &&
        hardware &&
        hardware->emulate_texture_buffer) {

        shader_map_is_sampler_buffer_emulated[shader] =
            false;
    }

    CHECK_GL_ERROR

    return shader;
}
