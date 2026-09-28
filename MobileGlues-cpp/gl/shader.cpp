// MobileGlues - gl/shader.cpp
// Copyright (c) 2025-2026 MobileGL-Dev
// Licensed under the GNU Lesser General Public License v2.1:
//   https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt
// SPDX-License-Identifier: LGPL-2.1-only
// End of Source File Header

#include <cctype>
#include <cstdint>
#include <cstring>
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

// Minecraft 26.3 compiles OpenGL shaders through ShaderC and submits the
// resulting SPIR-V with glShaderBinary/glSpecializeShader.  MobileGlues used to
// leave glSpecializeShader as a no-op stub, which means the shader object never
// receives executable code on a GLES-only driver.  Keep the SPIR-V per shader
// until specialization, then translate it through the same SPIRV-Cross ESSL path
// used by GLSL conversion.
static UnorderedMap<GLuint, std::vector<uint32_t>> shader_spirv;

bool can_run_essl3(unsigned int esversion, const char* glsl) {
    if (strncmp(glsl, "#version 100", 12) == 0) {
        return true;
    }

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
    bool es3_ability = can_run_essl3(hardware->es_version, glsl);
    return es3_ability;
}

bool check_if_sampler_buffer_used(std::string str) {
    return str.find("samplerBuffer") != std::string::npos;
}

static bool apply_spirv_specialization(spvc_context context, spvc_compiler compiler, GLuint count,
                                       const GLuint* ids, const GLuint* values) {
    if (count == 0) return true;
    if (!ids || !values) {
        LOG_E("glSpecializeShader: specialization arrays are null")
        return false;
    }

    for (GLuint i = 0; i < count; ++i) {
        spvc_constant constant = spvc_compiler_get_constant_handle(compiler, ids[i]);
        if (!constant) {
            LOG_W_FORCE("glSpecializeShader: specialization constant %u was not found", ids[i])
            continue;
        }

        spvc_type type = spvc_compiler_get_type_handle(compiler, spvc_constant_get_type(constant));
        if (!type) {
            LOG_E("glSpecializeShader: could not inspect specialization constant %u", ids[i])
            return false;
        }

        switch (spvc_type_get_basetype(type)) {
        case SPVC_BASETYPE_BOOLEAN:
            spvc_constant_set_scalar_u32(constant, 0, 0, values[i] ? 1u : 0u);
            break;
        case SPVC_BASETYPE_INT8:
        case SPVC_BASETYPE_INT16:
        case SPVC_BASETYPE_INT32:
            spvc_constant_set_scalar_i32(constant, 0, 0, static_cast<int>(values[i]));
            break;
        case SPVC_BASETYPE_UINT8:
        case SPVC_BASETYPE_UINT16:
        case SPVC_BASETYPE_UINT32:
            spvc_constant_set_scalar_u32(constant, 0, 0, values[i]);
            break;
        case SPVC_BASETYPE_INT64:
            spvc_constant_set_scalar_i64(constant, 0, 0, static_cast<long long>(static_cast<int32_t>(values[i])));
            break;
        case SPVC_BASETYPE_UINT64:
            spvc_constant_set_scalar_u64(constant, 0, 0, static_cast<unsigned long long>(values[i]));
            break;
        case SPVC_BASETYPE_FP16:
            // glSpecializeShader supplies the bit pattern in a GLuint.  SPIR-V
            // specialization constants for FP16 are uncommon in Minecraft, so
            // preserve the low 16 bits as the scalar value expected by SPVC.
            spvc_constant_set_scalar_u16(constant, 0, 0, static_cast<unsigned short>(values[i] & 0xffffu));
            break;
        case SPVC_BASETYPE_FP32: {
            float f = 0.0f;
            uint32_t bits = values[i];
            std::memcpy(&f, &bits, sizeof(f));
            spvc_constant_set_scalar_fp32(constant, 0, 0, f);
            break;
        }
        case SPVC_BASETYPE_FP64: {
            // OpenGL exposes the specialization value as 32 bits.  Zero-extend
            // it rather than inventing a second 32-bit half.
            spvc_constant_set_scalar_fp64(constant, 0, 0, static_cast<double>(values[i]));
            break;
        }
        default:
            LOG_W_FORCE("glSpecializeShader: unsupported specialization type %d for constant %u",
                        static_cast<int>(spvc_type_get_basetype(type)), ids[i])
            return false;
        }
    }
    return true;
}

void glShaderBinary(GLsizei count, const GLuint* shaders, GLenum binaryformat, const void* binary, GLsizei length) {
    LOG()
    if (count < 0 || !shaders || !binary || length <= 0) {
        LOG_E("glShaderBinary: invalid arguments")
        return;
    }

    if (binaryformat != GL_SHADER_BINARY_FORMAT_SPIR_V) {
        // Non-SPIR-V formats remain native-driver binaries.
        GLES.glShaderBinary(count, shaders, binaryformat, binary, length);
        CHECK_GL_ERROR
        return;
    }

    if ((length % static_cast<GLsizei>(sizeof(uint32_t))) != 0) {
        LOG_E("glShaderBinary: SPIR-V length %d is not 4-byte aligned", length)
        return;
    }

    const uint32_t* words = static_cast<const uint32_t*>(binary);
    const size_t word_count = static_cast<size_t>(length) / sizeof(uint32_t);
    if (word_count < 5 || words[0] != 0x07230203u) {
        LOG_E("glShaderBinary: invalid SPIR-V module (magic=0x%08x)", word_count ? words[0] : 0u)
        return;
    }

    for (GLsizei i = 0; i < count; ++i) {
        if (shaders[i] == 0) continue;
        shader_spirv[shaders[i]] = std::vector<uint32_t>(words, words + word_count);
    }
    LOG_D("glShaderBinary: captured %d SPIR-V shader(s), %zu words", count, word_count)
    CHECK_GL_ERROR
}

void glSpecializeShader(GLuint shader, const GLchar* pEntryPoint, GLuint numSpecializationConstants,
                        const GLuint* pConstantIndex, const GLuint* pConstantValue) {
    LOG()
    auto it = shader_spirv.find(shader);
    if (it == shader_spirv.end() || it->second.empty()) {
        LOG_E("glSpecializeShader: no SPIR-V binary stored for shader %u", shader)
        return;
    }

    const char* entry = (pEntryPoint && *pEntryPoint) ? pEntryPoint : "main";
    spvc_context context = nullptr;
    if (spvc_context_create(&context) != SPVC_SUCCESS || !context) {
        LOG_E("glSpecializeShader: could not create SPIRV-Cross context")
        return;
    }

    spvc_parsed_ir ir = nullptr;
    spvc_compiler compiler = nullptr;
    spvc_compiler_options options = nullptr;
    const auto& spirv = it->second;

    bool ok = spvc_context_parse_spirv(context, spirv.data(), spirv.size(), &ir) == SPVC_SUCCESS && ir;
    if (ok) {
        ok = spvc_context_create_compiler(context, SPVC_BACKEND_GLSL, ir, SPVC_CAPTURE_MODE_TAKE_OWNERSHIP, &compiler) ==
             SPVC_SUCCESS && compiler;
    }
    if (ok) {
        ok = spvc_compiler_set_entry_point(compiler, entry, SpvExecutionModelVertex) == SPVC_SUCCESS;
        // The SPIR-V execution model above is corrected below from the shader
        // object's GL_SHADER_TYPE.  Keeping the initial call here gives SPVC a
        // deterministic entry point even for modules with one entry point.
        GLint shader_type = GL_VERTEX_SHADER;
        GLES.glGetShaderiv(shader, GL_SHADER_TYPE, &shader_type);
        const SpvExecutionModel model = (shader_type == GL_FRAGMENT_SHADER)
                                             ? SpvExecutionModelFragment
                                             : (shader_type == GL_COMPUTE_SHADER ? SpvExecutionModelGLCompute
                                                                                 : SpvExecutionModelVertex);
        if (ok) ok = spvc_compiler_set_entry_point(compiler, entry, model) == SPVC_SUCCESS;
    }
    if (ok) {
        ok = apply_spirv_specialization(context, compiler, numSpecializationConstants, pConstantIndex,
                                         pConstantValue);
    }
    if (ok) {
        ok = spvc_compiler_create_compiler_options(compiler, &options) == SPVC_SUCCESS && options;
    }
    if (ok) {
        ok = spvc_compiler_options_set_uint(options, SPVC_COMPILER_OPTION_GLSL_VERSION,
                                             hardware->es_version >= 300 ? hardware->es_version : 300) == SPVC_SUCCESS;
        ok = ok && spvc_compiler_options_set_bool(options, SPVC_COMPILER_OPTION_GLSL_ES, SPVC_TRUE) == SPVC_SUCCESS;
        ok = ok && spvc_compiler_install_compiler_options(compiler, options) == SPVC_SUCCESS;
    }

    const char* result = nullptr;
    if (ok) ok = spvc_compiler_compile(compiler, &result) == SPVC_SUCCESS && result;

    if (!ok) {
        LOG_E("glSpecializeShader: SPIRV-Cross failed: %s", spvc_context_get_last_error_string(context))
        spvc_context_destroy(context);
        return;
    }

    std::string essl(result);
    essl = removeLayoutBinding(essl);
    essl = processOutColorLocations(essl);
    essl = forceSupporterOutput(essl);

    const char* src = essl.c_str();
    GLES.glShaderSource(shader, 1, &src, nullptr);
    GLES.glCompileShader(shader);
    LOG_D("glSpecializeShader: translated SPIR-V shader %u to ESSL", shader)

    spvc_context_destroy(context);
    shader_spirv.erase(shader);
    CHECK_GL_ERROR
}

void glShaderSource(GLuint shader, GLsizei count, const GLchar* const* string, const GLint* length) {
    LOG()
    shaderInfo.id = 0;
    shaderInfo.converted = "";
    shaderInfo.frag_data_changed_converted.clear();
    shaderInfo.frag_data_changed = 0;
    size_t l = 0;
    for (int i = 0; i < count; i++)
        l += (length && length[i] >= 0) ? length[i] : strlen(string[i]);
    std::string glsl_src, essl_src;
    glsl_src.reserve(l + 1);
    if (length) {
        for (int i = 0; i < count; i++) {
            if (length[i] >= 0)
                glsl_src += std::string_view(string[i], length[i]);
            else
                glsl_src += string[i];
        }
    } else {
        for (int i = 0; i < count; i++) {
            glsl_src += string[i];
        }
    }

    bool is_sampler_buffer_emulated = hardware->emulate_texture_buffer && check_if_sampler_buffer_used(glsl_src);

    if (is_direct_shader(glsl_src.c_str())) {
        LOG_D("[INFO] [Shader] Direct shader source: ")
        LOG_D("%s", glsl_src.c_str())
        essl_src = glsl_src;
    } else {
        int glsl_version = getGLSLVersion(glsl_src.c_str());
        LOG_D("[INFO] [Shader] Shader source: ")
        LOG_D("%s", glsl_src.c_str())
        GLint shaderType;
        GLES.glGetShaderiv(shader, GL_SHADER_TYPE, &shaderType);
        int return_code = 0;
        essl_src = GLSLtoGLSLES(glsl_src.c_str(), shaderType, hardware->es_version, glsl_version, return_code);

        if (essl_src.empty()) {
            LOG_E("Failed to convert shader %d.", shader)
            return;
        }
        LOG_D("\n[INFO] [Shader] Converted Shader source: \n%s", essl_src.c_str())
    }
    if (!essl_src.empty()) {
        shaderInfo.id = shader;
        shaderInfo.converted = essl_src;
        const char* s[] = {essl_src.c_str()};
        GLES.glShaderSource(shader, count, s, nullptr);
        if (hardware->emulate_texture_buffer)
            shader_map_is_sampler_buffer_emulated[shader] = is_sampler_buffer_emulated;
    } else
        LOG_E("Failed to convert glsl.")
    CHECK_GL_ERROR
}

void glGetShaderiv(GLuint shader, GLenum pname, GLint* params) {
    LOG()
    GLES.glGetShaderiv(shader, pname, params);
    if (global_settings.ignore_error >= IgnoreErrorLevel::Partial && pname == GL_COMPILE_STATUS && !*params) {
        GLchar infoLog[512];
        GLES.glGetShaderInfoLog(shader, 512, nullptr, infoLog);
        LOG_W_FORCE("Shader %d compilation failed: \n%s", shader, infoLog)
        LOG_W_FORCE("Now try to cheat.")
        *params = GL_TRUE;
    }
    CHECK_GL_ERROR
}

GLuint glCreateShader(GLenum shaderType) {
    if (global_settings.fsr1_setting != FSR1_Quality_Preset::Disabled && !fsrInitialized) {
        InitFSRResources();
    }

    LOG()
    LOG_D("glCreateShader(%s)", glEnumToString(shaderType))
    GLuint shader = GLES.glCreateShader(shaderType);
    if (shader != 0 && hardware->emulate_texture_buffer) shader_map_is_sampler_buffer_emulated[shader] = false;
    CHECK_GL_ERROR
    return shader;
}