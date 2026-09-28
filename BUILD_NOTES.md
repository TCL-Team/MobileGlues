# MobileGlues 26.3 fork — first compatibility pass

## Main change
Minecraft 26.3 changed the OpenGL shader path: it uses ShaderC and SPIR-V for OpenGL. This fork adds a MobileGlues-side SPIR-V bridge:

- intercepts `glShaderBinary(GL_SHADER_BINARY_FORMAT_SPIR_V)`;
- stores the SPIR-V module per shader object;
- implements `glSpecializeShader()` instead of the old no-op stub;
- applies specialization constants through SPIRV-Cross;
- translates SPIR-V to GLSL ES and compiles it on the GLES driver;
- advertises `GL_ARB_gl_spirv` only when the opt-in 26.3 profile is enabled;
- reports SPIR-V as the supported shader binary format in that profile.

## Compatibility profile

Add this to `/sdcard/MG/config.json`:

```json
"mc26_3Compat": 1
```

The profile also enables full shader/program error ignoring, disables DSA, and selects conservative MultiDraw fallbacks.

## Important

This source snapshot does not contain the MobileGlues CMake submodules (`glslang`, `SPIRV-Cross`, `flat_hash_map`, `perfetto`, `xxhash`), so the native library cannot be built from this uploaded snapshot alone. Initialize the original repository's submodules before building.

This pass is a source-level fix and needs to be tested on the Adreno 710 with a fresh GLSL cache after the native library is rebuilt.
