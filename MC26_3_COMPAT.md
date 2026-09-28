# Minecraft 26.3 compatibility profile

This development patch adds an opt-in `mc26_3Compat` setting to MobileGlues.

```json
{
  "mc26_3Compat": 1
}
```

When enabled it:

- honors an existing `config.json` even when the launcher does not export legacy launcher environment variables;
- enables MobileGlues full shader/program error ignoring (`enableNoError` is effectively Level 2);
- disables the DSA compatibility layer for the profile;
- selects conservative MultiDraw backends: unroll for regular MultiDraw and one-draw-per-command indirect for the two indirect entry points.

This is a compatibility/diagnostic profile, not a performance profile. It may be slower than the normal batched paths.
