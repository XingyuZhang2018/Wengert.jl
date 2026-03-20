# Wengert.jl — Rename & Documentation Design

**Goal:** Rename package from YAADE → Wengert, write English README, publish to GitHub.

**Scope:**
- File renames: `src/YAADE.jl` → `src/Wengert.jl`, `ext/YAADECUDAExt.jl` → `ext/WengertCUDAExt.jl`
- Content updates: Project.toml, all test files, module names
- New file: `README.md` in English
- GitHub: create public repo `Wengert.jl`, push

**Directory name:** Keep as `YAADE.jl/` (no git history disruption).

---

## README structure

1. Title + one-line description
2. Background / motivation (Zygote limitation)
3. Installation
4. Quick start (scalar, array, @checkpoint)
5. API reference
6. GPU checkpointing explanation
7. Performance notes
8. Implementation notes (Wengert list, 1964 paper)
