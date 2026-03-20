# CI & Coverage Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add GitHub Actions CI (CPU on GitHub-hosted + GPU on self-hosted) and Codecov coverage reporting with README badges.

**Architecture:** Two independent workflow files under `.github/workflows/`. `CI.yml` runs the 55-test CPU suite on GitHub-hosted runners across Julia 1.9 and latest, uploads coverage to Codecov. `GPU.yml` runs the two GPU test scripts on a self-hosted runner labelled `gpu` (the user's RTX 4090 machine).

**Tech Stack:** GitHub Actions, julia-actions suite (setup-julia v2, cache v2, julia-buildpkg v1, julia-runtest v1, julia-processcoverage v1), codecov/codecov-action v4, Codecov.io.

---

### Task 1: Create CPU CI workflow

**Files:**
- Create: `.github/workflows/CI.yml`

**Step 1: Create directory and file**

```bash
mkdir -p ".github/workflows"
```

Create `.github/workflows/CI.yml` with this exact content:

```yaml
name: CI

on:
  push:
    branches: [master]
  pull_request:
    branches: [master]

jobs:
  test:
    name: Julia ${{ matrix.version }} — ${{ matrix.os }}
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        version:
          - '1.9'
          - '1'
        os:
          - ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - uses: julia-actions/setup-julia@v2
        with:
          version: ${{ matrix.version }}

      - uses: julia-actions/cache@v2

      - uses: julia-actions/julia-buildpkg@v1

      - uses: julia-actions/julia-runtest@v1
        with:
          coverage: true

      - uses: julia-actions/julia-processcoverage@v1

      - uses: codecov/codecov-action@v4
        with:
          files: lcov.info
          token: ${{ secrets.CODECOV_TOKEN }}
          fail_ci_if_error: false
```

**Step 2: Verify file structure**

```bash
cat ".github/workflows/CI.yml"
```

Expected: file exists, YAML is well-formed.

**Step 3: Commit**

```bash
git add .github/workflows/CI.yml
git commit -m "ci: add CPU CI workflow with Codecov coverage"
```

---

### Task 2: Create GPU workflow

**Files:**
- Create: `.github/workflows/GPU.yml`

**Step 1: Create GPU.yml**

Create `.github/workflows/GPU.yml` with this exact content:

```yaml
name: GPU

on:
  push:
    branches: [master]
  pull_request:
    branches: [master]
  workflow_dispatch:

jobs:
  gpu-test:
    name: GPU Tests (RTX 4090, self-hosted)
    runs-on: [self-hosted, gpu]

    steps:
      - uses: actions/checkout@v4

      - name: Instantiate project
        run: julia --project=. -e "using Pkg; Pkg.instantiate()"

      - name: GPU checkpoint tests
        run: julia --project=. test/test_gpu_checkpoint.jl

      - name: GPU timing tests (no OOM)
        run: julia --project=. test/test_gpu_memory_pressure.jl timing
```

Note: OOM mode (`oom`) is intentionally excluded — it exhausts VRAM and would break subsequent steps. The `timing` argument runs only the timing + correctness section.

**Step 2: Commit**

```bash
git add .github/workflows/GPU.yml
git commit -m "ci: add GPU workflow for self-hosted RTX 4090 runner"
```

---

### Task 3: Add README badges

**Files:**
- Modify: `README.md`

**Step 1: Read current README top**

Read `README.md` lines 1–5 to see current title/description.

**Step 2: Insert badges after title**

The current README starts with:
```markdown
# Wengert.jl

A tape-based reverse-mode...
```

Change it to:
```markdown
# Wengert.jl

[![CI](https://github.com/XingyuZhang2018/Wengert.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/XingyuZhang2018/Wengert.jl/actions/workflows/CI.yml)
[![GPU](https://github.com/XingyuZhang2018/Wengert.jl/actions/workflows/GPU.yml/badge.svg)](https://github.com/XingyuZhang2018/Wengert.jl/actions/workflows/GPU.yml)
[![Coverage](https://codecov.io/gh/XingyuZhang2018/Wengert.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/XingyuZhang2018/Wengert.jl)

A tape-based reverse-mode...
```

**Step 3: Commit**

```bash
git add README.md
git commit -m "docs: add CI and coverage badges to README"
```

---

### Task 4: Push to GitHub and add Codecov secret

**Step 1: Push all commits**

```bash
git push origin master
```

**Step 2: Add CODECOV_TOKEN secret to GitHub repo**

The user must do this manually (secrets cannot be set via API without admin token scope):

1. Go to https://codecov.io → sign in with GitHub → add the `XingyuZhang2018/Wengert.jl` repo
2. Copy the repository upload token from Codecov
3. Go to https://github.com/XingyuZhang2018/Wengert.jl/settings/secrets/actions
4. Click "New repository secret"
5. Name: `CODECOV_TOKEN`, Value: paste token from Codecov
6. Click "Add secret"

**Step 3: Register self-hosted runner (user must do manually)**

1. Go to https://github.com/XingyuZhang2018/Wengert.jl/settings/actions/runners
2. Click "New self-hosted runner"
3. Select Linux (or Windows, matching the RTX 4090 machine OS)
4. Follow the instructions to download and configure the runner on that machine
5. When asked for labels, add: `gpu`
6. Start the runner: `./run.sh` (Linux) or `./run.cmd` (Windows)

The runner must be running before GPU workflow jobs can execute.

**Step 4: Verify CI triggers**

After pushing, check:
- https://github.com/XingyuZhang2018/Wengert.jl/actions

Expected: CI workflow starts automatically on `ubuntu-latest` for Julia 1.9 and 1. GPU workflow will queue but wait for a self-hosted runner to come online.
