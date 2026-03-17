# CI & Coverage Design

**Goal:** Add GitHub Actions CI (CPU + GPU) and Codecov coverage to Wengert.jl.

## Workflows

### CI.yml (GitHub-hosted, CPU)
- Trigger: push/PR on master
- Matrix: Julia 1.9 + 1 (latest) × ubuntu-latest
- Steps: checkout → setup-julia → cache → test with coverage → upload to Codecov
- Tools: julia-actions/setup-julia, julia-actions/cache, julia-actions/julia-runtest, julia-actions/julia-processcoverage, codecov/codecov-action@v4

### GPU.yml (self-hosted, GPU)
- Trigger: push/PR on master + workflow_dispatch
- runs-on: [self-hosted, gpu]
- Steps: checkout → Pkg.instantiate → test_gpu_checkpoint.jl → test_gpu_memory_pressure.jl (timing only)
- OOM mode excluded (exhausts VRAM)

## Coverage
- Codecov: upload lcov.info from CPU test run
- Token stored as GitHub secret CODECOV_TOKEN

## README Badges
- CI badge (CI.yml)
- Coverage badge (Codecov)
- Added to top of README.md

## Self-hosted Runner
- Register at: GitHub repo → Settings → Actions → Runners → New self-hosted runner
- Label: gpu
- Runs on the RTX 4090 machine
