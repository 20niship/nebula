# Nebula — GPU Physics Simulator

A real-time GPU-accelerated physics simulator built with **Vulkan compute shaders** and **XPBD (Extended Position Based Dynamics)**. Supports cloth, fluid (PBF), smoke, and bidirectional cloth-fluid coupling, all running entirely on the GPU.

<div><video controls src="assets/sample.mp4" muted="false"></video></div>

---

## Features

| Scene | Description |
|---|---|
| `cloth_3d` | 3D cloth with stretch / bend constraints and self-collision |
| `string_2d` | 2D string simulation for self-collision validation |
| `fluid_pbf` | Position Based Fluids with vorticity confinement and viscosity |
| `fluid_sphere` | PBF fluid inside a sphere boundary |
| `smoke` | Buoyant smoke particle simulation |
| `screw_fluid` | Rotating screw immersed in PBF fluid |
| `cloth_scene` | Multi-cloth scenes (two-cloth / four-corner twist) |
| `multi_physics` | Bidirectional cloth ↔ fluid coupling |

All scenes expose **real-time parameter tweaking** via an ImGui panel (gravity, compliance, wind, solver iterations, …).

---

## Architecture

```
src/
  core/          Vulkan context, compute pipelines, attribute buffers
  engine/        XPBD cloth engine, PBF fluid engine, multi-physics engine
  graphics/      Render pipelines, cloth renderer
shaders/         GLSL compute & graphics shaders (compiled to SPIR-V)
examples/        Per-scene entry points
tests/           doctest unit tests
```

**Key technologies:**
- Vulkan 1.2 (MoltenVK on macOS)
- GLSL compute shaders compiled with `glslc`
- VulkanMemoryAllocator (VMA) for GPU memory
- ImGui + GLFW for UI and windowing
- GLM for math

---

## Requirements

| Tool | Version |
|---|---|
| CMake | ≥ 3.22 |
| C++ compiler | C++20 (clang / GCC) |
| Vulkan SDK | 1.3.280.1 (LunarG or Homebrew) |
| Task | latest |
| GLFW, GLM | via Homebrew or apt install |

---

## Getting Started

### 1. Install dependencies

```bash
# macOS
brew install shaderc vulkan-headers vulkan-loader molten-vk glfw glm go-task

# Ubuntu
sudo apt update
sudo apt install shaderc vulkan-headers vulkan-loader glfw3 libglm-dev golang-go
```

### 2. Clone and initialize submodules

```bash
git clone <repo-url>
cd nebula
git submodule update --init --recursive
```

### 3. Build and run

```bash
# Build everything
task build

# Run individual scenes
task run:cloth     # cloth_3d
task run:fluid     # fluid_pbf
task run:multi     # multi_physics (cloth + fluid coupling)
```

### Manual CMake build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
./build/multi_physics
```

---

## Task Reference

```bash
task build           # Build all targets
task build:cloth     # Build cloth_3d only
task build:fluid     # Build fluid_pbf only
task build:multi     # Build multi_physics only
task test            # Run unit tests
task shaders         # Recompile shaders only
task capture         # Render all test-case videos → sim_captures/
task clean           # Remove build directory
```

---

## Simulation Parameters (cloth_3d)

| Parameter | Default | Description |
|---|---|---|
| `--cloth-n` | 128 | Grid resolution (N×N particles) |
| `--world-size` | 10.0 | Simulation world size |
| `--grid-res` | 64 | Spatial hash grid resolution |
| `--dt` | 1/60 | Timestep (seconds) |

---

## License

See [LICENSE](LICENSE).
