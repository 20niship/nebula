# Nebula — GPU Physics Simulator

[![CI](https://github.com/20niship/nebula/actions/workflows/ci.yml/badge.svg)](https://github.com/20niship/nebula/actions/workflows/ci.yml)
[![Perf](https://github.com/20niship/nebula/actions/workflows/perf.yml/badge.svg)](https://github.com/20niship/nebula/actions/workflows/perf.yml)
[![Pages](https://github.com/20niship/nebula/actions/workflows/pages.yml/badge.svg)](https://github.com/20niship/nebula/actions/workflows/pages.yml)
[![Project page](https://img.shields.io/badge/project%20page-20niship.github.io%2Fnebula-blue?logo=github)](https://20niship.github.io/nebula/)
[![License: MIT](https://img.shields.io/github/license/20niship/nebula)](LICENSE)
![C++20](https://img.shields.io/badge/C%2B%2B-20-blue)
![Vulkan 1.2](https://img.shields.io/badge/Vulkan-1.2-red)
![Platforms](https://img.shields.io/badge/platform-Linux%20%7C%20macOS-lightgrey)

A real-time GPU physics simulator built entirely on **Vulkan compute shaders** — cloth, soft bodies, fluids, sand, snow and smoke in one small C++20 library.

[Project page](https://20niship.github.io/nebula/) · [Examples](#examples) · [日本語](README_ja.md)

https://github.com/user-attachments/assets/fc7bfe51-39a6-4694-a515-f3288ba808cb

---

## Why nebula

- **Cross-platform** — Plain Vulkan 1.2 with no vendor-specific extensions, so it is not tied to NVIDIA or to Ubuntu. Exercised on NVIDIA (Linux), Apple GPUs (macOS via MoltenVK) and Mesa lavapipe, a CPU Vulkan driver used by CI.
- **Fast** — Everything — physics, neighbor search, rendering — stays on the GPU. On the same scenes nebula is about 2–6x faster than NVIDIA Warp (Newton) for fluid and sand ([numbers below](#performance-nebula-vs-nvidia-warp)).
- **Many solvers** — XPBD cloth, XPBD tetrahedral soft bodies, Position Based Fluids, MPM (PIC/FLIP/APIC; elastic, Von Mises, Drucker-Prager, fluid), grid-based Pyro (smoke & fire), and cloth↔fluid coupling.
- **Simple dependencies** — No CUDA, no Python runtime, no heavyweight frameworks. Just Vulkan, GLFW and GLM from your package manager; VMA, argparse, doctest and tinyobjloader are fetched by CMake / git submodules.

---

## Quick start

Ubuntu and macOS, from clone to a running scene:

```bash
# 1. dependencies
sudo apt install shaderc vulkan-headers vulkan-loader glfw3 libglm-dev   # Ubuntu
brew install shaderc vulkan-headers vulkan-loader molten-vk glfw glm     # macOS

# 2. clone + build
git clone --recursive https://github.com/20niship/nebula && cd nebula
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j

# 3. run any example (every one supports --help)
./build/mpm_avalanche
```

More detail (Taskfile, manual builds, tools): [Build details](#build-details).

---

## Performance: nebula vs NVIDIA Warp

[`benchmarks/warp/`](benchmarks/warp) runs the same scenes on nebula (Vulkan compute) and on [Newton](https://github.com/newton-physics/newton), NVIDIA's physics engine built on Warp, and reports the time per frame. Frame dt, particle count, grid cell, initial layout and gravity are shared via `scenes.py`; the solvers themselves differ (nebula: explicit MLS-MPM / XPBD, Newton: implicit MPM / XPBD).

Measured on an RTX 3060 (Warp 1.17.0 / Newton 1.7.0, warmup 30 + 100 frames, median of 3):

| Scene | Particles | nebula [ms/frame] | Warp (Newton) [ms/frame] | Warp CUDA graph [ms/frame] |
|---|---:|---:|---:|---:|
| Fluid (dam break, MPM FLIP) | 54,872 | 11.34 | 26.71 | — |
| Granular (sand column, MPM APIC) | 45,000 | 10.87 | 61.31 | — |
| Soft body (XPBD tet cube) | 4,096 | 84.35 | 17.31 | 5.92 |

Scaling with particle count and grid resolution (lower is faster):

<p align="center">
  <img src="https://github.com/user-attachments/assets/fd63cdcb-4850-4952-9232-1e2106b8a4e6" alt="nebula vs NVIDIA Warp (Newton): scaling with particle count and grid resolution" width="900">
</p>

nebula is faster on both MPM scenes and its time stays almost flat as the grid is refined, while Newton's grows. On the soft body, nebula's time grows quickly with particle count because of its O(N²) particle-collision pass, and Newton is faster. These are timings of each engine's standard solver on the same scene, not of an identical algorithm — see [`benchmarks/warp/README.md`](benchmarks/warp/README.md) for the caveats.

### Reproduce

```sh
# nebula side (headless executable; OFF by default, not built in CI)
cmake -S . -B build-bench -G Ninja -DCMAKE_BUILD_TYPE=Release -DNEBULA_BUILD_BENCH_WARP=ON
cmake --build build-bench --target nebula_bench

# Warp side (environment from pyproject.toml via uv)
cd benchmarks/warp && uv sync
uv run python run_benchmark.py   # comparison table -> out/results.md
uv run python run_sweep.py       # particle/grid sweeps -> out/sweep_log.csv, out/sweep_overview.png
```

### Example: driving an engine headlessly

`benchmarks/warp/nebula_bench.cpp` steps `MPMEngine` without a window and times each frame:

```cpp
HeadlessCtx ctx;
ctx.init();

MPMEngine e;
e.domainSize   = glm::vec3(10.0f);
e.cellSize     = 10.0f / 64.0f;
e.maxParticles = pos.size();            // non-zero disables the automatic seed block
e.init(ctx.device, ctx.allocator, ctx.descriptorPool, ctx.commandPool, ctx.computeQueue, SHADER_DIR);
e.numSubsteps = 20;
e.flip_ratio  = 0.95f;                  // FLIP; -1 = APIC, 0 = PIC
e.addForce(GravityForce::FromDirection({0, -1, 0}, 9.8f));
e.setMaterials({presetWater(1000.0f, 2e5f)});
e.appendParticles(pos, vel);            // pos.w = particle volume

VkCommandBuffer cmd = ctx.beginCmd();
e.step(cmd, 1.0f / 60.0f);
ctx.submitCmd(cmd);                     // blocks until the GPU finishes
```

The Warp side (`benchmarks/warp/warp_bench.py`) builds the matching scene with `newton.ModelBuilder` and steps `SolverImplicitMPM` / `SolverXPBD`.

---

---

## Examples

One GIF per example in `examples/`. Every example takes its parameters on the command line (`--help`: gravity, resolution, solver iterations, …); the `pyro_*` examples are headless and dump `.pvox` voxels rendered by `tools/pyro_raymarch.py`.

### Cloth (XPBD)

<table>
<tr><td width="50%" valign="top"><p align="center"><img src="https://github.com/user-attachments/assets/981c832f-1f84-4dd2-8450-532a23a7d1cf" alt="cloth_3d" width="360"></p><p align="center"><b><code>cloth_3d</code></b><br>3D cloth with stretch/bend constraints and self-collision</p></td><td width="50%" valign="top"><p align="center"><img src="https://github.com/user-attachments/assets/12299196-82ec-4357-b23f-ed5fa5ddb349" alt="cloth_scene" width="360"></p><p align="center"><b><code>cloth_scene</code></b><br>Multi-cloth scenes: stacked cloths, or one cloth wrung by rotating its corners (<code>--scene 5</code> / <code>--scene 7</code>)</p></td></tr>
</table>

### Soft body (XPBD)

<table>
<tr><td width="50%" valign="top"><p align="center"><img src="https://github.com/user-attachments/assets/b5477e95-d4f6-4aac-946e-9afbbf6f8741" alt="xpbd_softbody" width="360"></p><p align="center"><b><code>xpbd_softbody</code></b><br>XPBD tetrahedral soft bodies: a Stanford Bunny and jelly cubes dropping and colliding</p></td><td></td></tr>
</table>

### Fluid / smoke (PBF)

<table>
<tr><td width="50%" valign="top"><p align="center"><img src="https://github.com/user-attachments/assets/7bdf1dd9-9d68-469e-9db3-44f7a2f4613d" alt="fluid_pbf" width="360"></p><p align="center"><b><code>fluid_pbf</code></b><br>Position Based Fluids dam break with moving emitters and OBJ boundaries</p></td><td width="50%" valign="top"><p align="center"><img src="https://github.com/user-attachments/assets/6714e4e6-9a08-48cd-ae86-66376b896fbb" alt="fluid_sphere" width="360"></p><p align="center"><b><code>fluid_sphere</code></b><br>PBF fluid confined in a sphere-shaped OBJ boundary</p></td></tr>
<tr><td width="50%" valign="top"><p align="center"><img src="https://github.com/user-attachments/assets/b0911aab-ed9f-4a47-8a79-653f943a5c64" alt="fluid_absorb" width="360"></p><p align="center"><b><code>fluid_absorb</code></b><br>An elliptical puddle absorbed by a sweeping cylindrical port</p></td><td width="50%" valign="top"><p align="center"><img src="https://github.com/user-attachments/assets/50753770-ae62-4557-a9f7-0d67d871a14f" alt="fluid_milk_crown" width="360"></p><p align="center"><b><code>fluid_milk_crown</code></b><br>Surface tension (Akinci cohesion): a droplet falling into a puddle</p></td></tr>
<tr><td width="50%" valign="top"><p align="center"><img src="https://github.com/user-attachments/assets/2bb095f7-2001-4895-8135-d4006f1894bb" alt="fluid_wave_foam" width="360"></p><p align="center"><b><code>fluid_wave_foam</code></b><br>A wave paddle hits bunny obstacles and generates foam</p></td><td width="50%" valign="top"><p align="center"><img src="https://github.com/user-attachments/assets/9c3b2a7d-7b3b-422b-aed3-f940774d2738" alt="screw_fluid" width="360"></p><p align="center"><b><code>screw_fluid</code></b><br>A 4-blade propeller stirring PBF fluid (kinematic boundary)</p></td></tr>
<tr><td width="50%" valign="top"><p align="center"><img src="https://github.com/user-attachments/assets/7e3b28b0-562b-41c6-bb54-93abea75f008" alt="smoke" width="360"></p><p align="center"><b><code>smoke</code></b><br>Particle smoke from buoyancy + vorticity confinement</p></td><td></td></tr>
</table>

### MPM (fluid, sand, snow, elastic)

<table>
<tr><td width="50%" valign="top"><p align="center"><img src="https://github.com/user-attachments/assets/4b78ac16-f645-4733-9ebe-a2adde83fc75" alt="mpm_elastic" width="360"></p><p align="center"><b><code>mpm_elastic</code></b><br>Elastic/plastic block; switchable PIC/FLIP/APIC and Elastic/Von Mises/Drucker-Prager</p></td><td width="50%" valign="top"><p align="center"><img src="https://github.com/user-attachments/assets/9e831969-e84e-448f-b36d-8f8c86600a4a" alt="mpm_multimaterial" width="360"></p><p align="center"><b><code>mpm_multimaterial</code></b><br>Elastic jelly mixed with Drucker-Prager sand</p></td></tr>
<tr><td width="50%" valign="top"><p align="center"><img src="https://github.com/user-attachments/assets/35baada6-4d39-4d27-9f46-7c04404f3f56" alt="mpm_geolayer" width="360"></p><p align="center"><b><code>mpm_geolayer</code></b><br>Three geological layers (rock / clay / soil) collapsing under a sphere</p></td><td width="50%" valign="top"><p align="center"><img src="https://github.com/user-attachments/assets/21fe2dc3-7e4b-445e-9ef9-dce6337c7704" alt="mpm_avalanche" width="360"></p><p align="center"><b><code>mpm_avalanche</code></b><br>Drucker-Prager snow avalanche over mountainous terrain (80k particles)</p></td></tr>
<tr><td width="50%" valign="top"><p align="center"><img src="https://github.com/user-attachments/assets/1fa2ff6d-faf7-4cf6-8dc1-c22d3d14156d" alt="mpm_snow_impact" width="360"></p><p align="center"><b><code>mpm_snow_impact</code></b><br>Von Mises plastic snow block struck by a moving box collider</p></td><td width="50%" valign="top"><p align="center"><img src="https://github.com/user-attachments/assets/8c071b55-3006-4914-83a7-8ce8e7cd01f5" alt="mpm_tire_sand" width="360"></p><p align="center"><b><code>mpm_tire_sand</code></b><br>A rolling tire kicking up slipping sand</p></td></tr>
<tr><td width="50%" valign="top"><p align="center"><img src="https://github.com/user-attachments/assets/385d6789-6010-4984-8c14-a40eb6e80aa7" alt="flip_spume" width="360"></p><p align="center"><b><code>flip_spume</code></b><br>FLIP liquid scenes (geyser, outfall, culvert, slab, dual) on MPMEngine</p></td><td></td></tr>
</table>

### Pyro (grid smoke & fire)

<table>
<tr><td width="50%" valign="top"><p align="center"><img src="https://github.com/user-attachments/assets/db93238e-eb6c-4ba9-acc8-c6b87a1ebe09" alt="pyro_basic" width="360"></p><p align="center"><b><code>pyro_basic</code></b><br>Grid-based combustion/smoke with sources and a moving sphere obstacle (headless)</p></td><td width="50%" valign="top"><p align="center"><img src="https://github.com/user-attachments/assets/3d188b26-538c-43a5-8e09-027f5435a696" alt="pyro_explosion" width="360"></p><p align="center"><b><code>pyro_explosion</code></b><br>Ground-level explosion forming a mushroom cloud (headless)</p></td></tr>
<tr><td width="50%" valign="top"><p align="center"><img src="https://github.com/user-attachments/assets/bb22bbb4-3361-4610-bfad-76d2e88b60a9" alt="pyro_cow_blast" width="360"></p><p align="center"><b><code>pyro_cow_blast</code></b><br>Dense blast wave hitting a cow-shaped STL obstacle (headless)</p></td><td></td></tr>
</table>

---

## Reference

<details>
<summary>Algorithms implemented</summary>

### 1. XPBD cloth (`SimulationEngine` / `ClothSceneEngine`)

Implements Müller et al.'s **Extended Position Based Dynamics (XPBD)**. Each frame is split into `numSubsteps` substeps, each of which runs:

1. **Predict**: integrate gravity and wind to get a predicted position `predP`
2. **SDF boundary collision**: resolve collisions with the world boundary (floor/walls) via position projection
3. **Stretch/bend constraints**: distance and bending constraints with compliance `stretchCompliance` / `bendCompliance`, solved with `solverIterations` iterations of a parallel Gauss-Seidel pass using **graph 2-coloring** (adjacent edges are never updated in the same dispatch)
4. **Self-collision**: neighbor search via the spatial hash grid (below), with density-based repulsion to prevent the cloth from passing through itself
5. **Velocity update**: recompute velocity from `(predP - P) / dt` and apply linear damping

`ClothSceneEngine` layers multiple cloths and time-varying pin constraints (`PinAnimated`) on top of this solver, e.g. rotating four corners to "wring" a cloth.

### 2. XPBD tetrahedral soft bodies (`SoftBodyEngine`)

Solves two XPBD constraint types on a tetrahedral mesh (a custom `.sb` format produced by `tools/gen_softbody.py`): edge constraints (stretch) and tetrahedron constraints (volume preservation). Edges and tets are each graph-colored for parallel iteration, and particle-particle collision (`particleCollisionRadius`) between multiple instances is also resolved.

### 3. Position Based Fluids (`FluidEngine`)

A near-faithful implementation of Macklin & Müller's **PBF** (2013):

1. Predict + SDF boundary collision
2. **Spatial hash construction** (count → local prefix scan → global scan → base-offset add → sort, a 5-pass counting sort) to find neighboring particles
3. **Density constraint**: evaluate density with a Poly6 kernel and solve for the Lagrange multiplier λ with **CFM relaxation** (eq. 11, `cfmEpsilon`) to get a position correction `Δp`. **Artificial pressure (s_corr, eq. 13)** suppresses particle clustering (tensile instability). Repeated `pbfIterations` times
4. Reapply SDF collision, update velocity
5. **Vorticity confinement** (eqs. 15–16, `vorticityEnabled`/`vorticityEpsilon`): reinforces vortices lost to numerical dissipation
6. **XSPH viscosity** (`viscosityC`): averages neighboring velocities for smoother, more viscous-looking flow

Boundaries can be loaded as particle-sampled OBJ meshes (via `tinyobjloader`) or generated procedurally (cylinders, rectangular frames, a 4-blade propeller, etc.). **Kinematic boundaries** (a rotating screw, a moving cylinder) upload their position/velocity to the GPU every frame. The `smoke` scene repurposes the same solver with `pbfIterations=0` (no incompressibility constraint) plus buoyancy and vorticity confinement for a lightweight smoke look. `fluid_absorb` adds a dedicated pass that stochastically absorbs particles inside shapes (cylinder/capsule/etc., via `AbsorberDesc`).

### 4. Material Point Method (`MPMEngine`)

A hybrid Lagrangian-Eulerian **MPM** solver. Rather than a dense grid, it rebuilds a Morton-sorted sparse grid every frame using the same **spatial hash** as the fluid solver:

1. Zero the grid/hash buffers → build the spatial hash (5 passes, shared logic with the fluid solver)
2. **P2G (Particle-to-Grid)**: scatter each particle's mass and momentum to grid nodes using B-spline weights. An APIC (Affine Particle-in-Cell) momentum-gradient matrix `B` preserves angular momentum
3. **Grid update**: normalize by mass, apply gravity and wall boundary conditions
4. **Boundary conditions**: a NanoVDB-style arbitrary-shape SDF (terrain, moving obstacles) plus analytic colliders (plane/sphere/box/capsule, with velocity for moving colliders)
5. **G2P (Grid-to-Particle)**: gather grid velocities back to particles (continuously blending **PIC/FLIP/APIC** via `flip_ratio`: 0=PIC, -1=APIC, 0–1=FLIP blend), update the deformation gradient `F`, compute stress per the assigned material model, and advance particle positions

**Material models** (a `MaterialParams` table, switchable per particle via a material ID):
- `ELASTIC`: Hencky elasticity / fixed-corotated (jelly, etc.)
- `VON_MISES`: metal-like plasticity with yield stress `q_max`
- `DRUCKER_PRAGER`: sand/soil model with friction angle `M_friction` and cohesion `q_cohesion` (used for the avalanche and geo-layer-collapse demos)
- `GRANULAR_POWDER` / `FLUID` / `VISCOPLASTIC_MUD`: additional presets for granular powder, weakly-compressible fluid, and viscoplastic mud

Multiple materials can coexist per particle within a single scene — e.g. three layers in `mpm_geolayer`, or two in `mpm_multimaterial`.

### 5. Grid-based smoke/fire (Pyro) solver (`PyroEngine`)

Unlike the particle-based MPM/PBF solvers, this is an **Eulerian grid method** in the spirit of Houdini Pyro. Density, temperature, fuel, flame, and velocity are solved directly on a dense Morton-ordered grid, double-buffered (A/B):

1. **Source injection** (density, temperature, fuel, inflow velocity)
2. **Combustion reaction**: fuel above `ignitionTemp` burns at `burnRate`, producing `heatRelease` (temperature rise), `smokeYieldPerFuel` (smoke generation), and `flameBrightness` (emission)
3. **Buoyancy**: rise from temperature (`buoyancyAlpha`) and sink from density/weight (`buoyancyBeta`)
4. **Vorticity confinement**: computes curl and adds a confinement force along its gradient to preserve turbulent detail
5. **Obstacle SDF boundary conditions**: zeroes velocity against an SDF built from an arbitrary mesh (STL)
6. **Pressure projection** (incompressibility): computes divergence and solves the Poisson equation with **Jacobi iteration** (`numJacobiIters` iterations) to remove divergence from the velocity field
7. **Advection**: semi-Lagrangian advection of every channel from buffer A to buffer B

It runs headless — with no particles or mesh to render — dumping every channel (density/temperature/fuel/flame/velocity/sdf) to a custom `.pvox` binary format, which is ray-marched and rendered offline in Python (`tools/pyro_raymarch.py`). Obstacle STLs can be rebuilt into an SDF at any interval via `MeshSDF.h`'s `buildMeshSDF()`, supporting moving/rotating obstacles (`pyro_basic`).

### 6. Bidirectional cloth↔fluid coupling (`MultiPhysicsEngine`)

Combines the XPBD cloth solver and the PBF fluid solver into a single buffer set and GPU pipeline, adding a **coupling force pass** (`kCouplingCloth_`) between them. `enableCoupling` toggles the two-way interaction (fluid pushing cloth / cloth pushing fluid) on or off.

### Shared infrastructure: the spatial hash grid

A neighbor-search algorithm shared by cloth, fluid, MPM, and the 2D string demo. The world is divided into `grid_res`³ cells, and a per-cell particle list is built with a 5-pass **counting sort**: `hash_count` (count particles per cell) → `hash_scan_local` (per-workgroup prefix sum) → `hash_scan_global` (a single cross-workgroup prefix sum) → `hash_add_base` (add the global offset) → `hash_sort` (reorder particles by cell). This gives O(N) neighbor search on the GPU via a Blelloch-style scan.

---

</details>

---

## Architecture

```
src/
  core/          Vulkan context, compute pipelines, attribute buffers,
                 material parameters, colliders, MeshSDF, source emitters
  engine/        SimulationEngine (XPBD cloth), ClothSceneEngine, SoftBodyEngine,
                 FluidEngine (PBF), MPMEngine, PyroEngine, MultiPhysicsEngine
  graphics/      Render pipelines, cloth renderer
shaders/         GLSL compute & graphics shaders (compiled to SPIR-V)
examples/        Per-scene entry points (18 scenes)
tests/           doctest unit tests, headless-execution helpers (HeadlessCtx)
tools/           .sb soft-body generation, STL terrain/cow/sphere generation, Pyro ray-march viewer (Python)
(assets are generated into the build directory by tools/gen_assets.py)
```

**Key technologies:**
- Vulkan 1.2 (MoltenVK on macOS)
- GLSL compute shaders compiled with `glslc`
- VulkanMemoryAllocator (VMA) for GPU memory
- GLFW for windowing
- GLM for math
- tinyobjloader for boundary mesh loading

---

## Requirements

| Tool | Version |
|---|---|
| CMake | ≥ 3.22 |
| C++ compiler | C++20 (clang / GCC) |
| Vulkan SDK | 1.3.280.1 (LunarG or Homebrew) |
| Task | latest |
| GLFW, GLM | via Homebrew or apt install |
| Python 3 | for the `tools/` scripts, `pyro_raymarch.py`, and `capture_sim.py` |

Developed on macOS (MoltenVK) and Linux (Ubuntu 24.04; CI uses Mesa lavapipe, benchmarks on an NVIDIA GPU). Windows is untested.

---

## Build details

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

# Run scenes registered in the Taskfile
task run:cloth     # cloth_3d
task run:fluid     # fluid_pbf
task run:multi     # multi_physics (cloth + fluid coupling)

# All other scenes (MPM, Pyro, softbody, etc.) run directly after building
./build/mpm_avalanche
./build/pyro_explosion --help   # every scene supports --help via argparse
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
task build              # Build all targets
task build:cloth        # Build cloth_3d only
task build:fluid        # Build fluid_pbf only
task build:multi        # Build multi_physics only
task build:smoke        # Build smoke only
task build:cloth-scene  # Build cloth_scene only
task build:absorb       # Build fluid_absorb only
task build:softbody     # Build xpbd_softbody only
task test               # Run unit tests
task shaders            # Recompile shaders only
task capture            # Run the per-test-case simulations and generate a grid video (sim_captures/simulation_results.mp4)
task clean              # Remove build directory
```

Executables not listed above (`mpm_*`, `pyro_*`, `screw_fluid`, `string_2d`, etc.) are still built by `task build`; launch them directly from `./build/<target>`.

---

## Simulation Parameters (cloth_3d example)

| Parameter | Default | Description |
|---|---|---|
| `--cloth-n` | 128 | Grid resolution (N×N particles) |
| `--world-size` | 10.0 | Simulation world size |
| `--grid-res` | 64 | Spatial hash grid resolution |
| `--dt` | 1/60 | Timestep (seconds) |

Scene-specific parameters (material constants, vorticity confinement strength, Jacobi iteration count, etc.) can be found by passing `--help` to any executable.

---

## License

See [LICENSE](LICENSE).
