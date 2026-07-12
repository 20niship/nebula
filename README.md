# Nebula — GPU Physics Simulator

A real-time GPU physics simulator built entirely with **Vulkan compute shaders**. It covers XPBD cloth, XPBD soft bodies, Position Based Fluids (PBF), the Material Point Method (MPM), a grid-based smoke/fire (Pyro) solver, and bidirectional cloth↔fluid coupling — all running on the GPU.

https://github.com/user-attachments/assets/fc7bfe51-39a6-4694-a515-f3288ba808cb

---

## Scenes

| Executable | Engine | Description |
|---|---|---|
| `cloth_3d` | SimulationEngine | 3D cloth with stretch/bend constraints and self-collision (free fall) |
| `cloth_scene` | ClothSceneEngine | Multi-cloth scenes (`--scene 5` = two stacked cloths / `--scene 7` = a single cloth "wrung" by rotating its 4 corners about a central axis) |
| `string_2d` | (standalone) | 2D string for self-collision validation (XPBD distance constraints, a harness that auto-exits after 3s) |
| `xpbd_softbody` | SoftBodyEngine | XPBD tetrahedral soft bodies (a Stanford Bunny + 9 jelly cubes dropping and colliding) |
| `fluid_pbf` | FluidEngine | Position Based Fluids with vorticity confinement and XSPH viscosity. `dam-break` / `source-flow` scenarios, dynamic loading of arbitrary OBJ boundaries |
| `fluid_sphere` | FluidEngine | PBF fluid confined inside a sphere-shaped OBJ boundary |
| `fluid_absorb` | FluidEngine | An elliptical puddle absorbed by a cylindrical "absorber port" that sweeps across it along X |
| `screw_fluid` | FluidEngine | A procedurally generated 4-blade propeller rotating inside PBF fluid, stirring it |
| `smoke` | FluidEngine (repurposed) | Particle-based smoke driven by buoyancy + vorticity confinement (no incompressibility constraint) |
| `multi_physics` | MultiPhysicsEngine | Cloth and PBF fluid sharing one world, with bidirectional coupling toggled on/off |
| `mpm_elastic` | MPMEngine | An elastic/plastic block falling; switchable PIC/FLIP/APIC transfer modes and Elastic/Von Mises/Drucker-Prager plasticity models |
| `mpm_multimaterial` | MPMEngine | Elastic "jelly" mixed with Drucker-Prager sand in the same simulation |
| `mpm_geolayer` | MPMEngine | Geological-layer collapse (hard rock = elastic / weak clay = Von Mises / loose soil = Drucker-Prager, three layers pushed by a sphere collider) |
| `mpm_avalanche` | MPMEngine | A Drucker-Prager snow avalanche over mountainous terrain (ridges + a couloir), 80k particles, with a runaway-velocity detector in the UI |
| `mpm_snow_impact` | MPMEngine | A Von Mises plastic snow block struck by a horizontally moving box collider |
| `pyro_basic` | PyroEngine | Basic grid-based combustion/smoke demo with multiple fire/smoke sources and a moving sphere obstacle (headless) |
| `pyro_explosion` | PyroEngine | A ground-level explosive burn aiming to form a "mushroom cloud" shape (headless) |
| `pyro_cow_blast` | PyroEngine | An ultra-dense, high-speed smoke blast wave hitting a cow-shaped STL obstacle (headless) |

Every windowed scene exposes **real-time parameter tweaking via an ImGui panel** (gravity, compliance, wind, solver iterations, and more). The `pyro_*` scenes have no window/ImGui — they run headless and dump `.pvox` voxel files for offline ray-march rendering via `tools/pyro_raymarch.py` (Python).

---

## Algorithms implemented

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
assets/          Assets such as bunny.obj, cow_obstacle.stl, sphere_obstacle.stl
```

**Key technologies:**
- Vulkan 1.2 (MoltenVK on macOS)
- GLSL compute shaders compiled with `glslc`
- VulkanMemoryAllocator (VMA) for GPU memory
- ImGui + GLFW for UI and windowing
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

**Currently macOS only** (uses MoltenVK / Metal). Linux should work but is untested.

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
