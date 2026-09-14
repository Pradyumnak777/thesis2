# MuJoCo + humanoid control — concepts notes

Personal reference for `unitreeg1_sim.py`. Written while going from "no robotics background"
to getting a PD-tracked G1 falling over in a physically-plausible way (which, read below, is
actually a meaningful milestone, not just a bug).

---

## 1. Two fundamentally different things: kinematic playback vs. physics simulation

`unitreeg1_sim.py` has two functions that both take retargeted motion and put it in the
simulator, but they do categorically different things.

### `kinematic_playback()` — teleportation, no physics

Every frame, it **directly overwrites** `data.qpos` with the retargeted joint angles, sets
`data.qvel[:] = 0.0`, then calls `mujoco.mj_forward(model, data)`.

- `mj_forward` does **not** advance time or integrate anything. It only recomputes quantities
  *derived* from whatever `qpos` currently holds — body positions in world space, geometry
  transforms, contact points — so the renderer has something correct to draw. Think of it like
  hitting "recalculate" in a spreadsheet after manually typing a value into a cell: it propagates
  that one change through dependent formulas, but the clock doesn't move.
- `qvel` is zeroed every frame because there's no real velocity here — the robot's pose is being
  *dictated*, frame by frame, like posing a mannequin. No gravity, no contact forces, no
  actuator torques exist in this mode at all.

Good for: sanity-checking that a retargeted trajectory *looks* right. Useless for: telling you
whether that trajectory is something a real robot's motors and legs could physically produce.

### `physics_simulation()` — real forward dynamics

Every physics tick, `mujoco.mj_step(model, data)` does the actual simulation:

1. Look at current `qpos`, `qvel`, and `ctrl` (actuator commands).
2. Compute every force acting on the system: gravity, contact/collision forces, joint passive
   forces, and actuator forces derived from `ctrl`.
3. Solve the equations of motion for acceleration (`qacc`).
4. **Integrate**: `qvel += qacc * dt`, `qpos += qvel * dt`, where `dt = model.opt.timestep`.

Now nothing is teleported — you can only influence the robot by writing torque commands into
`data.ctrl` before each `mj_step` call, and physics (including gravity and ground contact)
determines what actually happens to `qpos`/`qvel` as a result.

---

## 2. `qpos` vs `qvel`: why the array sizes don't match (`nq` vs `nv`)

`model.nq = 36`, `model.nv = 35`, `model.nu = 29`. Three different counts, three different things:

| | size | what it is |
|---|---|---|
| `nq` | 36 | length of `qpos` — generalized **positions** |
| `nv` | 35 | length of `qvel` — generalized **velocities** |
| `nu` | 29 | number of **actuators** (motors) |

The `nq`/`nv` mismatch comes from the pelvis's `<freejoint>` — the root body floating freely in
3D. Its position needs **7** numbers (3 for XYZ translation + **4** for an orientation
quaternion `[w,x,y,z]`), but its velocity only needs **6** (3 linear + 3 angular) — a quaternion
has 4 components but only 3 real degrees of rotational freedom, so there's no meaningful 4-wide
"quaternion velocity." One fewer DOF in velocity space than position space, for every free joint
in a model. This is why the same 29 actuated joints live at different offsets:

```python
q_now  = data.qpos[7:36]   # 29 actuated joint angles (root takes slots 0:7)
qd_now = data.qvel[6:35]   # same 29 joints' velocities (root takes slots 0:6)
```

Mixing these up (e.g. using `qvel[7:36]`) silently reads the wrong joints' velocities — an easy,
quiet bug.

---

## 3. Actuators: `ctrl`, `gear`, and why raw PD torque broke everything

### What a joint vs. an actuator actually is

A **joint** (`<joint>` in the XML) just describes that a hinge *can* rotate, and its range — a
kinematic/geometric fact about the skeleton. It doesn't move anything by itself. An **actuator**
(`<motor>`) is what actually drives that hinge — in the real G1 this is a physical electric motor;
in MuJoCo it's however you interpret `data.ctrl` for that actuator.

The pelvis has a joint (`<freejoint>`) but **no actuator** — and that's not a modeling oversight,
it's physically correct. A motor pushes one link against another *across* a joint; the pelvis
isn't attached to anything to push against. This is universal to legged/mobile robots: you can
never directly command the robot's global position — the only way to move the pelvis is
*indirectly*, via leg motors pushing feet against the ground and the ground pushing back
(reaction forces). This is **underactuation**: 29 motors, but 35 velocity-space DOF — 6 of them
(the floating base) you can never touch directly. Keep this in mind for §5, it's the whole reason
naive PD fell over.

### `data.ctrl` and `gear`

`data.ctrl` is the array you write control commands into, one entry per actuator, every tick.
For a `<motor>` actuator, MuJoCo interprets it (scaled by `gear`) as torque:

```
actual_torque_applied = gear * ctrl
```

Checked this model's values directly:

```python
model.actuator_gear[:3]       # [[1,0,0,0,0,0], [1,0,0,0,0,0], [1,0,0,0,0,0], ...]
model.actuator_ctrlrange[:3]  # [[-1, 1], [-1, 1], [-1, 1], ...]
```

`gear=1` means `ctrl` and torque (N·m) are numerically identical — but `ctrlrange=[-1,1]` caps
you at **±1 N·m per joint**. The hip/knee joints need up to ~139 N·m to hold the robot up
(`actuatorfrcrange` in the XML: `-88 88`, `-139 139`, etc.). Naively writing a PD torque like
`tau=80` straight into `ctrl` gets silently clipped to `1.0` — effectively almost no torque at
all, and a very confusing bug to chase (nothing errors, it just doesn't work).

### The fix: normalize by each joint's real torque limit

```python
joint_ids  = model.actuator_trnid[:, 0]        # actuator_trnid: which joint does each actuator drive?
max_torque = model.jnt_actfrcrange[joint_ids, 1]  # that joint's max |torque|, from the XML
...
data.ctrl[:] = np.clip(tau / max_torque, -1.0, 1.0)
```

`model.actuator_trnid` is shape `(nu, 2)` — "transmission ID," which body/joint/site each
actuator is wired to (column 0 is the joint ID for a simple motor transmission). `model.jnt_actfrcrange`
is shape `(njnt, 2)` — `[min, max]` torque for **every** joint in the model (root joint included).
Indexing one by the other (`jnt_actfrcrange[joint_ids, 1]`) gives, per actuator, its true max
torque in the *same order* as `data.ctrl` — so a real-unit PD torque can be turned into a valid
`[-1, 1]` fraction of what that specific joint can actually deliver, instead of one wrong global
scale.

---

## 4. The PD control loop, piece by piece

```python
tau = kp * (q_ref[i] - q_now) + kd * (qdot_ref[i] - qd_now)   # 1. decide
data.ctrl[:] = np.clip(tau / max_torque, -1.0, 1.0)            # 2. command
mujoco.mj_step(model, data)                                     # 3. execute
```

Three distinct jobs, even though they run back to back:

1. **The control law** — pure math, nothing MuJoCo-specific. `kp * (position error)` pulls the
   joint toward the target angle (proportional term). `kd * (velocity error)` pulls it toward
   the target *speed* and, importantly, acts as damping — without it the joint would overshoot
   and oscillate forever, like a frictionless spring. Using `qdot_ref` (not 0) in the D-term
   means the controller resists deviation *from the intended motion*, not the motion itself —
   otherwise it'd fight against the trajectory it's supposed to be tracking.
2. **Writing the command** — `data.ctrl` is a mailbox. Nothing physical happens yet; this just
   makes the decision visible to the simulator for the next step.
3. **`mj_step`** — the only line that actually advances time or changes robot state. Reads
   `ctrl`, combines with gravity/contact, integrates forward by `model.opt.timestep`.

**Universal pattern**: *look at current state → compute a command → apply it → step physics →
repeat.* Every control approach in §5 fits this same loop; they only differ in how step 1 is
computed.

### Reference velocity: `qdot_ref` via `np.gradient`

GMR only outputs joint **positions** (`dof_pos`), no velocities, so we approximate:

```python
dt_ref = 1.0 / fps                              # time between motion-capture frames (~0.0333s @ 30fps)
qdot_ref = np.gradient(q_ref, dt_ref, axis=0)   # ≈ dq/dt from discrete samples
```

`np.gradient` uses central differences at interior frames (`(q[i+1] - q[i-1]) / (2*dt_ref)`,
more accurate than a one-sided difference) and falls back to forward/backward differences at the
two edges. It's a real approximation, not ground truth — if `q_ref` is noisy, differentiating
amplifies that noise into a jerkier `qdot_ref`; smoothing `q_ref` first is the standard fix if
tracking looks jittery.

### Two clocks: `dt_ref` vs `physics_dt`, and substepping

```python
physics_dt = model.opt.timestep                     # 0.002s — MuJoCo's own integration step
substeps   = max(1, round(dt_ref / physics_dt))      # ≈ 17 physics steps per mocap frame
```

`dt_ref` (mocap sample spacing, ~0.033s) and `physics_dt` (MuJoCo's integration step, 0.002s)
come from completely unrelated sources and don't match. Physics needs small steps to stay
numerically stable (contact especially — a foot hitting the ground is fast and stiff; too large
a step and the solver can miss it or wildly overcorrect, the classic "simulation explodes"
failure). So for every one target frame `q_ref[i]`, the loop runs `mj_step` ~17 times before
moving to `q_ref[i+1]` and rendering — the controller recomputes `tau` fresh on every one of
those 17 finer-grained steps, even though the target itself only updates once per rendered frame.

### One-time setup vs. per-frame reset

```python
mujoco.mj_resetData(model, data)   # clear any stale state
data.qpos[...] = <frame 0 pose>
data.qvel[:] = 0.0                 # initial condition: starts at rest
mujoco.mj_forward(model, data)     # make derived quantities (contacts, body transforms) consistent
                                    # with the pose we just wrote, before physics starts stepping
```

This mirrors `kinematic_playback`'s `qvel[:] = 0.0` / `mj_forward` calls, but the *intent* is
opposite: there it runs every frame (repeatedly discarding velocity, because we're teleporting
forever). Here it runs **once**, before the loop starts — after that, `mj_step` owns `qvel`
completely; zeroing it every frame here would erase all momentum every tick and break the
simulation (a falling robot would never even accelerate).

---

## 5. Why it falls over anyway, and what robotics actually does about it

### The observed symptom

Smoke-testing the PD loop above: `ctrl` saturates at `±1.0` (max available torque) almost
immediately, and the robot's root height collapses from `0.72` → `0.11` within half a second,
tracking error climbing the whole time. **This isn't "not enough torque"** — torque was already
maxed out and it still fell. The real issue is structural.

### The real issue: coordination, not magnitude

PD control here is 29 *independent* spring-dampers — the knee doesn't know what the ankle is
doing, no joint knows where the center of mass is relative to the feet. But standing/walking is
a *coordination* problem: the only lever on the (unactuated) pelvis is the combined effect of all
leg joints producing the right ground reaction forces to keep the center of mass over the support
foot (see §3's underactuation point). You can have max torque in every single joint and still
fall, because torque magnitude was never the bottleneck — nothing in the PD formula has any
notion of "stay balanced."

### The general landscape (walking example: right foot planted, left leg swinging forward)

Every one of these answers the same question — *what torque should each of the 29 motors output,
right now?* — differently. Crucially: **only the last one is "learning" in the ML/training
sense.** The rest are classical control/optimization using known physics equations, computed
fresh every tick, no training data or gradient descent involved anywhere.

| # | Approach | What decides the torque | "Learned"? |
|---|---|---|---|
| 1 | **PD control** (what we built) | Fixed formula per joint, independently: `kp*err + kd*errdot`. No knowledge of other joints, contact, or balance. | No — static formula |
| 2 | **Computed torque / feedforward+feedback** | Solve the robot's known equations of motion *forward*-in-reverse ("inverse dynamics": given the mass/inertia distribution and the acceleration I want, Newton's laws say I need *this* torque"), computed fresh each tick from current state. PD only mops up small residual error on top. | No — analytic physics, no data |
| 3 | **Whole-body control (QP)** | Pose one optimization every tick across *all* 29 torques jointly: best track [swing foot target] + [keep CoM over support foot] + [don't violate contact constraints — stance foot can push, never pull]. A numerical solver (Quadratic Program) answers this from scratch, ~500–1000×/sec on real hardware. | No — real-time numerical optimization |
| 4 | **Trajectory optimization / MPC** | Same idea as #3, but over a whole future window (or the entire motion) at once, offline or on a sliding horizon — can even revise the reference trajectory itself ("place the foot 2cm left, that's more stable") rather than trusting the retargeted motion blindly. | No — optimization over a horizon |
| 5 | **Reinforcement learning** | Train a neural network (via reward = tracking accuracy + staying upright, millions of simulated attempts, domain-randomized) to map `(state, reference target) → torque`. At deployment: one fast forward pass per tick, no runtime solve. | **Yes** — network weights learned via gradient descent on simulated experience |

**The one-line version**: #1–4 are "a human derives the right formula/optimization from known
physics, evaluate it live." #5 is "train a function to approximate good behavior from
experience, then evaluate that trained function live." GMR-style retargeting pipelines like this
one are, in current practice, most often the *front end* feeding reference motions into a #5-style
RL training loop — worth keeping in mind if this side-quest continues past "get something
physically plausible on screen."

---

## 6. MuJoCo API quick reference (everything used so far)

| Call / attribute | What it does |
|---|---|
| `mujoco.MjModel.from_xml_path(path)` | Parse the XML into a `model` — the static description (bodies, joints, actuators, meshes). Doesn't change over time. |
| `mujoco.MjData(model)` | The mutable simulation state for a given model — `qpos`, `qvel`, `ctrl`, and everything derived from them. This is what changes every step. |
| `mujoco.mj_forward(model, data)` | Recompute all *derived* quantities (body world positions, contacts, sensor readings) from the current `qpos`/`qvel`/`ctrl`, **without** advancing time. |
| `mujoco.mj_step(model, data)` | Full forward-dynamics step: compute forces (gravity, contact, actuators) → solve for acceleration → integrate `qvel`/`qpos` forward by `model.opt.timestep`. Advances time. |
| `mujoco.mj_resetData(model, data)` | Clear a `data` object back to a default/zero state — used once before setting a custom initial condition, so no stale values linger from a previous run. |
| `mujoco.Renderer(model, height, width)` | Offscreen renderer object — turns a `data` state into pixels. Needs `update_scene()` then `render()` per frame. |
| `model.nq` / `model.nv` / `model.nu` | Counts: length of `qpos` (position-space DOF), length of `qvel` (velocity-space DOF), number of actuators. Differ when there's a free joint — see §2. |
| `model.opt.timestep` | The physics integration step size (`physics_dt`), read from the XML or MuJoCo's default (`0.002s` here). |
| `data.qpos` / `data.qvel` | Current generalized positions / velocities. Root free-joint pose lives at `qpos[0:7]` (3 pos + 4 quat), root free-joint velocity at `qvel[0:6]` (3 lin + 3 ang). |
| `data.ctrl` | Actuator command array, one entry per actuator (`nu` long). What *you* set every tick to influence the sim. |
| `model.actuator_gear` | Per-actuator scale factor: `torque = gear * ctrl`. `1` here, so `ctrl` is torque directly (within `ctrlrange`). |
| `model.actuator_ctrlrange` | Valid `[min, max]` for `data.ctrl` per actuator — `[-1, 1]` here. Values outside this get clipped. |
| `model.actuator_trnid` | Per-actuator "what am I attached to" — column 0 is the driven joint's ID for a simple motor. |
| `model.jnt_actfrcrange` | Per-joint `[min, max]` total actuator torque allowed, from the XML's `actuatorfrcrange`. |
| `np.gradient(arr, dt, axis=0)` | Numerical derivative via finite differences (central in the interior, one-sided at the edges) — used to turn `dof_pos` into an approximate `qdot_ref`. |

---

## Appendix: the offscreen-rendering saga (environment, not MuJoCo concepts)

Not robotics/MuJoCo per se, but worth keeping since it ate real debugging time:

- Running headless over SSH (no `DISPLAY`) breaks MuJoCo's default GLFW renderer.
- `MUJOCO_GL=egl` is the right fix on this machine (real NVIDIA GPUs present), but needs a
  recent-enough `PyOpenGL` for the `EGLDeviceEXT` binding MuJoCo's EGL backend uses.
- **`difftrack`'s `PyOpenGL` is hard-pinned to `3.1.0` for `pyrender`** (used by the unrelated
  ExpertEdit rendering pipeline) — upgrading it there previously caused a silent segfault in
  `pipeline.py`'s rendering (not an error, and not until `render()` actually ran — see
  `CLAUDE.local.md`'s Environments section). **Do not upgrade PyOpenGL in `difftrack`.**
- `MUJOCO_GL=osmesa` (CPU fallback) and `Xvfb` (virtual display) both avoid the PyOpenGL/EGL
  path entirely, but need system packages (`libOSMesa`, `Xvfb`) not installed on this machine,
  and no sudo to add them.
- Fix: a separate `mujoco_sim` conda env, isolated from `difftrack`, with its own fresh
  `PyOpenGL` — no `pyrender`/`smplx` in it to conflict with. Run via
  `MUJOCO_GL=egl /scratch/pbk5339/conda_envs/mujoco_sim/bin/python demo/simulator/unitreeg1_sim.py`.
