This is a project that attempts to recreate the expertEdit method (2026 ECCV). The code for this specifically is under demo/arch/. They do not release their code so this is an attempt at my own recreation. The egoexo4d dataset is used.

On the side, some experiments for robotics are also being performed under simulator/ using Mujoco.

---

## Reference paper

ExpertEdit: *Learning Skill-Aware Motion Editing from Expert Videos* (ECCV 2026, UT Austin).
- Paper: https://arxiv.org/abs/2604.10466 — project page: https://vision.cs.utexas.edu/projects/expert_edit/
- No code release. The authors *did* release a metadata-only eval benchmark, which is vendored here at [expertedit_eval_benchmark_v1.0/](expertedit_eval_benchmark_v1.0/): novice↔expert pseudo-pairs, DTW frame correspondences, technique metadata, kinematic signal definitions. No media, no pose arrays — it is IDs + mappings that you join against a local EgoExo4D copy. Field meanings are in [docs/field_dictionary.md](expertedit_eval_benchmark_v1.0/docs/field_dictionary.md).
- Techniques covered by the benchmark: midrange_jumpshot, mikan_layup, reverse_layup, penalty_kick, plus karate (reverse_punch, spinning_back_kick, front_kick, roundhouse_kick). This recreation currently only works on **mid-range jump shot**.

## The core idea being recreated

Take a novice's motion, locate the *operative moment* of the skill, and rewrite only that window so it looks like an expert's — leaving the rest of the clip untouched. Everything happens in SMPL pose space, not pixel space.

Two-stage, PoseGPT/MaskGIT-flavored:

1. **Stage 1 — pose tokenizer** ([demo/arch/pose_tokenizer.py](demo/arch/pose_tokenizer.py)): VQ autoencoder over motion. Causal 6-layer transformer encoder → split the 256-d latent into two halves → two independent 256-entry codebooks (`c1`, `c2`, 128-d each) → non-causal 6-layer transformer decoder back to the 69-d pose vector. Codebook lookup is done on **L2-normalized** vectors on both sides (encoder output and codebook weights) so the encoder can't drift away from the codebook scale. Straight-through estimator for backprop. Trained with MSE recon + VQ/commitment loss.
2. **Stage 2 — motion infiller** ([demo/arch/motion_infiller.py](demo/arch/motion_infiller.py)): 12-layer *bidirectional* transformer, BERT-style MLM over the frozen Stage-1 token IDs. Separate embedding tables and separate classification heads per codebook stream. Token id 256 is the `[MASK]` token. Loss is cross-entropy on masked positions only.

**Masking is the whole trick.** The mask is not random — it's centered on the *kinematic peak* `t*`, found by `get_kinematic_peaks()` in [demo/arch/train_infiller.py](demo/arch/train_infiller.py#L30): argmax of vertical root-translation velocity (the jump takeoff). Training samples the span fraction uniformly in [0, 0.3]; inference uses a fixed 0.15.

At inference the model is trained only on *expert* clips, so infilling a masked window of a *novice* clip hallucinates the expert version of that moment — that is the "edit".

## Pose vector convention

Everywhere in `demo/arch/` and `demo/app/`, one frame is a **69-d** vector:

```
[0:3]   trans_world       global root translation
[3:6]   root_orient       root orientation, axis-angle
[6:69]  body_pose         21 joints × 3, axis-angle
```

Sequences are fixed at **T = 90** frames @ 30 fps (45 before `t*`, 44 after). Longer clips get a center crop at test time / random crop at train time; shorter clips repeat-pad the last frame. See `smplPoseLoader` in [demo/arch/data_utils.py](demo/arch/data_utils.py).

Note the joint-count mismatch: SMPL's `body_pose` wants 23 joints (69 dims) but only 21 (63 dims) are tracked. Anything that calls `smplx` pads the missing 2 joints with zeros — see `get_vertices()` in [train_tokenizer.py](demo/arch/train_tokenizer.py#L64) and `_vertices()` in [pipeline.py](demo/app/pipeline.py#L335). Keep that convention if you add new SMPL forward passes.

Y is up in WHAM world coordinates (gravity-aligned), which is why the peak heuristic uses index 1. The retargeting path converts to Z-up.

## Pipeline, end to end

```
video (.mp4)
  → WHAM                    3rd-party HMR, per-frame SMPL for every tracked person
  → actor-selection         pick the track with the largest 95th–5th pctile vertical pelvis travel
  → wham_output_selected.pkl   one person: {trans_world, pose_world, betas}
  → 90-frame window
  → Stage 1 tokenizer       [1,90,69] → token ids [1,90,2]
  → mask around t*
  → Stage 2 infiller        predict tokens inside the mask
  → splice tokens           novice tokens outside the mask, predicted tokens inside
  → Stage 1 decoder         tokens → [1,90,69]
  → write back into the full-length original, body_pose only
  → smplx + pyrender        side-by-side mp4, masked frames tinted orange
```

The "write back" step deliberately touches **only `pose_world[:, 3:66]`** — root translation and root orientation are kept from the original. This keeps the edited person in the same place in the world.

Two splice modes exist and both are wired through the UI:
- `"window"` (default): replace all 90 frames with decoder output. Hides tokenizer/infiller boundary artifacts but also inherits all tokenizer reconstruction error.
- `"mask"`: replace only masked frames. Truer to the paper's intent, but snaps at the boundary — there's a commented-out linear-blend attempt in [motion_edit.py](demo/arch/motion_edit.py#L221).

## Layout

| Path | What |
| --- | --- |
| [demo/arch/](demo/arch/) | **The recreation.** Models, both training loops, inference script. |
| [demo/app/](demo/app/) | Gradio web UI. `pipeline.py` is `motion_edit.py` + `smpl_viz.py` refactored into importable functions; `main.py` is the UI shell. |
| [demo/egoexo_scripts/](demo/egoexo_scripts/) | Mining EgoExo4D for jumpshot clips: proficiency filtering, narration search, ffmpeg cutting. |
| [demo/simulator/](demo/simulator/) | MuJoCo side-quest. Unitree G1 playback. |
| [demo/GMR/](demo/GMR/) | Vendored General Motion Retargeting (has its own CLAUDE.md). Also the source of `assets/body_models` for smplx. |
| [WHAM/](WHAM/) | Vendored WHAM. Gitignored. |
| [dataset_prep/](dataset_prep/) | The **ExpertAF-style** learner/expert pair dataset build (separate data line, see below). |
| [expertedit_eval_benchmark_v1.0/](expertedit_eval_benchmark_v1.0/) | Authors' released eval metadata. Read-only reference. |
| [archive/](archive/) | Old work. Gitignored. |

Loose scripts at `demo/` root ([smpl_viz.py](demo/smpl_viz.py), [get_pose.py](demo/get_pose.py), [run_wham.py](demo/run_wham.py), [retarget.py](demo/retarget.py), [inspect_pose_demo.py](demo/inspect_pose_demo.py)) are one-off tools with hardcoded input/output paths at the top — they're meant to be edited in place, not argparsed.

## Two separate data lines — don't confuse them

**(A) What actually trains the models today:** basketball mid-range jump shot, expert-only.
- `demo/egoexo_scripts/find_late_experts.py` → takes with `proficiency_score == "Late Expert"` on task "Basketball Drills - Mid-Range Jump Shooting" that are also covered by `atomic_descriptions`.
- `get_clips.py` → SentenceTransformer similarity against operative phrases ("player performs the jump and shoots", "shot is released"), threshold 0.5, then ffmpeg-cut a 90-frame window around each hit → `demo/basketball_expert_clips/Mid-range jump shot/` (729 clips).
- `run_wham.py --gpu N --shard-id N --num-shards 4` (or `run_wham.sh`) → `demo/basketball_expert_smpl_v2/` (564 dirs with `wham_output_selected.pkl`). **This is `ROOT_DIR` for both training scripts.**
- `demo/basketball_expert_smpl/` is the older v1 (3 techniques, different clip naming). `_v2` supersedes it.

**(B) The larger ExpertAF-style pair dataset**, built earlier and currently unused by `demo/arch/`: `dataset_prep/` → `dataset_out/{train,val}/<pair-id>/` with `learner_{ego,exo}.mp4`, `expert_{ego,exo}.mp4`, `{learner,expert}_pose.json`, `meta.json`. 25505 train / 1272 val. Built from ExpertAF's DTW pair JSONs (`expertaf_drive/`) + EgoExo4D downscaled 448 takes; see `download_data.sh` for the CLI download recipe. The pose JSONs use EgoExo4D's COCO-17 `ego_pose` annotations, **not** SMPL — that's why WHAM exists in this repo at all. `example.json` at the repo root is one raw ExpertAF entry, `pose.py` is the early PA-MPJPE + DTW exploration on this data.

The inference target in [motion_edit.py](demo/arch/motion_edit.py#L33) pulls a learner clip out of line (B) while the models are trained on line (A) — that's intentional (train on experts, edit a novice).

## Running things

Every script assumes **cwd = repo root** (`/scratch/pbk5339/thesis_new`). Paths are relative strings like `"demo/arch/tokenizer_ckpts_v3/..."`, not resolved against `__file__`. The exception is `demo/app/pipeline.py`, which resolves `REPO_ROOT` properly and can be run from anywhere.

Both training scripts are **DDP-only** — they call `dist.init_process_group("nccl")` unconditionally and read `LOCAL_RANK` from the environment, so they must be launched under `torchrun`. They will not run as a plain `python foo.py`.

Training is **step-based, not epoch-based**: `infinite_batches()` cycles the loader forever and an "epoch" is a fixed `steps = 1000` iterations. `sampler.set_epoch()` is called inside the generator, not the outer loop.

```bash
# stage 1
torchrun --nproc_per_node=<N> demo/arch/train_tokenizer.py
# stage 2 (needs a stage-1 ckpt; edit TOKENIZER_CKPT at the top)
torchrun --nproc_per_node=<N> demo/arch/train_infiller.py

# one-shot inference + pkl dump (edit VID_PATH / ckpt paths at the top)
python demo/arch/motion_edit.py

# UI: run on the lab machine, tunnel from the laptop
python demo/app/main.py --gpu 9
ssh -N -L 7860:127.0.0.1:7860 <usr>@<lab-machine>   # then http://127.0.0.1:7860

# same pipeline from the CLI
python demo/app/pipeline.py <input.mp4|input.pkl> --out demo/app/runs/cli --span 0.15 --splice window
```

Hyperparameters live as ALL-CAPS constants inside `if __name__ == "__main__"` blocks — there is no config file or argparse for training. Change them in the file.

## Environments

Three conda envs, and they are mutually incompatible on purpose:

- **`difftrack`** — the main env. Everything in `demo/arch/`, `demo/app/`, GMR retargeting, rendering.
- **`wham`** — WHAM's own torch/mmcv stack. `pipeline.py` never imports WHAM; it shells out via `conda run -n wham --no-capture-output` (see `_conda_run_prefix()`), reusing `$CONDA_EXE` so it picks the same conda install as the active shell. Override with `WHAM_CONDA_ENV`.
- **`gmr`** — only if running GMR's own setup per `demo/GMR/CLAUDE.md`; in practice the retargeting has been run from `difftrack`.

Env vars that matter:
```bash
export MUJOCO_GL=egl              # MuJoCo offscreen render
PYOPENGL_PLATFORM=egl             # pyrender; set in-process before importing pyrender
CUDA_VISIBLE_DEVICES=<id>         # must be set before torch import — main.py --gpu does this
```
`pipeline.py` also honors `TOKENIZER_CKPT`, `INFILLER_CKPT`, `SMPL_MODEL_DIR`, `WHAM_DIR` as env var overrides.

**`PyOpenGL` version conflict — check this first if rendering segfaults.** `difftrack` must stay on `PyOpenGL==3.1.0` (pyrender 0.1.45's hard pin; `PyOpenGL-accelerate` isn't needed and won't even build on Python 3.10, so it's uninstalled). The robotics side-quest's GMR/MuJoCo setup wants a newer PyOpenGL, and upgrading it in `difftrack` (e.g. via [demo/to_retarget.txt](demo/to_retarget.txt)'s `pip install --upgrade PyOpenGL PyOpenGL-accelerate` step) silently breaks pyrender: rendering doesn't error, it segfaults the whole process, and it doesn't happen at import — only when `renderer.render(scene)` actually runs. This already happened once (Sep 11 upgrade → broke `demo/app/`'s render step, not caught until Sep 13). If offscreen rendering in `demo/app/pipeline.py`, `demo/smpl_viz.py`, etc. segfaults after touching anything GMR/MuJoCo-related, check `pip show PyOpenGL` in `difftrack` before looking anywhere else — reinstall with `pip install PyOpenGL==3.1.0` (no `--upgrade`, and don't reinstall `PyOpenGL-accelerate`). Longer-term, the robotics work should probably move to the separate `gmr` env instead of sharing `difftrack`, to stop the two from fighting over one package.

**The numpy/chumpy alias shim**: chumpy (pulled in transitively when smplx unpickles legacy SMPL `.pkl` files) still does `from numpy import bool, int, float, ...`, removed in numpy ≥1.24. Several files re-add those aliases before importing smplx. If a new file touches smplx and blows up on `np.bool`, that's why — copy the block from [retarget.py](demo/retarget.py#L22) or `_np_compat()` in [pipeline.py](demo/app/pipeline.py#L299).

## Checkpoints

`tokenizer_ckpts_v3/` and `infiller_ckpts_v3/` are current; `_v2` and the unsuffixed dirs are older runs kept for comparison. All are gitignored. The pair everything currently points at:

- `demo/arch/tokenizer_ckpts_v3/pose_tokenizer_epoch_20.pth`
- `demo/arch/infiller_ckpts_v3/motion_infiller_epoch_4.pth`

Note `train_tokenizer.py` sets `RESUME_CKPT` to `pose_tokenizer_epoch_100.pth` and silently starts fresh if it doesn't exist — check the "Resumed tokenizer from..." line in the log rather than assuming.

Checkpoint dicts are `{epoch, model_state_dict, optimizer_state_dict, loss}`, saved from `model.module.state_dict()` (DDP wrapper stripped), so they load into a bare model.

## Robotics side-quest

SMPL motion → Unitree G1 (29 DOF) → MuJoCo. Independent of the ExpertEdit work; shares only the SMPL output format.

**Before touching this: read the `PyOpenGL` note under Environments.** This work and `demo/app/`'s rendering share `difftrack` and want different `PyOpenGL` versions — upgrading it for GMR/MuJoCo is what silently segfaults `pipeline.py`'s rendering later.

1. [demo/get_pose.py](demo/get_pose.py) — read a WHAM/edited pkl, rotate Y-up → Z-up with basis `[[0,0,1],[1,0,0],[0,1,0]]`, lift the pelvis so `min_z → 0.78` (G1 standing height), write `gmr_input_motion_expert.pkl` in GMR's expected schema (`mocap_frame_rate`, `gender`, `root_orient`, `pose_body`, `trans`, `betas`).
2. GMR retargeting — the exact command is recorded in [demo/to_retarget.txt](demo/to_retarget.txt). Produces `demo/GMR/g1_retargeted_expert.pkl` with `dof_pos` / `root_pos` / `root_rot`.
3. [demo/simulator/unitreeg1_sim.py](demo/simulator/unitreeg1_sim.py) — kinematic playback: teleport `qpos` frame by frame, `qvel = 0`, `mj_forward`, offscreen render. **GMR writes quaternions as `[x,y,z,w]`; MuJoCo wants `[w,x,y,z]`** — the column reorder is at [unitreeg1_sim.py:34](demo/simulator/unitreeg1_sim.py#L34). `physics_simulation()` is a stub; actual dynamics is the next step.

## Conventions and gotchas

- Comments in this repo are lowercase, informal, and often triple-quoted block comments used as section headers rather than docstrings. Match that. `#NOTE:` marks something the author wants to remember.
- Dead code is kept commented out on purpose (old non-DDP training loop, the PoseGPT vertex loss, `canonicalize_clip`, the mask-blend experiment). Don't delete it during cleanups — it's a record of what was tried.
- Every file has a commented-out `debugpy.listen(("127.0.0.1", 5678))` block. Uncomment to attach; leave it commented in anything committed.
- `positionalEncoding` is a *learned* parameter table, not sinusoidal, `max_len=500`.
- `poseTokenizer` reuses `self.pos_encoder` for both encoder and decoder (one shared table). Inference code in `motion_edit.py` / `pipeline.py` reaches into the tokenizer's internals (`c1`, `c2`, `decoder_proj`, `pos_encoder`, `transformer_decoder`, `output_proj_decoder`) to decode edited tokens — changing tokenizer attribute names breaks inference silently, not at import.
- The `F.normalize` on codebook lookup must be applied at inference too, mirroring `quantize()`. Both inference paths do this; a new one must as well.
- The actor-selection heuristic (vertical pelvis travel) is jumpshot-specific. Other techniques need their own — the ideas are sketched in the docstring of [demo/run_wham.py](demo/run_wham.py).
- `demo/app/runs/` and `demo/app/gradio_tmp/` accumulate per-run output dirs, gitignored.
