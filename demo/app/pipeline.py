'''
pipeline pieces used by the web UI (demo/app/main.py).

this is basically demo/arch/motion_edit.py + demo/smpl_viz.py refactored into
functions instead of a script, so the UI can call them one step at a time.

nothing here is gradio-specific -- can be used from a notebook/CLI too.
'''
import os

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")  # must be set BEFORE pyrender is imported

import sys
import time
import subprocess
from pathlib import Path

import numpy as np
import joblib
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
ARCH_DIR = REPO_ROOT / "demo" / "arch"
sys.path.insert(0, str(ARCH_DIR))  # arch modules import each other by flat name

from pose_tokenizer import poseTokenizer            # noqa: E402
from motion_infiller import MotionInfiller          # noqa: E402
from train_infiller import (                        # noqa: E402
    get_kinematic_peaks,
    create_masked_inputs_inference,
)

'''
paths -- override with env vars if you point at a different run
'''
TOKENIZER_CKPT = Path(os.environ.get(
    "TOKENIZER_CKPT", REPO_ROOT / "demo/arch/tokenizer_ckpts_v3/pose_tokenizer_epoch_20.pth"))
INFILLER_CKPT = Path(os.environ.get(
    "INFILLER_CKPT", REPO_ROOT / "demo/arch/infiller_ckpts_v3/motion_infiller_epoch_4.pth"))
SMPL_MODEL_DIR = Path(os.environ.get(
    "SMPL_MODEL_DIR", REPO_ROOT / "demo/GMR/assets/body_models"))
WHAM_DIR = Path(os.environ.get("WHAM_DIR", REPO_ROOT / "WHAM"))
WHAM_CONDA_ENV = os.environ.get("WHAM_CONDA_ENV", "wham")

TARGET_LEN = 90
MASK_TOKEN_ID = 256

_MODELS = {}      # device -> (tokenizer, infiller)
_SMPL = None      # cached smplx body model
_RENDERER = None  # cached pyrender offscreen renderer (W, H, renderer)


def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


'''
step 1: input -> a single-person WHAM dict
'''
def jumpshot_heuristic(wham_output):
    """
    wham_output: {track_id: person_dict}. picks the track whose pelvis travels
    the most vertically (same heuristic as demo/arch/motion_edit.py).
    """
    track_ids = list(wham_output.keys())
    people = list(wham_output.values())
    if len(people) == 1:
        return track_ids[0]

    displacements = []
    for person in people:
        vertical = person["trans_world"][:, 1]
        displacements.append(np.percentile(vertical, 95) - np.percentile(vertical, 5))
    return track_ids[int(np.argmax(displacements))]


def load_person(pkl_path):
    """
    accepts either a per-person pkl (wham_output_selected.pkl / edited_motion_smpl.pkl)
    or a raw multi-person wham_output.pkl, and always returns one person dict.
    """
    data = joblib.load(pkl_path)
    if isinstance(data, dict) and "pose_world" in data:
        return data
    if isinstance(data, dict):  # {track_id: person}
        best = jumpshot_heuristic(data)
        return data[best]
    raise ValueError(f"don't know how to read {pkl_path}: got {type(data)}")


def _conda_run_prefix():
    """
    prefix that runs a command inside WHAM_CONDA_ENV without that env ever
    being active in *this* process. `conda run -n <env>` replicates
    `conda activate <env> && ...` faithfully (PATH, LD_LIBRARY_PATH,
    activate.d hooks) -- so WHAM's own torch/mmcv/etc never has to be
    importable from the (difftrack) env running the rest of the pipeline.

    CONDA_EXE is set by conda itself whenever an env is active, so reusing it
    picks the same conda install that set up the current shell rather than
    hoping a bare "conda" is on PATH (not guaranteed on every cluster login node).
    """
    conda_exe = os.environ.get("CONDA_EXE", "conda")
    return [conda_exe, "run", "-n", WHAM_CONDA_ENV, "--no-capture-output"]


def run_wham(video_path, out_root, log=print):
    """
    runs WHAM's demo.py on a video and returns the path to wham_output_selected.pkl.
    WHAM writes into out_root/<video stem>/. runs inside WHAM_CONDA_ENV (default
    "wham"), regardless of which env this script itself is running under --
    override with the WHAM_CONDA_ENV env var if yours is named differently.
    """
    video_path = Path(video_path).resolve()
    out_root = Path(out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    actor_dir = out_root / video_path.stem
    selected_pkl = actor_dir / "wham_output_selected.pkl"
    if selected_pkl.exists():
        log(f"reusing existing WHAM output at {selected_pkl}")
        return selected_pkl

    if not (WHAM_DIR / "demo.py").exists():
        raise FileNotFoundError(f"WHAM not found at {WHAM_DIR} (set WHAM_DIR env var)")

    log(f"running WHAM (conda env '{WHAM_CONDA_ENV}') on {video_path.name} (this is the slow part)...")
    t0 = time.perf_counter()
    subprocess.run(
        _conda_run_prefix() + [
            "python", "demo.py",
            "--video", str(video_path),
            "--output_pth", str(out_root),
            "--save_pkl",
        ],
        cwd=str(WHAM_DIR), check=True,
    )
    log(f"WHAM done in {time.perf_counter() - t0:.1f}s")

    wham_output = joblib.load(actor_dir / "wham_output.pkl")
    best = jumpshot_heuristic(wham_output)
    joblib.dump(wham_output[best], selected_pkl)
    log(f"selected track {best} of {len(wham_output)} -> {selected_pkl}")
    return selected_pkl


'''
step 2: person dict -> the fixed-length window the models expect
'''
def motion_window(person, target_len=TARGET_LEN):
    """
    same convention as smplPoseLoader(split='test'): center crop if long enough,
    otherwise repeat-pad the last frame.

    returns motion [1, target_len, 69] float32 tensor, start_frame, valid_len
    """
    trans = person["trans_world"].reshape(-1, 3)
    pose = person["pose_world"]
    root_orient = pose[:, :3].reshape(-1, 3)
    body_pose = pose[:, 3:66].reshape(pose.shape[0], -1)
    motion = np.concatenate([trans, root_orient, body_pose], axis=-1).astype(np.float32)

    t_raw = motion.shape[0]
    if t_raw >= target_len:
        start = (t_raw - target_len) // 2
        valid_len = target_len
        motion = motion[start:start + target_len]
    else:
        start = 0
        valid_len = t_raw
        pad = np.tile(motion[-1:], (target_len - t_raw, 1))
        motion = np.concatenate([motion, pad], axis=0)

    return torch.tensor(motion).unsqueeze(0), start, valid_len


'''
step 3: models
'''
def get_models(device=None, log=print):
    """loads (and caches) the frozen tokenizer + infiller."""
    device = device or get_device()
    key = str(device)
    if key in _MODELS:
        return _MODELS[key]

    tokenizer = poseTokenizer(hidden_dim=384, out_dim=256, num_joints=21, num_layers=6).to(device)
    if TOKENIZER_CKPT.exists():
        ckpt = torch.load(TOKENIZER_CKPT, map_location=device)
        tokenizer.load_state_dict(ckpt["model_state_dict"])
        log(f"loaded tokenizer: {TOKENIZER_CKPT}")
    else:
        log(f"WARNING: {TOKENIZER_CKPT} not found -- tokenizer is random!")
    tokenizer.eval()

    infiller = MotionInfiller(vocab_size=256, emb_dim=128, hidden_dim=256,
                              num_layers=12, nhead=8).to(device)
    if INFILLER_CKPT.exists():
        ckpt = torch.load(INFILLER_CKPT, map_location=device)
        infiller.load_state_dict(ckpt["model_state_dict"])
        log(f"loaded infiller: {INFILLER_CKPT}")
    else:
        log(f"WARNING: {INFILLER_CKPT} not found -- infiller is random!")
    infiller.eval()

    for m in (tokenizer, infiller):
        for p in m.parameters():
            p.requires_grad = False

    _MODELS[key] = (tokenizer, infiller)
    return _MODELS[key]


'''
step 4: tokenize -> mask -> infill -> decode -> splice back
'''
@torch.no_grad()
def edit_motion(person, span_fraction=0.15, splice="window", device=None, log=print):
    """
    person: WHAM-style dict with trans_world / pose_world / betas
    span_fraction: size of the masked window around the kinematic peak
    splice: "window" replaces the whole 90-frame window with the decoder output,
            "mask"   replaces only the masked frames (leaves the rest untouched)

    returns (edited_person_dict, tokenizer_recon_person_dict, info_dict). The
    tokenizer-recon dict is the plain encode->quantize->decode round trip with
    no masking/infiller involved at all -- useful for isolating how much of the
    "edit" is actually just tokenizer reconstruction bias vs. the infiller.
    """
    device = device or get_device()
    tokenizer, infiller = get_models(device, log=log)

    data, start_frame, valid_len = motion_window(person)
    data = data.to(device)

    # x_recon here IS the pure tokenizer round trip (encode -> quantize -> decode,
    # no masking) -- the infiller never touches it. Stage 1 training already
    # optimizes exactly this reconstruction (see train_tokenizer.py), we just
    # never kept it around before now.
    x_recon, _, clean_indices = tokenizer(data)           # [1, T, 69], _, [1, T, 2]
    peak_indices = get_kinematic_peaks(data)              # [1]

    masked_tokens, mask_labels = create_masked_inputs_inference(
        clean_indices, peak_indices,
        mask_token_id=MASK_TOKEN_ID, span_fraction=span_fraction,
    )

    logits1, logits2 = infiller(masked_tokens)
    pred_tokens = torch.stack([logits1.argmax(-1), logits2.argmax(-1)], dim=-1)  # [1, T, 2]

    edited_tokens = clean_indices.clone()
    edited_tokens[mask_labels] = pred_tokens[mask_labels]

    n_masked = int(mask_labels.sum().item())
    n_changed = int(((edited_tokens != clean_indices) & mask_labels.unsqueeze(-1)).sum().item())

    # decode with the frozen stage-1 codebooks + decoder
    z_q1 = F.normalize(tokenizer.c1(edited_tokens[:, :, 0]), dim=-1)
    z_q2 = F.normalize(tokenizer.c2(edited_tokens[:, :, 1]), dim=-1)
    z_q = torch.cat([z_q1, z_q2], dim=-1)

    h_dec = tokenizer.decoder_proj(z_q)
    h_dec = tokenizer.pos_encoder(h_dec)
    edited_motion = tokenizer.output_proj_decoder(tokenizer.transformer_decoder(h_dec))

    decoded_body = edited_motion.squeeze(0).cpu().numpy()[:, 6:69]   # [T, 63]
    mask_np = mask_labels.squeeze(0).cpu().numpy()[:valid_len]

    full_pose = person["pose_world"]
    end_frame = start_frame + valid_len
    edited_pose = full_pose.copy()

    if splice == "mask":
        for idx in np.where(mask_np)[0]:
            edited_pose[start_frame + idx, 3:66] = decoded_body[idx]
    else:  # "window"
        edited_pose[start_frame:end_frame, 3:66] = decoded_body[:valid_len]

    edited_person = {
        "trans_world": person["trans_world"],
        "pose_world": edited_pose,
        "betas": person.get("betas", None),
        "edit_start_frame": start_frame,
        "edit_end_frame": end_frame,
        "edit_mask": mask_np,
    }

    # pure tokenizer round trip: no mask, no infiller -- the whole window is
    # just decode(quantize(encode(novice_window))). Splice mode doesn't apply
    # here since there's no unmasked region to preserve; it's the reconstruction
    # for the full window either way.
    tok_recon_body = x_recon.squeeze(0).cpu().numpy()[:, 6:69]   # [T, 63]
    tok_recon_pose = full_pose.copy()
    tok_recon_pose[start_frame:end_frame, 3:66] = tok_recon_body[:valid_len]
    tokenizer_recon_person = {
        "trans_world": person["trans_world"],
        "pose_world": tok_recon_pose,
        "betas": person.get("betas", None),
        "edit_start_frame": start_frame,
        "edit_end_frame": end_frame,
    }

    info = {
        "raw_frames": int(full_pose.shape[0]),
        "window": (int(start_frame), int(end_frame)),
        "peak_frame_in_window": int(peak_indices[0].item()),
        "masked_frames": n_masked,
        "changed_token_values": n_changed,
        "splice": splice,
        "span_fraction": span_fraction,
    }
    log(f"peak @ frame {info['peak_frame_in_window']} of the window, "
        f"{n_masked} frames masked, infiller changed {n_changed}/{n_masked * 2} token values")
    return edited_person, tokenizer_recon_person, info


'''
step 5: rendering (demo/smpl_viz.py, but callable and with a shared camera)
'''
def _np_compat():
    import numpy as _np
    for name, typ in [("bool", bool), ("int", int), ("float", float),
                      ("complex", complex), ("object", object), ("str", str), ("unicode", str)]:
        if not hasattr(_np, name):
            setattr(_np, name, typ)


def _get_smpl():
    global _SMPL
    if _SMPL is None:
        import smplx
        _SMPL = smplx.create(model_path=str(SMPL_MODEL_DIR), model_type="smpl", gender="neutral")
    return _SMPL


def _get_renderer(w, h):
    global _RENDERER
    import pyrender
    if _RENDERER is None or _RENDERER[0] != (w, h):
        if _RENDERER is not None:
            _RENDERER[1].delete()
        _RENDERER = ((w, h), pyrender.OffscreenRenderer(viewport_width=w, viewport_height=h))
    return _RENDERER[1]


def _look_at(eye, target, up=np.array([0, 1, 0])):
    eye, target, up = map(np.asarray, (eye, target, up))
    z = eye - target; z /= np.linalg.norm(z)
    x = np.cross(up, z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    pose = np.eye(4)
    pose[:3, 0], pose[:3, 1], pose[:3, 2], pose[:3, 3] = x, y, z, eye
    return pose


def _vertices(person, frame_slice=None):
    model = _get_smpl()
    pose_world = person["pose_world"]
    trans_world = person["trans_world"]
    betas = person["betas"]
    betas = betas.mean(axis=0) if getattr(betas, "ndim", 1) == 2 else betas

    if frame_slice is not None:
        pose_world = pose_world[frame_slice]
        trans_world = trans_world[frame_slice]

    n = pose_world.shape[0]
    with torch.no_grad():
        out = model(
            global_orient=torch.tensor(pose_world[:, :3], dtype=torch.float32),
            body_pose=torch.tensor(pose_world[:, 3:], dtype=torch.float32),
            transl=torch.tensor(trans_world, dtype=torch.float32),
            betas=torch.tensor(betas, dtype=torch.float32).unsqueeze(0).repeat(n, 1),
        )
    return out.vertices.numpy(), model.faces


def _render_frames(all_verts, faces, cam_pose, w, h, colors=None):
    import trimesh
    import pyrender
    renderer = _get_renderer(w, h)
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)

    frames = []
    for t in range(all_verts.shape[0]):
        mesh = trimesh.Trimesh(vertices=all_verts[t], faces=faces, process=False)
        if colors is not None:
            mesh.visual.vertex_colors = colors[t]
        py_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=True)

        scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0], ambient_light=[0.4] * 3)
        scene.add(py_mesh)
        scene.add(camera, pose=cam_pose)
        scene.add(light, pose=cam_pose)
        color, _ = renderer.render(scene)
        frames.append(color)
    return frames


def render_triple(original, tok_recon, edited, out_dir, fps=30, size=640, window_only=True,
                  include_tok_recon=True, log=print):
    """
    renders original + edited, and optionally the pure-tokenizer-reconstruction
    (no infiller involved) as a 3rd video -- set include_tok_recon=False to skip
    it (it's the extra rendering cost of the three, since it needs its own SMPL
    forward + render pass same as the other two). all rendered videos share one
    camera so they're directly comparable. edited frames inside the mask are
    tinted; tok_recon has no mask (it's a plain autoencoder round trip) so it
    renders untinted.

    returns (original_mp4, tokenizer_recon_mp4_or_None, edited_mp4)
    """
    _np_compat()
    import imageio

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_slice = None
    if window_only and "edit_start_frame" in edited:
        frame_slice = slice(int(edited["edit_start_frame"]), int(edited["edit_end_frame"]))

    t0 = time.perf_counter()
    verts_orig, faces = _vertices(original, frame_slice)
    verts_edit, _ = _vertices(edited, frame_slice)
    verts_tok = _vertices(tok_recon, frame_slice)[0] if include_tok_recon else None

    # one camera fitted to whichever sequences are actually being rendered, so
    # turning tok_recon on/off doesn't shift the framing of the other two
    seqs = [verts_orig, verts_edit] + ([verts_tok] if include_tok_recon else [])
    flat = np.concatenate(seqs).reshape(-1, 3)
    lo, hi = flat.min(0), flat.max(0)
    center = (lo + hi) / 2
    radius = np.linalg.norm(hi - lo) * 0.75
    cam_pose = _look_at(eye=center + np.array([0, radius * 0.3, radius]), target=center)

    # tint the frames the infiller actually rewrote (edited only)
    colors = None
    mask = edited.get("edit_mask")
    if mask is not None and frame_slice is not None and len(mask) == verts_edit.shape[0]:
        base = np.array([150, 160, 190, 255], dtype=np.uint8)
        hot = np.array([230, 120, 60, 255], dtype=np.uint8)
        colors = [hot if m else base for m in mask]

    n_videos = 3 if include_tok_recon else 2
    log(f"rendering {verts_orig.shape[0]} frames x{n_videos} ...")
    orig_frames = _render_frames(verts_orig, faces, cam_pose, size, size)
    edit_frames = _render_frames(verts_edit, faces, cam_pose, size, size, colors=colors)
    tok_frames = _render_frames(verts_tok, faces, cam_pose, size, size) if include_tok_recon else None

    orig_mp4 = out_dir / "original.mp4"
    edit_mp4 = out_dir / "edited.mp4"
    imageio.mimwrite(orig_mp4, orig_frames, fps=fps)
    imageio.mimwrite(edit_mp4, edit_frames, fps=fps)

    tok_mp4 = None
    if include_tok_recon:
        tok_mp4 = out_dir / "tokenizer_recon.mp4"
        imageio.mimwrite(tok_mp4, tok_frames, fps=fps)

    names = [orig_mp4.name] + ([tok_mp4.name] if tok_mp4 else []) + [edit_mp4.name]
    log(f"rendered in {time.perf_counter() - t0:.1f}s -> {', '.join(names)}")
    return orig_mp4, tok_mp4, edit_mp4


'''
the whole thing, end to end
'''
def run_pipeline(input_path, out_dir, span_fraction=0.15, splice="window",
                 fps=30, size=640, window_only=True, render=True, show_tok_recon=False, log=print):
    """
    input_path: .mp4 (runs WHAM first) or .pkl (WHAM output, raw or selected)
    show_tok_recon: also render the pure-tokenizer-reconstruction 3rd video (no
        infiller). off by default -- it's a real extra SMPL+render pass, not
        free, so it's opt-in. the tokenizer_recon_smpl.pkl is always saved
        either way since it comes for free out of edit_motion()'s own forward pass.
    returns a dict of results/paths.
    """
    input_path = Path(input_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if input_path.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}:
        pkl_path = run_wham(input_path, out_dir / "wham", log=log)
    elif input_path.suffix.lower() in {".pkl", ".pth"}:
        pkl_path = input_path
    else:
        raise ValueError(f"unsupported input: {input_path.suffix}")

    person = load_person(pkl_path)
    log(f"input pose: {person['pose_world'].shape[0]} frames")

    edited, tok_recon, info = edit_motion(person, span_fraction=span_fraction, splice=splice, log=log)

    input_pkl = out_dir / "input_smpl.pkl"
    tok_recon_pkl = out_dir / "tokenizer_recon_smpl.pkl"
    edited_pkl = out_dir / "edited_motion_smpl.pkl"
    joblib.dump(person, input_pkl)
    joblib.dump(tok_recon, tok_recon_pkl)
    joblib.dump(edited, edited_pkl)
    log(f"saved {edited_pkl}")

    orig_mp4 = tok_mp4 = edit_mp4 = None
    if render:
        orig_mp4, tok_mp4, edit_mp4 = render_triple(person, tok_recon, edited, out_dir, fps=fps, size=size,
                                                    window_only=window_only,
                                                    include_tok_recon=show_tok_recon, log=log)

    return {
        "info": info,
        "source_pkl": Path(pkl_path),
        "input_pkl": input_pkl,
        "tokenizer_recon_pkl": tok_recon_pkl,
        "edited_pkl": edited_pkl,
        "original_mp4": orig_mp4,
        "tokenizer_recon_mp4": tok_mp4,
        "edited_mp4": edit_mp4,
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="run the tokenizer+infiller edit on one clip")
    ap.add_argument("input", help=".mp4 or WHAM .pkl")
    ap.add_argument("--out", default=str(REPO_ROOT / "demo/app/runs/cli"))
    ap.add_argument("--span", type=float, default=0.15)
    ap.add_argument("--splice", choices=["window", "mask"], default="window")
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--full-clip", action="store_true", help="render all frames, not just the 90-frame window")
    ap.add_argument("--show-tok-recon", action="store_true",
                    help="also render the tokenizer-only reconstruction (3rd video, no infiller)")
    a = ap.parse_args()

    res = run_pipeline(a.input, a.out, span_fraction=a.span, splice=a.splice,
                       render=not a.no_render, window_only=not a.full_clip,
                       show_tok_recon=a.show_tok_recon)
    print(res["info"])
    print("tokenizer recon pkl:", res["tokenizer_recon_pkl"])
    print("edited pkl:", res["edited_pkl"])