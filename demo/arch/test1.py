import os
import torch
import numpy as np
import joblib

from pose_tokenizer import poseTokenizer
from data_utils import smplPoseLoader

def test_autoencoder_reconstruction(
    checkpoint_path="demo/arch/tokenizer_ckpts_v2/pose_tokenizer_epoch_500.pth",
    input_motion_path="demo/arch/inference_test/learner_exo/wham_output_selected.pkl",
    output_recon_path="demo/arch/recon_test_motion.pkl",
    target_len=90,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    print(f"[*] Loading model checkpoint: {checkpoint_path}")
    # 1. Load Trained Pose Tokenizer
    model = poseTokenizer().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # 2. Load and Preprocess Input Motion using smplPoseLoader
    print(f"[*] Loading unedited WHAM motion: {input_motion_path}")
    with open(input_motion_path, 'rb') as f:
        raw_data = joblib.load(f)

    # Use smplPoseLoader's formatting to extract [trans (3), root_orient (3), body_pose (63)] -> (T, 69)
    loader_helper = smplPoseLoader.__new__(smplPoseLoader)
    motion_seq = loader_helper._extract_and_format_poses(raw_data)  # shape: (T_raw, 69)
    seq_len = motion_seq.shape[0]

    # Adjust sequence to target length (90 frames)
    if seq_len >= target_len:
        start = (seq_len - target_len) // 2
        motion_seq = motion_seq[start : start + target_len]
    else:
        pad_len = target_len - seq_len
        last_frame = np.tile(motion_seq[-1:], (pad_len, 1))
        motion_seq = np.concatenate([motion_seq, last_frame], axis=0)

    orig_poses = torch.tensor(motion_seq, dtype=torch.float32)
    print(f"    Processed input tensor shape: {orig_poses.shape}")  # (90, 69)
    print(f"    Input range: min={orig_poses.min().item():.3f}, max={orig_poses.max().item():.3f}")

    # 3. Model Forward Pass
    motion_in = orig_poses.unsqueeze(0).to(device)  # shape: (1, 90, 69)

    with torch.no_grad():
        if hasattr(model, "encode") and hasattr(model, "decode"):
            tokens = model.encode(motion_in)
            recon_output = model.decode(tokens)
        else:
            # Fallback if model forward returns (reconstructed, vq_loss, tokens)
            outputs = model(motion_in)
            recon_output = outputs[0]
            tokens = outputs[2] if len(outputs) > 2 else outputs[1]

    # 4. Compute Metrics
    recon_poses = recon_output.squeeze(0).cpu()  # shape: (90, 69)
    l1_err = torch.nn.functional.l1_loss(recon_poses, orig_poses).item()
    mse_err = torch.nn.functional.mse_loss(recon_poses, orig_poses).item()

    print("\n--- Diagnostic Results ---")
    print(f"Reconstruction L1 Error  : {l1_err:.4f}")
    print(f"Reconstruction MSE Error : {mse_err:.4f}")
    print(f"Token Sequence Shape     : {tokens.shape if isinstance(tokens, torch.Tensor) else len(tokens)}")

    if isinstance(tokens, torch.Tensor):
        flat_tokens = tokens.flatten().cpu().numpy()
        unique_tokens = np.unique(flat_tokens)
        print(f"Unique Codebook Usage    : {len(unique_tokens)} active tokens out of {tokens.numel()} sequence tokens")

    # 5. Unpack and Save Back to WHAM Format for smpl_viz.py
    recon_np = recon_poses.numpy()  # (90, 69)
    
    # Slice according to _extract_and_format_poses:
    # 0:3 -> trans_world
    # 3:6 -> root orientation (joint 0)
    # 6:69 -> body pose (joints 1..21 = 63 dims)
    recon_trans = recon_np[:, :3]
    recon_root = recon_np[:, 3:6]
    recon_body = recon_np[:, 6:69]

    # Pad the missing 2 joints (6 dimensions for hands/wrists) to reach full 24 SMPL joints (72 dims)
    pad_joints = np.zeros((recon_body.shape[0], 6), dtype=np.float32)
    recon_full_pose = np.concatenate([recon_root, recon_body, pad_joints], axis=-1)  # shape: (90, 72)

    out_dict = raw_data.copy() if isinstance(raw_data, dict) else {}
    out_dict["trans_world"] = recon_trans
    out_dict["pose_world"] = recon_full_pose

    os.makedirs(os.path.dirname(output_recon_path), exist_ok=True)
    joblib.dump(out_dict, output_recon_path)
    print(f"\n[+] Successfully saved unpacked motion to: {output_recon_path}")
    print(f"[+] Run visualizer: python demo/smpl_viz.py --input {output_recon_path}\n")

if __name__ == "__main__":
    test_autoencoder_reconstruction()