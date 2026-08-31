#codebook usage + MAE on the NEW expert training data itself -- separates
#"stage 1 training never used the codebook" from "this learner clip specifically is OOD"
import glob, joblib, numpy as np, torch
from torch.utils.data import DataLoader
from data_utils import smplPoseLoader
from pose_tokenizer import poseTokenizer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tok = poseTokenizer(hidden_dim=384, out_dim=256, num_joints=21, num_layers=6).to(device)
tok.load_state_dict(torch.load("demo/arch/tokenizer_ckpts_v2/pose_tokenizer_epoch_500.pth",
                                map_location=device)['model_state_dict'])
tok.eval()

expert = smplPoseLoader(root_dir="demo/basketball_expert_smpl_v2",
                         target_len=90, split='train')
loader = DataLoader(expert, batch_size=1, shuffle=True)

seen1, seen2, maes = set(), set(), []
for i, x in enumerate(loader):
    if i >= 50: break
    x = x.to(device)
    with torch.no_grad():
        recon, _, idx = tok(x)
    seen1.update(idx[..., 0].flatten().tolist())
    seen2.update(idx[..., 1].flatten().tolist())
    maes.append((recon[..., 6:69] - x[..., 6:69]).abs().mean().item())

print(f"expert data (n={min(50,len(expert))}): c1 {len(seen1)}/256, c2 {len(seen2)}/256")
print(f"expert body MAE: {np.mean(maes):.4f}")