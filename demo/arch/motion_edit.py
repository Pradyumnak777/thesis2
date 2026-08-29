'''
this is inference visualization. run an actual pose and get the edit pose?
1. load video
3. run wham and get the pose pkl
3. get the operative moment using the formula (pelvis movement)
4. pass thru model
'''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "9"


import subprocess
from pathlib import Path
import numpy as np
import joblib
# from ..inspect_pose_demo import jumphot_heuristic
from torch.utils.data import DataLoader
from data_utils import smplPoseLoader
from pose_tokenizer import poseTokenizer
import torch
import torch.nn as nn
from motion_infiller import MotionInfiller
from train_infiller import get_kinematic_peaks, create_masked_inputs

VID_PATH = Path("dataset_prep/dataset_out/val/unc_basketball_02-24-23_01_34-----18-----unc_basketball_03-30-23_02_4-----0-----Arms/learner_exo.mp4")
OUT_DIR = Path("demo/arch/inference_test") 
TOKENIZER_CKPT = "demo/arch/tokenizer_ckpts/pose_tokenizer_epoch_400.pth"
INFILLER_CKPT = "demo/arch/infiller_ckpts/motion_infiller_epoch_140.pth"
# OUT_DIR = (OUT_DIR / VID_PATH.stem).resolve()

def jumphot_heuristic(BASE):#base is the path to the smpl directory, NOT the smpl file itself!
    wham_output      = joblib.load(f"{BASE}/wham_output.pkl")
    slam_results     = joblib.load(f"{BASE}/slam_results.pth")
    tracking_results = joblib.load(f"{BASE}/tracking_results.pth")

    track_ids = list(wham_output.keys())
    people = list(wham_output.values())

    if len(people) == 1:
        best_id = track_ids[0]
    else:
        '''
        some heuristic needs to be defined
        -> jumpshot: pelvis in y axis(?)
        '''
        pelvis_trans = []
        for person in people:
            trans_world = person["trans_world"]  # (frames, 3)
            vertical_trans = trans_world[:,1] # y axis
            
            #find difference between min and max
            jump_displacement = np.percentile(vertical_trans, 95) - np.percentile(vertical_trans, 5)

            pelvis_trans.append(jump_displacement.sum())
        
        best_id = track_ids[int(np.argmax(pelvis_trans))]

    return best_id, wham_output



if __name__ == "__main__":
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ACTOR_DIR = OUT_DIR / VID_PATH.stem
    selected_pkl = ACTOR_DIR / "wham_output_selected.pkl"
    WHAM_DIR = Path("WHAM").resolve()

    if not selected_pkl.exists():
        subprocess.run(
            [
                "python", "demo.py",
                "--video", str(VID_PATH.resolve()),
                "--output_pth", str(OUT_DIR.resolve()),
                "--save_pkl",
            ],
            cwd=str(WHAM_DIR),
            check=True,
        )
        '''
        simple heuristic to select only actor pose
        '''
        best_idx, wham_output = jumphot_heuristic(str(ACTOR_DIR))
        selected_person = wham_output[best_idx]
        joblib.dump(selected_person, selected_pkl)
    else:
        selected_person = joblib.load(selected_pkl)
    
    '''
    now, run this through the model
    '''
    dataset = smplPoseLoader(root_dir=str(ACTOR_DIR), target_len=90, split='test')
    train_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        drop_last=False
    )
    
    #initialiing both models-
    tokenizer = poseTokenizer(hidden_dim=384, out_dim=256, num_joints=21, num_layers=6).to(device)
    if os.path.exists(TOKENIZER_CKPT):
        ckpt = torch.load(TOKENIZER_CKPT, map_location=device)
        tokenizer.load_state_dict(ckpt['model_state_dict'])
        print(f"Loaded frozen tokenizer from {TOKENIZER_CKPT}")
    else:
        print(f"Warning: {TOKENIZER_CKPT} not found. Running with uninitialized tokenizer.")
    
    tokenizer.eval()
    for param in tokenizer.parameters():
        param.requires_grad = False
        
    infiller = MotionInfiller(vocab_size=256, emb_dim=256, hidden_dim=512, num_layers=12, nhead=8).to(device)
    ckpt_inf = torch.load(INFILLER_CKPT, map_location=device)
    infiller.load_state_dict(ckpt_inf['model_state_dict'])
    infiller.eval()
    for param in infiller.parameters():
        param.requires_grad = False
    print(f"Loaded infiller from {INFILLER_CKPT}")
    
    '''
    run the pipeline
    '''
    for _, data in enumerate(train_loader):
        data = data.to(device)
    
        with torch.no_grad():
            _, _, clean_indices = tokenizer(data)  # [B, T, 2]
            peak_indices = get_kinematic_peaks(data) # [B]   
        
        # create masked input tokens around kinematic peak t*
        masked_tokens, mask_labels = create_masked_inputs(
            clean_indices, peak_indices, mask_token_id=256, span_radius=8
        )
        
        logits1, logits2 = infiller(masked_tokens)  # [B, T, 256], [B, T, 256]
        '''
        #NOTE: logits output is:
        For every frame $t$, the model outputs 256 floating-point values representing how confident it is 
        that the frame belongs to token 0, token 1, token 2, ..., all the way to token 255
        '''
        pred_tokens1 = torch.argmax(logits1, dim=-1)  # [1, 90]
        pred_tokens2 = torch.argmax(logits2, dim=-1)  # [1, 90]
        pred_tokens = torch.stack([pred_tokens1, pred_tokens2], dim=-1)  # [1, 90, 2]
        
        # Splice: keep novice tokens outside mask, insert predicted expert tokens inside mask
        edited_tokens = clean_indices.clone()
        edited_tokens[mask_labels] = pred_tokens[mask_labels]
        
        '''
        pass thru decoder
        '''
        #Lookup continuous vectors in frozen Stage 1 codebooks
        z_q1 = tokenizer.c1(edited_tokens[:, :, 0])  # [1, 90, 128]
        z_q2 = tokenizer.c2(edited_tokens[:, :, 1])  # [1, 90, 128]
        z_q = torch.cat([z_q1, z_q2], dim=-1)         # [1, 90, 256]
        
        h_dec = tokenizer.decoder_proj(z_q)
        h_dec = tokenizer.pos_encoder(h_dec)
        h_dec_out = tokenizer.transformer_decoder(h_dec)
        edited_motion = tokenizer.output_proj_decoder(h_dec_out)
        
        # [1, 90, 69] -> trans (3), root_orient (3), body_pose (63)
        edited_motion_np = edited_motion.squeeze(0).cpu().numpy()
        edited_body_poses = edited_motion_np[:, 6:69]

        full_trans = selected_person['trans_world']
        full_pose = selected_person['pose_world']
        
        full_root_orient = full_pose[:, :3]
        full_body_original = full_pose[:, 3:66]
        full_body_edited = full_body_original.copy()

        t_raw = full_body_original.shape[0]
        target_len = 90

        if t_raw >= target_len:
            start_frame = (t_raw - target_len) // 2
            end_frame = start_frame + target_len
            full_body_edited[start_frame:end_frame] = edited_body_poses[:(end_frame - start_frame)]
        else:
            full_body_edited[:t_raw] = edited_body_poses[:t_raw]
        
        edited_pose_world = full_pose.copy()
        edited_pose_world[:, 3:66] = full_body_edited
        
        output_dict = {
            'trans_world': full_trans,
            # 'root_orient': full_root_orient,
            'pose_world': edited_pose_world,
            'betas': selected_person.get('betas', None)
        }

        save_file = ACTOR_DIR / "edited_motion_smpl.pkl"
        joblib.dump(output_dict, save_file)
        print(f"Saved edited motion parameters to {save_file}")
        break