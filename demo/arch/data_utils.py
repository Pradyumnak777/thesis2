'''
example of path to a file:
thesis_new/demo/basketball_expert_smpl/Mid-range jump shot/sfu_basketball_03_12__7.467-9.467__cam01/wham_output_selected.pkl
'''
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "9"

import glob
import joblib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

ROOT = "demo/basketball_expert_smpl/Mid-range jump shot"

class smplPoseLoader(Dataset):
    def __init__(self, root_dir=ROOT, target_len=90, split='train'):
        super().__init__()
        self.root_dir = root_dir
        self.target_len = target_len  #fixed sequence length T (e.g. 90 frames for jumpshot)
        self.split = split
        
        # 1. Find all wham_output_selected.pkl files inside the action subdirectories
        # Example pattern: demo/basketball_expert_smpl/Mid-range jump shot/*/wham_output_selected.pkl
        search_pattern = os.path.join(self.root_dir, "*", "wham_output_selected.pkl")
        self.file_list = sorted(glob.glob(search_pattern))
        
        if len(self.file_list) == 0:
            # Fallback in case clips are nested one more folder down
            search_pattern = os.path.join(self.root_dir, "**", "wham_output_selected.pkl")
            self.file_list = sorted(glob.glob(search_pattern, recursive=True))
            
        print(f"[{split.upper()}] Found {len(self.file_list)} pose sequence clips in {self.root_dir}")

    def __len__(self):
        return len(self.file_list)

    def _extract_and_format_poses(self, pkl_data):
        """
        Extracts translation, root orientation, and joint rotations from WHAM pkl dict.
        Output vector x_t is [trans (3), orient (3), body_poses (63)] -> dim 69
        """
        # 1. Global translation (3,)
        trans = pkl_data['trans_world']
            
        # 2. Root orientation (3,)
        root_orient = pkl_data['pose_world'][:,:3]
        
        # 3. Body joint rotations (21 * 3 = 63,) -> or it could be 24 too(?)
        body_pose = pkl_data['pose_world'][:,3:66]
        
        # Flatten all components if they have extra singleton dimensions
        trans = trans.reshape(trans.shape[0], 3)
        root_orient = root_orient.reshape(root_orient.shape[0], 3)
        body_pose = body_pose.reshape(body_pose.shape[0], -1)  # should be (T, 63)
        
        # Concatenate along feature dimension -> [T_raw, 69]
        motion_sequence = np.concatenate([trans, root_orient, body_pose], axis=-1)
        return motion_sequence.astype(np.float32)

    def _extract_betas(self, pkl_data):
        """
        SMPL shape params for this clip's subject, needed for the PoseGPT-style
        vertex loss (mesh forward pass). WHAM stores one betas vector per frame;
        average them since shape is constant for a subject within a clip.
        """
        betas = pkl_data['betas']
        betas = betas.mean(axis=0) if betas.ndim == 2 else betas
        return betas.astype(np.float32)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        
        with open(file_path, 'rb') as f:
            data = joblib.load(f)
            
        motion = self._extract_and_format_poses(data)  # [T_raw, 69]
        betas = self._extract_betas(data)  # [10]
        seq_len = motion.shape[0]
        
        # 2. Adjust sequence to fixed target_len (T = 90)
        if seq_len >= self.target_len:
            # If sequence is longer, take a slice centered around the action or random crop
            if self.split == 'train':
                start = np.random.randint(0, seq_len - self.target_len + 1)
            else:
                start = (seq_len - self.target_len) // 2
            motion = motion[start : start + self.target_len]
        else:
            # If sequence is shorter, repeat pad the final frame to fill target_len
            pad_len = self.target_len - seq_len
            last_frame = np.tile(motion[-1:], (pad_len, 1))
            motion = np.concatenate([motion, last_frame], axis=0)
            
        # Convert to float tensor: [target_len, 69], and betas: [10]
        return torch.tensor(motion, dtype=torch.float32), torch.tensor(betas, dtype=torch.float32)


def get_dataloader(root_dir=ROOT, target_len=90, batch_size=16, shuffle=True, num_workers=2):

    dataset = smplPoseLoader(root_dir=root_dir, target_len=target_len, split='train')
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    return loader


if __name__ == "__main__":
    print("Testing data_utils.py...")
    
    '''
    getting dataset size
    '''
    dataset = smplPoseLoader(root_dir = ROOT)
    print(len(dataset)) #number of pose sequences with length 90 frames (THIS is fed into the tokenizer)
    
    # loader = get_dataloader(batch_size=4)
    # for batch in loader:
    #     print("Batch motion shape:", batch.shape) # Expected: [4, 90, 69]
    #     break