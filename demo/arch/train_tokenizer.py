'''
traning loop for the pose tokenizer
'''
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "9"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from data_utils import smplPoseLoader
from pose_tokenizer import poseTokenizer
from itertools import cycle
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler


import numpy as np
if not hasattr(np, 'bool'):
    np.bool = bool
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'complex'):
    np.complex = complex
if not hasattr(np, 'object'):
    np.object = object
if not hasattr(np, 'str'):
    np.str = str
if not hasattr(np, 'unicode'):
    np.unicode = str

import smplx

import debugpy
# debugpy.listen(("127.0.0.1", 5678))
# print("Waiting for debugger attach on port 5678...")
# debugpy.wait_for_client()
# print("Debugger attached! Running code...")

'''
#NOTE: world size is hte number of GPUs and rank is the GPU indexes (can be local or global)

global and local ranks are identical if its a single machine..
'''
# os.environ["LOCAL_RANK"] = 
# os.environ["RANK"] = 
# os.environ["WORLD_SIZE"] = 

def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank
    
def cleanup_ddp():
    dist.destroy_process_group()

    

def get_vertices(body_model, trans, root_orient, body_pose_63, betas):
    """
    trans: [B, N, 3], root_orient: [B, N, 3], body_pose_63: [B, N, 63], betas: [B, 10]
    SMPL wants 23 joints (69 dims) in body_pose but we only track 21 (63 dims),
    so pad the missing 2 joints with zeros, same as smpl_viz.py does at export time.
    """
    B, N = trans.shape[0], trans.shape[1]
    pad_joints = torch.zeros(B, N, 6, device=trans.device, dtype=trans.dtype)
    body_pose_69 = torch.cat([body_pose_63, pad_joints], dim=-1)
    out = body_model(
        global_orient=root_orient.reshape(B * N, 3),
        body_pose=body_pose_69.reshape(B * N, 69),
        transl=trans.reshape(B * N, 3),
        betas=betas.unsqueeze(1).expand(-1, N, -1).reshape(B * N, -1),
    )
    return out.vertices.reshape(B, N, -1, 3)


def infinite_batches(loader, sampler=None):
    epoch = 0
    while True:
        if sampler is not None:
            sampler.set_epoch(epoch)   #reshuffle differently each pass
        for batch in loader:
            yield batch
        epoch += 1


if __name__ == "__main__":
    #for parallel
    local_rank = setup_ddp()
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if local_rank == 0:
        print(f"Using device: {device}")

    # Hyperparameters
    ROOT_DIR = "demo/basketball_expert_smpl_v2"
    TARGET_LEN = 90  # Tframes for jumpshot
    BATCH_SIZE = 64
    NUM_EPOCHS = 100
    LR = 5e-5
    SAVE_DIR = "demo/arch/tokenizer_ckpts_v3"
    RESUME_CKPT = os.path.join(SAVE_DIR, "pose_tokenizer_epoch_100.pth")
    # VERTEX_LOSS_FRAC = 0.15  # PoseGPT appendix: 10-20% of frames is sufficient
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 1. Initialize Dataset & DataLoader
    dataset = smplPoseLoader(root_dir=ROOT_DIR, target_len=TARGET_LEN, split='train')
    # train_loader = DataLoader(
    #     dataset,
    #     batch_size=BATCH_SIZE,
    #     shuffle=True,
    #     num_workers=2,
    #     pin_memory=True,
    #     drop_last=True
    # )
    
    # # 2. Instantiate Model and Optimizer
    # model = poseTokenizer(
    #     hidden_dim=384,
    #     out_dim=256,
    #     num_joints=21,
    #     num_layers=6
    # ).to(device)
    

    
    '''
    below for distributed/parallel
    '''
    
    train_sampler = DistributedSampler(
        dataset, 
        num_replicas=dist.get_world_size(), 
        rank=dist.get_rank(), #get the GPU its on?
        shuffle=True
    )

    
    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        sampler=train_sampler
    )

    
    
    '''
    #NOTE: sending model to DDP
    '''
    
    model = poseTokenizer(
        hidden_dim=384,
        out_dim=256,
        num_joints=21,
        num_layers=6
    ).to(local_rank) #sends to some GPU
    
    '''
    code for resuming from checkpoint
    '''
    start_epoch = 1
    if RESUME_CKPT and os.path.exists(RESUME_CKPT):
        ckpt = torch.load(RESUME_CKPT, map_location=f"cuda:{local_rank}")
        model.load_state_dict(ckpt['model_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        if local_rank == 0:
            print(f"Resumed tokenizer from {RESUME_CKPT} (epoch {ckpt['epoch']})")

    
    
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # PoseGPT-style vertex loss: frozen SMPL body model turns predicted/GT pose
    # params into 3D vertices, so the reconstruction loss isn't fooled by
    # axis-angle representation ambiguity
    # body_model = smplx.create(
    #     model_path="demo/GMR/assets/body_models",
    #     model_type="smpl",
    #     gender="neutral",
    # ).to(local_rank)
    # body_model.eval()
    # for param in body_model.parameters():
    #     param.requires_grad = False
        
    '''
    below is for step/iter based training loop
    '''
    
    count = 0
    steps = 1000
    train_iter = infinite_batches(train_loader, train_sampler)
    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        # train_sampler.set_epoch(epoch) #NOTE: seems to be imp!!
        
        
        running_recon_loss = 0.0
        running_vq_loss = 0.0
        running_total_loss = 0.0
        while count < steps:
            
            batch_x, _ = next(train_iter) #normally "betas" is returned but im not using vertex loss this time..
            batch_x = batch_x.to(local_rank)
            
            optimizer.zero_grad()

            # Forward pass through Tokenizer
            x_recon, total_vq_loss, stacked_indices = model(batch_x)

            # Outer reconstruction loss (MSE between input motion and decoded motion)
            recon_loss = F.mse_loss(x_recon, batch_x)

            total_loss = recon_loss + total_vq_loss

            # Backward pass & weight update
            total_loss.backward()
            optimizer.step()

            # Accumulate metrics
            running_recon_loss += recon_loss.item()
            # running_vertex_loss += vertex_loss.item()
            running_vq_loss += total_vq_loss.item()
            running_total_loss += total_loss.item()
            
            count += 1
            
            #logging after steps
            if local_rank == 0 and count % 50 == 0: #printing only from GPU 1
                avg_recon_mini = running_recon_loss / count 
                # avg_vertex = running_vertex_loss / len(train_loader)
                avg_vq_mini = running_vq_loss / count
                avg_total_mini = running_total_loss / count 
                print(f"epoch: {epoch}, step: {count}")
                print(f"Total Loss: {avg_total_mini:.6f} | Recon Loss: {avg_recon_mini:.6f} | VQ Loss: {avg_vq_mini:.6f}")

        # Epoch logging
        count = 0
        avg_recon = running_recon_loss / steps #iters per epoch
        # avg_vertex = running_vertex_loss / len(train_loader)
        avg_vq = running_vq_loss / steps
        avg_total = running_total_loss / steps 

        # print(f"Epoch [{epoch:03d}/{NUM_EPOCHS:03d}] | Total Loss: {avg_total:.6f} | Recon Loss: {avg_recon:.6f} | Vertex Loss: {avg_vertex:.6f} | VQ Loss: {avg_vq:.6f}")
        

        # Save checkpoint periodically
        if dist.get_rank() == 0:
            print(f"Epoch [{epoch:03d}/{NUM_EPOCHS:03d}] | Total Loss: {avg_total:.6f} | Recon Loss: {avg_recon:.6f} | VQ Loss: {avg_vq:.6f}")
            # Use .module to strip the DDP wrapper before saving

            if epoch % 10 == 0 or epoch == NUM_EPOCHS:
                if local_rank == 0:
                    ckpt_path = os.path.join(SAVE_DIR, f"pose_tokenizer_epoch_{epoch}.pth")
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': avg_total,
                    }, ckpt_path)
                    print(f"--> Saved checkpoint to {ckpt_path}")

    cleanup_ddp()
    
    '''
    below is for standard training loop
    '''
    # # 3. Training Loop
    # print("Starting Stage 1: Pose Tokenizer Training...")
    # model.train()

    # for epoch in range(1, NUM_EPOCHS + 1):
    #     running_recon_loss = 0.0
    #     running_vertex_loss = 0.0
    #     running_vq_loss = 0.0
    #     running_total_loss = 0.0

    #     for batch_idx, (batch_x, batch_betas) in enumerate(train_loader):
    #         # batch_x shape: [B, T, 69], batch_betas shape: [B, 10]
    #         batch_x = batch_x.to(device)
    #         batch_betas = batch_betas.to(device)

    #         optimizer.zero_grad()

    #         # Forward pass through Tokenizer
    #         x_recon, total_vq_loss, stacked_indices = model(batch_x)

    #         # Outer reconstruction loss (MSE between input motion and decoded motion)
    #         recon_loss = F.mse_loss(x_recon, batch_x)

    #         # # Vertex loss on a random subset of frames (PoseGPT: 10-20% is enough)
    #         # T = batch_x.shape[1]
    #         # n_sample = max(1, int(T * VERTEX_LOSS_FRAC))
    #         # frame_idx = torch.randperm(T, device=device)[:n_sample]

    #         # gt_verts = get_vertices(
    #         #     body_model,
    #         #     batch_x[:, frame_idx, 0:3], batch_x[:, frame_idx, 3:6],
    #         #     batch_x[:, frame_idx, 6:69], batch_betas
    #         # )
    #         # pred_verts = get_vertices(
    #         #     body_model,
    #         #     x_recon[:, frame_idx, 0:3], x_recon[:, frame_idx, 3:6],
    #         #     x_recon[:, frame_idx, 6:69], batch_betas
    #         # )
    #         # vertex_loss = F.mse_loss(pred_verts, gt_verts)

    #         # Total loss = Recon Loss + Vertex Loss + Codebook/Commitment Loss
    #         total_loss = recon_loss + total_vq_loss

    #         # Backward pass & weight update
    #         total_loss.backward()
    #         optimizer.step()

    #         # Accumulate metrics
    #         running_recon_loss += recon_loss.item()
    #         # running_vertex_loss += vertex_loss.item()
    #         running_vq_loss += total_vq_loss.item()
    #         running_total_loss += total_loss.item()

    #     # Epoch logging
    #     avg_recon = running_recon_loss / len(train_loader)
    #     # avg_vertex = running_vertex_loss / len(train_loader)
    #     avg_vq = running_vq_loss / len(train_loader)
    #     avg_total = running_total_loss / len(train_loader)

    #     # print(f"Epoch [{epoch:03d}/{NUM_EPOCHS:03d}] | Total Loss: {avg_total:.6f} | Recon Loss: {avg_recon:.6f} | Vertex Loss: {avg_vertex:.6f} | VQ Loss: {avg_vq:.6f}")
    #     print(f"Epoch [{epoch:03d}/{NUM_EPOCHS:03d}] | Total Loss: {avg_total:.6f} | Recon Loss: {avg_recon:.6f} | VQ Loss: {avg_vq:.6f}")

    #     # Save checkpoint periodically
    #     if epoch % 50 == 0 or epoch == NUM_EPOCHS:
    #         ckpt_path = os.path.join(SAVE_DIR, f"pose_tokenizer_epoch_{epoch}.pth")
    #         torch.save({
    #             'epoch': epoch,
    #             'model_state_dict': model.state_dict(),
    #             'optimizer_state_dict': optimizer.state_dict(),
    #             'loss': avg_total,
    #         }, ckpt_path)
    #         print(f"--> Saved checkpoint to {ckpt_path}")

    print("pose tokenizer training done. next: train bidirectional transformer to make edits")