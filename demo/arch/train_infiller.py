'''
stage 2 training: motion infiller with kinematic peak masking
'''

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "9"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data_utils import smplPoseLoader
from pose_tokenizer import poseTokenizer
from motion_infiller import MotionInfiller
from itertools import cycle
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_DDP():
    #define local rank?
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_DDP():
    dist.destroy_process_group()

def get_kinematic_peaks(motion_batch):
    """
    motion_batch: [B, T, 69], where root translation is at indices 0:3
    y-axis is index 1 (vertical translation)
    """
    # vertical root translation r_y: [B, T]
    root_y = motion_batch[:, :, 1]
    
    # compute vertical velocity v_y(t) = r_y(t) - r_y(t-1) -> shape: [B, T-1]
    v_y = root_y[:, 1:] - root_y[:, :-1]
    
    # find apex/takeoff frame index for each clip in the batch
    peak_indices = torch.argmax(v_y, dim=-1) + 1  # [B]
    return peak_indices


def create_masked_inputs(clean_tokens, peak_indices, mask_token_id=256, span_radius=8):
    """
    clean_tokens: [B, T, 2]
    peak_indices: [B]
    replaces tokens in window [t* - span, t* + span] with mask_token_id (256)
    """
    B, T, _ = clean_tokens.shape
    masked_tokens = clean_tokens.clone()
    mask_labels = torch.zeros((B, T), dtype=torch.bool, device=clean_tokens.device)

    for b in range(B):
        t_star = peak_indices[b].item()
        start = max(0, t_star - span_radius)
        end = min(T, t_star + span_radius + 1)
        
        masked_tokens[b, start:end, :] = mask_token_id
        mask_labels[b, start:end] = True
        
    return masked_tokens, mask_labels


if __name__ == "__main__":
    local_rank = setup_DDP() #this will be the gpu number
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if local_rank == 0:
        print(f"Using device: {device}")

    # hyperparams
    ROOT_DIR = "demo/basketball_expert_smpl_v2"
    TARGET_LEN = 90
    BATCH_SIZE = 16
    NUM_EPOCHS = 800
    LR = 1e-4
    TOKENIZER_CKPT = "demo/arch/tokenizer_ckpts_v3/pose_tokenizer_epoch_250.pth"
    SAVE_DIR = "demo/arch/infiller_ckpts_v3"
    os.makedirs(SAVE_DIR, exist_ok=True)

    
    
    # 1. dataloader
    dataset = smplPoseLoader(root_dir=ROOT_DIR, target_len=TARGET_LEN, split='train')
    
    #NOTE: defining a distirbuted sampler
    
    train_sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank = dist.get_rank(),
        shuffle=True
    )
    
    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        sampler=train_sampler
    )
    
    # train_loader = DataLoader(
    #     dataset,
    #     batch_size=BATCH_SIZE,
    #     shuffle=True,
    #     num_workers=2,
    #     pin_memory=True,
    #     drop_last=True
    # )

    # 2. load and freeze stage 1 tokenizer
    tokenizer = poseTokenizer(hidden_dim=384, out_dim=256, num_joints=21, num_layers=6).to(local_rank)
    # tokenizer = DDP(tokenizer, device_ids=[local_rank], output_device=local_rank) # no need as its frozen
    
    if os.path.exists(TOKENIZER_CKPT):
        ckpt = torch.load(TOKENIZER_CKPT, map_location=f"cuda:{local_rank}")
        tokenizer.load_state_dict(ckpt['model_state_dict'])
        if local_rank == 0:
            print(f"Loaded frozen tokenizer from {TOKENIZER_CKPT}")
    else:
        if local_rank == 0:
            print(f"Warning: {TOKENIZER_CKPT} not found. Running with uninitialized tokenizer.")
    
    tokenizer.eval()
    for param in tokenizer.parameters():
        param.requires_grad = False

    # 3. initialize stage 2 infiller model & optimizer
    infiller = MotionInfiller(vocab_size=256, emb_dim=128, hidden_dim=256, num_layers=12, nhead=8).to(local_rank)
    infiller = DDP(infiller, device_ids=[local_rank], output_device=local_rank)
    
    optimizer = torch.optim.AdamW(infiller.parameters(), lr=LR, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()

    print("Starting Stage 2: Motion Infiller Training...")
    infiller.train()

    count = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        train_sampler.set_epoch(epoch)
        train_iter = cycle(train_loader)
        
        running_loss = 0.0
        running_acc1 = 0.0
        running_acc2 = 0.0
        total_masked_tokens = 0
        
        while count < 5000:
            batch_x, _ = next(train_iter)
            batch_x = batch_x.to(local_rank)  # [B, T, 69]

            # extract discrete tokens using frozen tokenizer
            with torch.no_grad():
                _, _, clean_indices = tokenizer(batch_x)  # [B, T, 2]
                peak_indices = get_kinematic_peaks(batch_x) # [B]
                
            # create masked input tokens around kinematic peak t*
            masked_tokens, mask_labels = create_masked_inputs(
                clean_indices, peak_indices, mask_token_id=256, span_radius=8
            )

            optimizer.zero_grad()

            # forward pass through infiller
            logits1, logits2 = infiller(masked_tokens)  # [B, T, 256], [B, T, 256]

            # select only the masked positions for cross entropy
            target1 = clean_indices[:, :, 0][mask_labels]  # [N_masked]
            target2 = clean_indices[:, :, 1][mask_labels]  # [N_masked]
            
            pred1 = logits1[mask_labels]  # [N_masked, 256]
            pred2 = logits2[mask_labels]  # [N_masked, 256]

            # cross entropy loss on masked frames
            loss1 = criterion(pred1, target1)
            loss2 = criterion(pred2, target2)
            total_loss = loss1 + loss2

            total_loss.backward()
            optimizer.step()

            # compute accuracy on masked tokens
            with torch.no_grad():
                acc1 = (pred1.argmax(dim=-1) == target1).float().sum().item()
                acc2 = (pred2.argmax(dim=-1) == target2).float().sum().item()
                num_m = target1.size(0)

                running_loss += total_loss.item()
                running_acc1 += acc1
                running_acc2 += acc2
                total_masked_tokens += num_m

            count+=1

            if local_rank == 0 and count % 100 == 0: #printing only from GPU 1
                avg_loss_mini = running_loss / count
                avg_acc1_mini = (running_acc1 / total_masked_tokens) * 100
                avg_acc2_mini = (running_acc2 / total_masked_tokens) * 100
                print(f"epoch: {epoch}, count: {count}")
                print(f"MLM Loss: {avg_loss_mini:.4f} | Codebook 1 Acc: {avg_acc1_mini:.2f}% | Codebook 2 Acc: {avg_acc2_mini:.2f}%")

        # save periodic checkpoints
        count = 0
        
        if dist.get_rank() == 0:
            #logging
            avg_loss = running_loss / 5000
            avg_acc1 = (running_acc1 / total_masked_tokens) * 100
            avg_acc2 = (running_acc2 / total_masked_tokens) * 100
            print(f"====epoch: {epoch}=====")
            print(f"MLM Loss: {avg_loss:.4f} | Codebook 1 Acc: {avg_acc1:.2f}% | Codebook 2 Acc: {avg_acc2:.2f}%")
            
            if epoch % 80 == 0 or epoch == NUM_EPOCHS:
                
                ckpt_path = os.path.join(SAVE_DIR, f"motion_infiller_epoch_{epoch}.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': infiller.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                }, ckpt_path)
                print(f"--> Saved infiller checkpoint to {ckpt_path}")

    cleanup_DDP()
    print("Stage 2 Training Complete. Both models ready for inference and editing!")