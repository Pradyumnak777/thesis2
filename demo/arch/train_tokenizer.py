'''
traning loop for the pose tokenizer
'''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "9"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from data_utils import smplPoseLoader
from pose_tokenizer import poseTokenizer


if __name__ == "__main__":
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparameters
    ROOT_DIR = "demo/basketball_expert_smpl/Mid-range jump shot"
    TARGET_LEN = 90  # Tframes for jumpshot
    BATCH_SIZE = 16
    NUM_EPOCHS = 400
    LR = 5e-5
    SAVE_DIR = "demo/arch/tokenizer_ckpts"
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 1. Initialize Dataset & DataLoader
    dataset = smplPoseLoader(root_dir=ROOT_DIR, target_len=TARGET_LEN, split='train')
    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True
    )

    # 2. Instantiate Model and Optimizer
    model = poseTokenizer(
        hidden_dim=384,
        out_dim=256,
        num_joints=21,
        num_layers=6
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # 3. Training Loop
    print("Starting Stage 1: Pose Tokenizer Training...")
    model.train()

    for epoch in range(1, NUM_EPOCHS + 1):
        running_recon_loss = 0.0
        running_vq_loss = 0.0
        running_total_loss = 0.0

        for batch_idx, batch_x in enumerate(train_loader):
            # batch_x shape: [B, T, 69]
            batch_x = batch_x.to(device)

            optimizer.zero_grad()

            # Forward pass through Tokenizer
            x_recon, total_vq_loss, stacked_indices = model(batch_x)

            # Outer reconstruction loss (MSE between input motion and decoded motion)
            recon_loss = F.mse_loss(x_recon, batch_x)

            # Total loss = Recon Loss + Codebook/Commitment Loss
            total_loss = recon_loss + total_vq_loss

            # Backward pass & weight update
            total_loss.backward()
            optimizer.step()

            # Accumulate metrics
            running_recon_loss += recon_loss.item()
            running_vq_loss += total_vq_loss.item()
            running_total_loss += total_loss.item()

        # Epoch logging
        avg_recon = running_recon_loss / len(train_loader)
        avg_vq = running_vq_loss / len(train_loader)
        avg_total = running_total_loss / len(train_loader)

        print(f"Epoch [{epoch:03d}/{NUM_EPOCHS:03d}] | Total Loss: {avg_total:.6f} | Recon Loss: {avg_recon:.6f} | VQ Loss: {avg_vq:.6f}")

        # Save checkpoint periodically
        if epoch % 40 == 0 or epoch == NUM_EPOCHS:
            ckpt_path = os.path.join(SAVE_DIR, f"pose_tokenizer_epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_total,
            }, ckpt_path)
            print(f"--> Saved checkpoint to {ckpt_path}")

    print("pose tokenizer training done. next: train bidirectional transformer to make edits")