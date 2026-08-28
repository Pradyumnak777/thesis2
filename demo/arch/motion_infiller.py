'''
after pose tokenizer outputs the 2 IDs of the codebook vectors for each frame.
'''

import torch
import torch.nn as nn
from pose_tokenizer import positionalEncoding

class MotionInfiller(nn.Module):
    def __init__(self, vocab_size=256, emb_dim=256, hidden_dim=512, num_layers=12, nhead=8):
        super().__init__()
        self.vocab_size = vocab_size # 256 regular tokens + 1 mask token (id: 256)
        self.mask_token_id = vocab_size 
        
        # separate embedding tables for each codebook stream
        self.emb1 = nn.Embedding(self.vocab_size + 1, emb_dim)
        self.emb2 = nn.Embedding(self.vocab_size + 1, emb_dim)
        
        self.pos_encoder = positionalEncoding(d_model=hidden_dim)
        
        # 12 layer bidirectional transformer backbone
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)
        
        # classification heads to predict original token IDs (0 to 255)
        self.head1 = nn.Linear(hidden_dim, vocab_size)
        self.head2 = nn.Linear(hidden_dim, vocab_size)

    def forward(self, token_indices):
        """
        token_indices: [B, T, 2] containing integer IDs (or mask token ID 256)
        returns logits1: [B, T, 256], logits2: [B, T, 256]
        """
        tokens1 = token_indices[:, :, 0] # [B, T]
        tokens2 = token_indices[:, :, 1] # [B, T]
        
        e1 = self.emb1(tokens1) # [B, T, 256]
        e2 = self.emb2(tokens2) # [B, T, 256]
        
        h = torch.cat([e1, e2], dim=-1) # [B, T, 512]
        h = self.pos_encoder(h)
        
        # bidirectional pass (no causal mask)
        h_out = self.transformer(h) # [B, T, 512]
        
        logits1 = self.head1(h_out) # [B, T, 256]
        logits2 = self.head2(h_out) # [B, T, 256]
        
        return logits1, logits2 #this will be decoded by the decoder trained in stage 1, to get back the smpl pose data