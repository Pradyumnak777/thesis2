'''
building a pose tokenizer which has-
1. a causal transformer encoder
2. the codebook/quantization
3. a non causal decoder
'''

import torch
import torch.nn as nn
import math

class positionalEncoding(nn.Module):
    def __init__(self, d_model, max_len = 500):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        
    def forward(self, x):
        """
        input(x) is [B, T, d_model]
        output is [B, T, d_model], but with positional embeddings added
        """
        seq_len = x.size(1)
        # Slice up to length T and broadcast add across the batch dimension
        return x + self.pos_embedding[:, :seq_len, :] #as seg_len <= max_len
        
class poseTokenizer(nn.Module):
    def __init__(self, hidden_dim = 384, out_dim = 256, num_joints = 21, num_layers = 6): #the codebook/quantization happens on the output dim
        super().__init__()
        self.input_dim = 3 + 3 + (num_joints*3) #this is the input ot the transformer. this is converted to hidden_dim via a projection
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        
        '''
        defining encoder layers below
        '''
        self.encoder_proj = nn.Linear(self.input_dim, self.hidden_dim)
        self.pos_encoder = positionalEncoding(d_model=self.hidden_dim)
        
        #now, there's 6 layers. defining backbone first
        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=4,
            dim_feedforward=self.hidden_dim*4,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer, num_layers=num_layers)
        self.output_proj_encoder = nn.Linear(self.hidden_dim, out_dim) #this si what goes into the codebook/quantization
        
        '''
        defining the codelock here
        '''
        #creating 2 codebooks with 256 entries each
        self.c1 = nn.Embedding(num_embeddings=256, embedding_dim=128) #creates a learnable table/matrix
        self.c2 = nn.Embedding(num_embeddings=256, embedding_dim=128)
        #weight initialization
        self.c1.weight.data.uniform_(-1.0 / 256, 1.0 / 256)
        self.c2.weight.data.uniform_(-1.0 / 256, 1.0 / 256)
        
        '''
        defining decoder below
        '''
        #takes in a 256 dim input
        self.decoder_proj = nn.Linear(self.out_dim, self.hidden_dim)
        # self.pos_encoder = positionalEncoding(d_model=self.hidden_dim)
        #now, there's 6 layers. defining backbone first
        transformer_decoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=4,
            dim_feedforward=self.hidden_dim*4,
            activation='gelu',
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerEncoder(transformer_decoder_layer, num_layers=num_layers)
        #outputs the original input (6 +3j)
        self.output_proj_decoder = nn.Linear(self.hidden_dim, self.input_dim)
        
        

    def quantize(self, z, codebook): #function for a single codebook
        #calculate z^2
        B, T, D = z.shape
        z_flat = z.reshape(-1, D) #(B*T, D)
        z_sq = torch.sum(z_flat ** 2, dim=-1, keepdim=True) #[B*T, 1]
        
        #calc e^2, for codebook -> [1, K]
        e_sq = torch.sum(codebook.weight ** 2, dim=-1, keepdim = True).t() #for transposing
        
        #(z . e): shape [B*T, K]
        ze_dot = torch.matmul(z_flat, codebook.weight.t())
        
        #l2 dist
        dist = z_sq + e_sq - 2 * ze_dot
        
        indices = torch.argmin(dist, dim=-1) #(b*t)
        #lookup
        z_q = codebook(indices).view(B, T, D)
        
        #loss- codebook + commitment
        loss = torch.mean((z_q - z.detach()) ** 2) + 0.25 * torch.mean((z_q.detach() - z) ** 2)
        
        #straight thru estimator(for backprop trick)
        z_q = z + (z_q - z).detach()
        
        return z_q, loss, indices.view(B, T)
        
        
        
    def forward(self, x):
        '''
        thru the encoder first
        '''
        #input= [B, T, in_dim], 'T' is frames/pose sequence
        T = x.shape[1]
        h = self.encoder_proj(x)  # [B, T, d_model]
        h = self.pos_encoder(h) #adding position info before self attn
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        h_out = self.transformer_encoder(h, mask=causal_mask, is_causal=True)  # [B, T, d_model]
        z_e = self.output_proj_encoder(h_out)

        '''
        codebook/quantization part
        '''
        #"map" z_e to a codebook vector (how?)
        
        #splitting z_e as there are 2 codebooks
        z_e1, z_e2 = torch.chunk(z_e, chunks=2, dim=-1) #[B, T, 256] -> [B, T, 128]
        
        z_q1, vq_loss1, indices1 = self.quantize(z_e1, self.c1)
        z_q2, vq_loss2, indices2 = self.quantize(z_e2, self.c2)
        
        # Concatenate quantized vectors and sum VQ losses
        z_q = torch.cat([z_q1, z_q2], dim=-1)  # [B, T, 256]
        total_vq_loss = vq_loss1 + vq_loss2
        stacked_indices = torch.stack([indices1, indices2], dim=-1)
        
        '''
        thorugh the decoder
        '''
        h_dec = self.decoder_proj(z_q)
        h_dec = self.pos_encoder(h_dec)
        h_dec_out = self.transformer_decoder(h_dec)  # Standard bidirectional attention (no causal mask!)
        x_recon = self.output_proj_decoder(h_dec_out)   # [B, T, input_dim]
        
        return x_recon, total_vq_loss, stacked_indices
    

'''
note! recon loss must be computed in the outer training loop, after dataloader!
'''