import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.utils.data
import matplotlib.pyplot as plt
from torch import Tensor

import os
import sys

from utils import weighted_MSELoss

class ENCODER_BASE(nn.Module):
    def __init__(self, nn_featrues:nn.Module, *args, **kwargs) -> None:
        super(ENCODER_BASE, self).__init__(*args, **kwargs)
        self.nn_features = nn_featrues
    
    def forward(self, x: torch.Tensor):
        return self.nn_features(x)


class DECODER_BASE(nn.Module):
    def __init__(self, nn_featrues:nn.Module, *args, **kwargs) -> None:
        super(DECODER_BASE, self).__init__(*args, **kwargs)
        self.nn_features = nn_featrues
    
    def forward(self, x: torch.Tensor):
        return self.nn_features(x)


class ROM_BASE(nn.Module):
    def __init__(self, ENCODER:nn.Module, DECODER:nn.Module, 
                       seq_length: int, *args, **kwargs) -> None:
        super(ROM_BASE, self).__init__(*args, **kwargs)
        self.ENCODER = ENCODER
        self.DECODER = DECODER
        self.hidden_dim = ENCODER.hidden_dims[-1]
        self.seq_length = seq_length
        self.C_forward = None
    
    def forward(self, state: torch.Tensor):
        z = self.ENCODER(state)
        z_next = torch.mm(z, self.C_forward)
        pred_s_next = self.DECODER(z_next)

        
        return pred_s_next

    def batch_latent_forward(self, batch_z: torch.Tensor):
        B = batch_z.shape[0]
        if self.C_forward.dim() == 2:
            C_forward = self.C_forward.unsqueeze(0).repeat(B, 1, 1)
        else:
            C_forward = self.C_forward
        batch_z_next = torch.bmm(batch_z, C_forward)
        return batch_z_next

    def latent_forward(self, z: torch.Tensor):
        z_next = torch.mm(z, self.C_forward)
        return z_next
    
    def compute_loss(self, state_seq, state_next_seq, 
                 weight_matrix=None,
                 mi_k: int = 2, mi_temperature: float = 0.1):
        B = state_seq.shape[0]
        device = state_seq.device
        z_seq = torch.zeros(B, self.seq_length, self.hidden_dim).to(device)
        z_next_seq = torch.zeros(B, self.seq_length, self.hidden_dim).to(device)

        loss_fwd = 0.0
        loss_identity = 0.0
        mi_loss = torch.tensor(0.0, device=device)
        entropy_loss = torch.tensor(0.0, device=device)

        # decode
        for i in range(self.seq_length):
            z_seq[:, i, :] = self.ENCODER(state_seq[:, i, :])
            z_next_seq[:, i, :] = self.ENCODER(state_next_seq[:, i, :])

        # Koopman
        z_seq_pinv = self.batch_pinv(z_seq, I_factor=1e-1)
        forward_weights = torch.bmm(z_seq_pinv, z_next_seq).mean(dim=0).repeat(B, 1, 1)
        self.C_forward = forward_weights.detach().clone()
        pred_z_next = self.batch_latent_forward(z_seq)

        # MSE
        for i in range(self.seq_length):
            recon_s = self.DECODER(z_seq[:, i, :])
            recon_s_next = self.DECODER(pred_z_next[:, i, :])

            if weight_matrix is not None:
                loss_fwd += weighted_MSELoss()(recon_s_next, state_next_seq[:, i, :], weight_matrix).sum()
                loss_identity += weighted_MSELoss()(recon_s, state_seq[:, i, :], weight_matrix).sum()
            else:
                loss_fwd += F.mse_loss(recon_s_next, state_next_seq[:, i, :])
                loss_identity += F.mse_loss(recon_s, state_seq[:, i, :])


        mi_loss = self.info_nce(z_seq, k=mi_k, temperature=mi_temperature, use_cosine=True)

        entropy_loss = self.von_neumann_entropy(z_seq, center=True)

        return {
            "loss_fwd": loss_fwd,
            "loss_identity": loss_identity,
            "loss_mi": mi_loss,
            "loss_entropy": entropy_loss,
            "C_forward": self.C_forward.mean(dim=0),
        }

    def info_nce(self, z_seq: torch.Tensor, k: int = 2, 
                                  temperature: float = 0.1, use_cosine: bool = True):
        
        B, T, H = z_seq.shape
        device = z_seq.device
        
        if T <= 2 * k:
            return torch.tensor(0.0, device=device)
        
        # Only compute [k, T-k-1] for time steps with complete neighborhoods
        valid_indices = torch.arange(k, T - k, device=device)
        n_valid = len(valid_indices)
        
        if n_valid == 0:
            return torch.tensor(0.0, device=device)
        
        # Normalize the entire sequence (if using cosine similarity)
        if use_cosine:
            z_normalized = F.normalize(z_seq, dim=-1)  # [B, T, H]
        else:
            z_normalized = z_seq
        
        total_loss = 0.0
        
        # Calculate InfoNCE for each valid anchor position
        for n in valid_indices:
            # anchor: z_n [B, H]
            anchor = z_normalized[:, n, :]
            
            # Positive sample: neighborhood P_n = {z_{n±i} | 1≤i≤k}
            pos_indices = (torch.cat([
                torch.arange(n-k, n, device=device),  # past k
                torch.arange(n+1, n+k+1, device=device)  # k future
            ]))
            
            # Negative samples: all non-neighborhood positions (excluding themselves and neighbors)
            all_indices = torch.arange(T, device=device)
            neighbor_and_self = torch.cat([
                torch.tensor([n], device=device),  # self
                pos_indices  # Neighborhood
            ])
            neg_mask = ~torch.isin(all_indices, neighbor_and_self)
            neg_indices = all_indices[neg_mask]
            
            if len(neg_indices) == 0:
                continue
            
            # Construct positive and negative samples
            positives = z_normalized[:, pos_indices, :]  # [B, 2k, H]  
            negatives = z_normalized[:, neg_indices, :]   # [B, n_neg, H]
            
            # Calculate similarity score
            pos_scores = torch.bmm(
                anchor.unsqueeze(1),  # [B, 1, H]
                positives.transpose(1, 2)  # [B, H, 2k]
            ).squeeze(1) / temperature  # [B, 2k]
            
            neg_scores = torch.bmm(
                anchor.unsqueeze(1),  # [B, 1, H]
                negatives.transpose(1, 2)  # [B, H, n_neg]
            ).squeeze(1) / temperature  # [B, n_neg]
            
            # Multiple positive variant InfoNCE loss
            # log(sum(exp(pos_scores)) / (sum(exp(pos_scores)) + sum(exp(neg_scores))))
            
            # Numerically stable calculations
            max_pos = pos_scores.max(dim=1, keepdim=True)[0]  # [B, 1]
            max_neg = neg_scores.max(dim=1, keepdim=True)[0]  # [B, 1]
            max_all = torch.max(max_pos, max_neg)  # [B, 1]
            
            pos_scores_stable = pos_scores - max_all
            neg_scores_stable = neg_scores - max_all
            
            pos_exp_sum = torch.exp(pos_scores_stable).sum(dim=1)  # [B]
            neg_exp_sum = torch.exp(neg_scores_stable).sum(dim=1)  # [B]
            
            # InfoNCE loss: -log(positive sample probability)
            loss = -torch.log(pos_exp_sum / (pos_exp_sum + neg_exp_sum + 1e-8))
            total_loss += loss.mean()
        
        return total_loss / n_valid

    def von_neumann_entropy(self, z_seq: torch.Tensor, center: bool = True, eps: float = 1e-8):
        B, T, H = z_seq.shape
        Z = z_seq.reshape(B * T, H)
        
        if center:
            Z = Z - Z.mean(dim=0, keepdim=True)
        
        # Use more stable covariance calculation
        N = Z.shape[0]
        Sigma = torch.mm(Z.t(), Z) / (N - 1)  # Unbiased estimation
        
        # Ensure positive definiteness
        Sigma = Sigma + eps * torch.eye(H, device=Z.device)
        trace = torch.trace(Sigma)
        rho = Sigma / trace.clamp_min(eps)
        
        # Using eigvalsh is more stable for real symmetric matrices
        evals = torch.linalg.eigvalsh(rho)
        evals = evals.clamp_min(eps)
        
        S = -(evals * torch.log(evals)).sum()
        return S

    @staticmethod
    def batch_pinv(z_seq: torch.Tensor, I_factor:float=1e-2):
        # inverse of z_seq
        # za_seq: [B, T, Dim_s]
        # I_factor: Identity factor
        B, T, D = z_seq.size()
        device = z_seq.device

        trans = T < D
        if trans:
            z_seq = torch.transpose(z_seq, 1, 2)
            T, D = D, T

        if not z_seq.is_cuda:
            z_seq = z_seq.to('cpu')
            I = torch.eye(D)[None, :, :].repeat(B, 1, 1).to('cpu')
        else:
            I = torch.eye(D)[None, :, :].repeat(B, 1, 1).to(device)
            
        z_seq_T = torch.transpose(z_seq, 1, 2)
        z_seq_pinv = torch.linalg.solve(
            torch.bmm(z_seq_T, z_seq) + I_factor * I,
            z_seq_T
        )
        if trans:
            z_seq_pinv = torch.transpose(z_seq_pinv, 1, 2)

        return z_seq_pinv.to(device)
    
    def save_C_forward(self, path, C_forward):
        C_forward_filename = path + '/' + 'C_forward.pt'
        print('[INFO] Saving C_forward weights to:', C_forward_filename)
        torch.save(C_forward, C_forward_filename)
        
    def save_model(self, path):
        # Save the model
        self.to('cpu')
        filename = path + '/forward_model.pt'
        print('[INFO] Saving forward_model weights to:', filename)
        torch.save(self.state_dict(), filename)
    
