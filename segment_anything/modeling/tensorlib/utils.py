import torch
import torch.nn as nn

def mark_only_td_as_trainable(model: nn.Module):
    for n, p in model.named_parameters():
        p.requires_grad = False
        
        if 'td_' in n:
            p.requires_grad = True

        if 'lora_' in n:
            p.requires_grad = True
        
        if 'no_mask_embed' in n:
            p.requires_grad = True

        if 'mask_decoder' in n:
            p.requires_grad = True
            
        if 'norm' in n:
            p.requires_grad = True
            
            
def orthogonal_reg(sam, type, device):
    reg = []
    if type == 'tucker':
        for n, p in sam.named_parameters():
            if 'td_U1' in n or 'td_U2' in n:
                reg.append(torch.norm(torch.mm(torch.transpose(p, 0, 1), p) - torch.eye(p.shape[-1]).to(device), p='fro'))
            
            if 'td_G' in n:
                for idx in range(p.shape[-1]):
                    reg.append(torch.norm(torch.mm(torch.transpose(p[:, :, idx], 0, 1), p[:, :, idx]) - torch.eye(p[:, :, idx].shape[-1]).to(device), p='fro'))

    
    loss = torch.zeros(len(reg)).to(device)
    for idx in range(len(loss)):
        loss[idx] = reg[idx]
    return torch.mean(loss)