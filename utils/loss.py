import torch
import numpy as np
import torch.nn.functional as F
    
def build_loss(args, device):
    loss = torch.nn.CTCLoss()
    return loss
    