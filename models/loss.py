import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd.function import Function


class ImprovedDiceLossWithMag(nn.Module):
    def __init__(self, num_classes, smooth=1e-8, eps=1e-7, mag_weight=0.1):
        """
        Args:
            num_classes: number of classes
            smooth: Dice Loss's smoothing factor
            eps: to avoid division by zero
            mag_weight: Magnification Loss weight
        """
        super(ImprovedDiceLossWithMag, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.eps = eps
        self.mag_weight = nn.Parameter(torch.tensor(mag_weight))  # 可学习的权重系数
    
    def forward(self, predictions, targets, seq_len):
        '''
        Dice Loss with Magnification Loss
        parameters:
            predictions: model output logits, shape (B*L, C)
            targets: target labels, same shape as predictions, values between 0-1
            seq_len: actual sequence lengths, shape (B,)
        returns:
            total_loss: computed loss value
        '''
        batch_size = seq_len.shape[0]
        max_len = targets.shape[0] // batch_size
        
        mask = torch.arange(max_len, device=predictions.device)[None, :] < seq_len[:, None]
        mask = mask.reshape(-1, 1).expand(-1, self.num_classes)  # (bs*max_len, num_classes)
        valid_mask = mask
        
        sigmoid_preds = torch.sigmoid(predictions)
        mag_loss = self._compute_mag_loss(sigmoid_preds, valid_mask)

        dice_loss = 0
        for i in range(self.num_classes):
            pred_i = sigmoid_preds[:, i][valid_mask[:, i]]
            target_i = targets[:, i][valid_mask[:, i]].float()
            intersection = torch.sum(pred_i * target_i)
            pred_sum = torch.sum(pred_i)
            target_sum = torch.sum(target_i)
            
            dice_coef = (2.0 * intersection + self.smooth) / (pred_sum + target_sum + self.smooth + self.eps)
            dice_loss += (1.0 - dice_coef)

        dice_loss = dice_loss / self.num_classes
        total_loss = dice_loss + self.mag_weight * mag_loss
        
        return total_loss
    
    def _compute_mag_loss(self, sigmoid_preds, valid_mask):
        all_valid_probs = sigmoid_preds[valid_mask]
        
        base_mag_loss = torch.mean(all_valid_probs)
        
        return base_mag_loss        


class SimplifiedFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, smooth=1e-8):
        super().__init__()
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, pred_logits, targets):
        """
        focal loss
        parameters:
            pred_logits: model output logits, shape (B, C) or (B*L, C)
            targets: target labels, same shape as pred_logits, values between 0-1
        returns:
            focal_loss: computed loss value
        """
        pred_probs = torch.sigmoid(pred_logits)
        bce_loss = F.binary_cross_entropy(
            pred_probs, targets, reduction='none')
        
        p_t = targets * pred_probs + (1 - targets) * (1 - pred_probs)
        focal_weight = (1 - p_t) ** self.gamma
        
        loss_elements = bce_loss * focal_weight
        num_samples = loss_elements.shape[0]
        return loss_elements.sum() / max(num_samples, 1)


class HybridMacroMicroLoss(nn.Module):
    def __init__(self, num_classes, macro_weight=0.5, gamma=2.0, smooth=1e-8):
        super().__init__()
        self.num_classes = num_classes
        print("nnnnn", num_classes)
        self.macro_weight = nn.Parameter(torch.tensor(macro_weight))
        self.gamma = nn.Parameter(torch.tensor(gamma))
        self.smooth = smooth

        self.macro_loss = ImprovedDiceLossWithMag(num_classes, smooth)
        self.micro_loss = SimplifiedFocalLoss(gamma, smooth)

    def forward(self, pred, target, seq_len, class_weights=None):
        macro = self.macro_loss(pred, target, seq_len)
        micro = self.micro_loss(pred, target)
        return self.macro_weight * macro + (1 - self.macro_weight) * micro


class FinetuneBinaryLoss(nn.Module):
    def __init__(self, smooth=1e-8, gamma=2, alpha=1, beta=0.6):
        super(FinetuneBinaryLoss, self).__init__()
        self.smooth = smooth
        self.gamma = gamma
        self.class_weights = None
    
    def forward(self, binary_logits, labels, seq_lens):
        batch_size = labels.shape[0]
        self.to(labels.device)
        if binary_logits.dim() > 1:
            binary_logits = binary_logits.view(-1)
        labels = labels.float().view(-1)
        
        pos = labels.sum()
        neg = labels.numel() - pos
        total = pos + neg + 2*self.smooth
        
        # calculate alpha for positive and negative classes
        alpha_p = (neg + self.smooth) / total
        alpha_n = (pos + self.smooth) / total
        
        p = torch.sigmoid(binary_logits)
        pt = p * labels + (1 - p) * (1 - labels)
        
        alpha = alpha_p * labels + alpha_n * (1 - labels)
        focal_weight = (1 - pt).pow(self.gamma)

        loss = -alpha * focal_weight * torch.log(pt.clamp_min(self.smooth))
        return loss.mean()