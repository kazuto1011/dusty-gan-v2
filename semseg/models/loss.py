import torch
import torch.nn.functional as F


class FocalLoss(torch.nn.Module):
    def __init__(self, gamma, alpha=None, reduction="none"):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alpha", alpha)
        self.reduction = reduction
        assert self.reduction in ("none", "mean")

    def forward(self, logit, label):
        xentropy = F.cross_entropy(logit, label, weight=self.alpha, reduction="none")
        label = label.long().unsqueeze(1)
        probs = F.softmax(logit, dim=1).gather(dim=1, index=label).squeeze(1)
        focal_loss = (1 - probs) ** self.gamma * xentropy
        if self.reduction == "none":
            return focal_loss
        elif self.reduction == "mean":
            return focal_loss.mean()
