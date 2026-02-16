"""
Novel fairness-aware loss function for knowledge distillation.
Core innovation: penalizes accuracy gaps between demographic groups.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import NUM_GROUPS, DISTILL_TEMPERATURE


class FairnessLoss(nn.Module):
    """
    Fairness loss: average pairwise accuracy gap across demographic groups.

    Computes the mean absolute difference in accuracy between every pair
    of demographic groups within a batch. Driving this loss to zero means
    all groups achieve equal accuracy (Accuracy Parity).

    For N=8 groups, there are C(8,2)=28 pairs.
    """

    def __init__(self, num_groups=NUM_GROUPS, smooth=True, epsilon=1e-7):
        """
        Args:
            num_groups: Number of demographic groups
            smooth: Use soft accuracy (sigmoid-based) for differentiability
            epsilon: Small value to avoid division by zero
        """
        super().__init__()
        self.num_groups = num_groups
        self.smooth = smooth
        self.epsilon = epsilon

    def forward(self, logits, labels, group_ids):
        """
        Compute the fairness loss.

        Args:
            logits: Model output logits, shape (B,)
            labels: Ground truth labels (0=real, 1=fake), shape (B,)
            group_ids: Demographic group IDs, shape (B,)

        Returns:
            Scalar fairness loss value
        """
        group_accs = []
        group_masks = []

        for g in range(self.num_groups):
            mask = (group_ids == g)
            if mask.sum() < 2:  # Need at least 2 samples per group
                continue

            group_logits = logits[mask]
            group_labels = labels[mask].float()

            if self.smooth:
                # Soft accuracy: differentiable approximation
                # p(correct) = label * sigmoid(logit) + (1-label) * sigmoid(-logit)
                probs = torch.sigmoid(group_logits)
                correct_probs = group_labels * probs + (1 - group_labels) * (1 - probs)
                acc = correct_probs.mean()
            else:
                # Hard accuracy (not differentiable but exact)
                preds = (torch.sigmoid(group_logits) > 0.5).float()
                acc = (preds == group_labels).float().mean()

            group_accs.append(acc)

        if len(group_accs) < 2:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        # Compute average pairwise accuracy gap
        loss = torch.tensor(0.0, device=logits.device)
        count = 0
        for i in range(len(group_accs)):
            for j in range(i + 1, len(group_accs)):
                loss = loss + torch.abs(group_accs[i] - group_accs[j])
                count += 1

        return loss / max(count, 1)


class DistillationLoss(nn.Module):
    """
    Knowledge distillation loss using KL divergence between
    teacher and student soft logits with temperature scaling.
    """

    def __init__(self, temperature=DISTILL_TEMPERATURE):
        super().__init__()
        self.temperature = temperature

    def forward(self, student_logits, teacher_logits):
        """
        Compute KL-divergence distillation loss.

        For binary classification, we convert logits to 2-class probabilities
        then compute KL-divergence.

        Args:
            student_logits: Student model output, shape (B,)
            teacher_logits: Teacher model output (detached), shape (B,)

        Returns:
            Scalar distillation loss
        """
        T = self.temperature

        # Convert binary logits to 2-class logits: [logit_real, logit_fake]
        student_2class = torch.stack([-student_logits, student_logits], dim=1)
        teacher_2class = torch.stack([-teacher_logits, teacher_logits], dim=1)

        # Soft probabilities with temperature
        student_soft = F.log_softmax(student_2class / T, dim=1)
        teacher_soft = F.softmax(teacher_2class / T, dim=1)

        # KL divergence (scaled by T^2 as per Hinton et al.)
        loss = F.kl_div(student_soft, teacher_soft, reduction="batchmean") * (T ** 2)

        return loss


class CombinedFairDistillLoss(nn.Module):
    """
    Combined loss for fairness-aware knowledge distillation.

    L_total = α * L_distill + β * L_fairness + γ * L_cls

    Where:
        - L_distill: KL-divergence between student and teacher soft logits
        - L_fairness: Average pairwise accuracy gap across demographic groups
        - L_cls: Binary cross-entropy with ground truth labels
    """

    def __init__(self, alpha=0.7, beta=0.2, gamma=0.1,
                 temperature=DISTILL_TEMPERATURE, num_groups=NUM_GROUPS,
                 curriculum=False, max_beta=None, warmup_epochs=30):
        """
        Args:
            alpha, beta, gamma: Loss weights for distillation, fairness, classification
            temperature: Distillation temperature
            num_groups: Number of demographic groups
            curriculum: Whether to gradually increase beta (fairness weight)
            max_beta: Maximum beta value (if curriculum=True, beta starts at beta/4)
            warmup_epochs: Epochs over which to ramp up beta
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.curriculum = curriculum
        self.max_beta = max_beta or beta
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0

        self.distill_loss = DistillationLoss(temperature)
        self.fairness_loss = FairnessLoss(num_groups)
        self.cls_loss = nn.BCEWithLogitsLoss()

    def set_epoch(self, epoch):
        """Update current epoch for curriculum scheduling."""
        self.current_epoch = epoch

    def get_current_beta(self):
        """Get the current beta value (with optional curriculum)."""
        if not self.curriculum:
            return self.beta

        # Linear warmup from beta/4 to max_beta
        start_beta = self.max_beta / 4
        if self.current_epoch >= self.warmup_epochs:
            return self.max_beta
        progress = self.current_epoch / self.warmup_epochs
        return start_beta + (self.max_beta - start_beta) * progress

    def forward(self, student_logits, teacher_logits, labels, group_ids):
        """
        Compute combined loss.

        Args:
            student_logits: Student model output, shape (B,)
            teacher_logits: Teacher model output (detached), shape (B,)
            labels: Ground truth labels, shape (B,)
            group_ids: Demographic group IDs, shape (B,)

        Returns:
            total_loss: Combined loss scalar
            loss_dict: Dict with individual loss values for logging
        """
        l_distill = self.distill_loss(student_logits, teacher_logits)
        l_fairness = self.fairness_loss(student_logits, labels, group_ids)
        l_cls = self.cls_loss(student_logits, labels.float())

        current_beta = self.get_current_beta()
        total_loss = (self.alpha * l_distill +
                      current_beta * l_fairness +
                      self.gamma * l_cls)

        loss_dict = {
            "total": total_loss.item(),
            "distill": l_distill.item(),
            "fairness": l_fairness.item(),
            "cls": l_cls.item(),
            "beta": current_beta,
        }

        return total_loss, loss_dict
