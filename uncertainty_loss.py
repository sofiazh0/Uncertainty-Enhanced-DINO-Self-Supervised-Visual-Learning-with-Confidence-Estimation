import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


class UncertaintyAwareDINOLoss(nn.Module):
    def __init__(self, out_dim, teacher_temp, student_temp, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_output, teacher_output, epoch):
        student_means, student_vars = student_output
        teacher_mean, teacher_var = teacher_output

        # Ensure teacher tensors are detached
        teacher_mean = teacher_mean.detach()
        teacher_var = teacher_var.detach()

        # Get dimensions and match sizes
        d = student_means.shape[-1]
        teacher_batch = teacher_mean.shape[0]
        student_means = student_means[:teacher_batch]
        student_vars = student_vars[:teacher_batch]

        # Temperature scaling with lower temperature
        temp_scale = max(0.05, 0.5 - epoch * 0.02)  # Lower temperature
        student_means = student_means / (self.student_temp * temp_scale)
        teacher_mean = teacher_mean / (self.teacher_temp * temp_scale)

        # Center and normalize
        if self.center is None:
            self.center = torch.zeros(1, d, device=teacher_mean.device)
        teacher_mean = teacher_mean - self.center

        student_means = F.normalize(student_means, dim=-1, p=2)
        teacher_mean = F.normalize(teacher_mean, dim=-1, p=2)

        # Compute logits with higher sharpness
        sharpness = min(64.0, 16.0 + epoch * 4.0)  # Higher sharpness
        student_logits = student_means * sharpness
        teacher_logits = teacher_mean * sharpness

        # Compute probabilities
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_probs = F.softmax(teacher_logits, dim=-1)

        # Main loss
        main_loss = -torch.sum(teacher_probs *
                               student_log_probs, dim=-1).mean()

        # Uncertainty loss with stronger scaling
        student_vars = F.softplus(student_vars) + 0.5  # Larger offset
        teacher_var = F.softplus(teacher_var) + 0.5

        uncertainty_loss = F.mse_loss(
            torch.log(student_vars),
            torch.log(teacher_var)
        ) * 10.0  # Scale up uncertainty loss

        # Total loss with stronger uncertainty component
        uncertainty_weight = min(5.0, 0.5 * (epoch + 1))  # Higher weight
        total_loss = main_loss + uncertainty_weight * uncertainty_loss

        # Print loss details every epoch
        if not hasattr(self, f'loss_printed_epoch_{epoch}'):
            print(f"\nEpoch {epoch} loss details:")
            print(f"Temperature scale: {temp_scale:.4f}")
            print(f"Sharpness: {sharpness:.4f}")
            print(f"Main loss: {main_loss:.4f}")
            print(f"Uncertainty loss: {uncertainty_loss:.4f}")
            print(f"Uncertainty weight: {uncertainty_weight:.4f}")
            print(f"Total loss: {total_loss:.4f}")
            print(
                f"Student probs min/max: [{student_log_probs.exp().min():.4f}, {student_log_probs.exp().max():.4f}]")
            print(
                f"Teacher probs min/max: [{teacher_probs.min():.4f}, {teacher_probs.max():.4f}]\n")
            setattr(self, f'loss_printed_epoch_{epoch}', True)

        # Update center
        self.update_center(teacher_mean.mean(0))

        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.mean(teacher_output, dim=0, keepdim=True)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(batch_center)
            batch_center = batch_center / dist.get_world_size()
        self.center = self.center * self.center_momentum + \
            batch_center * (1 - self.center_momentum)
