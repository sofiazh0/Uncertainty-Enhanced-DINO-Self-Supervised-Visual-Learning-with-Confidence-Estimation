import torch


class UncertaintyMetrics:
    def __init__(self):
        self.reset()

    def reset(self):
        self.total_uncertainty = 0
        self.epistemic_uncertainty = 0
        self.aleatoric_uncertainty = 0
        self.count = 0

    def update(self, teacher_mean, teacher_var, student_means, student_vars):
        # Calculate uncertainties
        total_unc = torch.mean(teacher_var + student_vars.mean(0))
        epist_unc = torch.mean((teacher_mean - student_means.mean(0))**2)
        aleat_unc = torch.mean(teacher_var)

        # Update running averages
        self.total_uncertainty = (
            self.total_uncertainty * self.count + total_unc) / (self.count + 1)
        self.epistemic_uncertainty = (
            self.epistemic_uncertainty * self.count + epist_unc) / (self.count + 1)
        self.aleatoric_uncertainty = (
            self.aleatoric_uncertainty * self.count + aleat_unc) / (self.count + 1)
        self.count += 1

    def get_metrics(self):
        return {
            "total_uncertainty": self.total_uncertainty.item(),
            "epistemic_uncertainty": self.epistemic_uncertainty.item(),
            "aleatoric_uncertainty": self.aleatoric_uncertainty.item()
        }
