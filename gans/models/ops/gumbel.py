from torch import nn
from torch.distributions import RelaxedBernoulli


class GumbelSigmoid(nn.Module):
    """
    Binary-case of Gumbel-Softmax:
    https://arxiv.org/pdf/1611.00712.pdf

    used in "Learning to Drop Points for LiDAR Scan Synthesis":
    https://arxiv.org/abs/2102.11952
    """

    def __init__(
        self,
        temperature: float = 1.0,
        straight_through: bool = True,
    ):
        super().__init__()
        self.temperature = temperature
        self.straight_through = straight_through

    def forward(self, logits):
        outcome_soft = RelaxedBernoulli(self.temperature, logits=logits).rsample()
        if self.straight_through:
            outcome_hard = (outcome_soft > 0.5).to(logits)
            return (outcome_hard - outcome_soft).detach() + outcome_soft
        else:
            return outcome_soft

    def extra_repr(self):
        return f"tau={self.temperature}, straight_through={self.straight_through}"
