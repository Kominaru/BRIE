from torch import mean
from torch.nn.functional import logsigmoid


def bpr_loss(pos_scores, neg_scores):
    """Bayesian Personalised Ranking (BPR) pairwise loss function.
        Note that the sizes of pos_scores and neg_scores should be equal.
        Args:
            pos_scores (tensor): Tensor containing predictions for known positive items.
            neg_scores (tensor): Tensor containing predictions for sampled negative items.
        Returns:
            loss.
        """
    maxi = logsigmoid(pos_scores - neg_scores)
    loss = -mean(maxi)
    return loss
