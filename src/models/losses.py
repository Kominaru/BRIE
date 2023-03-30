from typing import Any, Optional
from torchmetrics.retrieval.base import RetrievalMetric
from torchmetrics.functional import auroc
from torch import Tensor, tensor
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


class UserwiseAUCROC(RetrievalMetric):

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    def __init__(
        self,
        empty_target_action: str = "neg",
        ignore_index: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            empty_target_action=empty_target_action,
            ignore_index=ignore_index,
            **kwargs,
        )

    def _metric(self, preds: Tensor, target: Tensor) -> Tensor:
        return auroc(preds, target, task='binary')
