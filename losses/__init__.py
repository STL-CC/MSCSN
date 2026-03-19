"""Loss functions used by the paper-aligned Stage 1 model."""

from .sup_con_loss import SupConLoss
from .sup_info_loss import SupInfoLoss

__all__ = ["SupConLoss", "SupInfoLoss"]
