import torch
import torch.nn.functional as F

from losses.sup_con_loss import SupConLoss
from utils import (
    cluster_center_alignment,
    compute_cluster_centers,
    compute_self_entropy,
    slide_mts_general,
)


class SupInfoLoss(torch.nn.modules.loss._Loss):
    """Stage 1 multi-term objective used to learn window representations."""

    def __init__(self, args):
        super().__init__()
        self.alpha_arr = args.alpha_arr
        self.args = args
        self.supcon = SupConLoss()

    def forward(self, batch, labels, model):
        slide_num = self.args.slide_num
        loss_weight = self.args.loss_weight

        encoded_batches = []
        dim_label_batches = []
        class_label_batches = []
        rind_batches = []
        logits_batches = []

        center_list = []
        self.rep_candidates = []
        classcon_losses = []
        rindcon_losses = []

        device = batch.device

        for view_index in range(slide_num):
            alpha = self.alpha_arr[view_index]
            (
                flat_windows,
                dim_labels,
                class_labels,
                rind,
            ) = slide_mts_general(
                batch,
                labels,
                alpha,
                return_labels=True,
                return_rind=True,
            )

            logits, encoded = model(flat_windows.unsqueeze(1).to(device))

            logits_batches.append(logits)
            encoded_batches.append(encoded)
            dim_label_batches.append(dim_labels)
            class_label_batches.append(class_labels)
            rind_batches.append(rind)
            self.rep_candidates.append(flat_windows)

            class_labels_device = class_labels.to(device)
            rind_device = rind.to(device)
            center_list.append(compute_cluster_centers(class_labels_device, encoded))

            classcon_losses.append(self.supcon(encoded, class_labels_device))
            rindcon_losses.append(self.supcon(encoded, rind_device))

        self.encoded_x = torch.concat(encoded_batches)
        self.dim_labels = torch.concat(dim_label_batches)
        self.class_labels = torch.concat(class_label_batches)
        self.rinds = torch.concat(rind_batches)
        self.logits = torch.concat(logits_batches)

        class_labels_device = self.class_labels.to(device)
        rinds_device = self.rinds.to(device)

        loss_class_all = self.supcon(self.encoded_x, class_labels_device)
        loss_class_sep = sum(classcon_losses) / slide_num
        loss_rind_all = self.supcon(self.encoded_x, rinds_device)
        loss_rind_sep = sum(rindcon_losses) / slide_num
        loss_ce = F.cross_entropy(self.logits, class_labels_device)
        loss_self_entropy = compute_self_entropy(self.logits)
        loss_cca = cluster_center_alignment(center_list)

        total_loss = (
            loss_weight[0] * loss_class_all
            + loss_weight[1] * loss_class_sep
            + loss_weight[2] * loss_ce
            + loss_weight[3] * loss_self_entropy
            + loss_weight[4] * loss_cca
            + loss_weight[5] * loss_rind_all
            + loss_weight[6] * loss_rind_sep
        )

        with torch.no_grad():
            _, predicted = torch.max(self.logits.data, 1)
            total = self.class_labels.size(0)
            correct = (predicted.cpu() == self.class_labels).sum().item()
            acc = correct / total if total > 0 else 0.0

        return (
            total_loss,
            loss_class_all,
            loss_class_sep,
            loss_ce,
            loss_self_entropy,
            loss_cca,
            loss_rind_all,
            loss_rind_sep,
            acc,
        )
