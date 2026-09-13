from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional

from koemi.model.network import KoemiOutput
from koemi.training.dataset import IGNORE_TARGET_ID


@dataclass(frozen=True)
class TrainingObjective:
    task_loss: Tensor
    thinking_loss: Tensor
    total_loss: Tensor
    answer_loss: Tensor | None = None
    effective_token_weight: float = 0.0


def token_cross_entropy(logits: Tensor, target_ids: Tensor, label_smoothing: float = 0.0) -> Tensor:
    flat_loss = functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        target_ids.reshape(-1),
        ignore_index=IGNORE_TARGET_ID,
        reduction="none",
        label_smoothing=label_smoothing,
    )
    return flat_loss.view(target_ids.shape)


def calculate_training_objective(
    output: KoemiOutput,
    target_ids: Tensor,
    thinking_mask: Tensor,
    thinking_loss_weight: float,
    label_smoothing: float = 0.0,
    *,
    supervised_token_count: int | None = None,
    thinking_token_count: int | None = None,
) -> TrainingObjective:
    if thinking_loss_weight < 0.0:
        raise ValueError("thinking_loss_weight must be non-negative")
    if thinking_mask.shape != target_ids.shape:
        raise ValueError("thinking_mask must have the same shape as target_ids")
    if (supervised_token_count is None) != (thinking_token_count is None):
        raise ValueError("supervised and thinking token counts must be provided together")
    supervised_mask = target_ids != IGNORE_TARGET_ID
    thinking_positions = supervised_mask & thinking_mask
    if supervised_token_count is None:
        supervised_token_count = int(supervised_mask.sum().item())
        thinking_token_count = int(thinking_positions.sum().item())
    if supervised_token_count < 1:
        raise ValueError("training objective requires at least one supervised target token")
    if thinking_token_count < 0 or thinking_token_count > supervised_token_count:
        raise ValueError("thinking token count must be between zero and the supervised token count")
    answer_token_count = supervised_token_count - thinking_token_count
    token_loss = token_cross_entropy(output.logits, target_ids, label_smoothing)
    thinking_loss_sum = (token_loss * thinking_positions).sum()
    answer_positions = supervised_mask & ~thinking_mask
    answer_loss_sum = (token_loss * answer_positions).sum()
    task_loss = (thinking_loss_sum + answer_loss_sum) / supervised_token_count
    thinking_loss = thinking_loss_sum / max(1, thinking_token_count)
    answer_loss = answer_loss_sum / max(1, answer_token_count)
    effective_token_weight = answer_token_count + thinking_loss_weight * thinking_token_count
    if effective_token_weight <= 0.0:
        raise ValueError("thinking_loss_weight removes every supervised target token")
    total_loss = (answer_loss_sum + thinking_loss_weight * thinking_loss_sum) / effective_token_weight
    return TrainingObjective(task_loss, thinking_loss, total_loss, answer_loss, effective_token_weight)
