from __future__ import annotations

import logging
import math
import time
from contextlib import nullcontext
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from koemi.configuration.settings import TrainingSettings
from koemi.model.execution import ExecutionMode
from koemi.model.network import KoemiModel, KoemiOutput
from koemi.training.dataset import IGNORE_TARGET_ID
from koemi.training.objective import TrainingObjective, calculate_training_objective


@dataclass(frozen=True)
class TrainingResult:
    mean_loss: float
    mean_task_loss: float
    mean_thinking_loss: float
    mean_surprise: float
    supervised_token_count: int
    token_count: int
    expert_activation_counts: tuple[int, ...]
    elapsed_seconds: float
    validation_loss: float | None
    validation_perplexity: float | None
    optimizer_steps: int
    tokens_per_second: float
    final_learning_rate: float
    precision: str
    mean_answer_loss: float | None = None
    mean_answer_bpb: float | None = None
    validation_answer_loss: float | None = None
    validation_answer_bpb: float | None = None


class Trainer:
    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger

    def train(
        self,
        model: KoemiModel,
        loader: DataLoader[dict[str, Tensor]],
        settings: TrainingSettings,
        validation_loader: DataLoader[dict[str, Tensor]] | None = None,
    ) -> TrainingResult:
        execution_mode = ExecutionMode(settings.execution_mode)
        device = self.resolve_device(settings.device)
        precision, autocast_dtype = self.resolve_precision(device, settings.precision)
        model.to(device)
        model.train()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
        )
        planned_steps = max(1, math.ceil(len(loader) / settings.gradient_accumulation_steps) * settings.epochs)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: self.learning_rate_factor(step, settings.warmup_steps, planned_steps)
        )
        scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda" and precision == "fp16")
        accumulator = MetricAccumulator()
        optimizer_steps = 0
        final_validation: MetricAccumulator | None = None
        start_time = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        for epoch_index in range(1, settings.epochs + 1):
            epoch_metrics = MetricAccumulator()
            accumulated_batches = 0
            for batch_index, batch in enumerate(loader, start=1):
                supervised_token_count, thinking_token_count = self.count_supervision(batch)
                if supervised_token_count == 0:
                    continue
                input_ids, target_ids, thinking_mask = self.move_batch(batch, device, settings.pin_memory)
                with self.autocast_context(device, autocast_dtype):
                    output = model(input_ids, execution_mode=execution_mode)
                    objective = calculate_training_objective(
                        output,
                        target_ids,
                        thinking_mask,
                        settings.thinking_loss_weight,
                        settings.label_smoothing,
                        supervised_token_count=supervised_token_count,
                        thinking_token_count=thinking_token_count,
                    )
                    scaled_loss = objective.total_loss / settings.gradient_accumulation_steps
                scaler.scale(scaled_loss).backward()
                accumulated_batches += 1
                if accumulated_batches == settings.gradient_accumulation_steps:
                    self.optimizer_step(model, optimizer, scheduler, scaler, settings, accumulated_batches)
                    optimizer_steps += 1
                    accumulated_batches = 0
                epoch_metrics.add(output, objective, supervised_token_count, thinking_token_count)
            if accumulated_batches > 0:
                self.optimizer_step(model, optimizer, scheduler, scaler, settings, accumulated_batches)
                optimizer_steps += 1
            if epoch_metrics.supervised_token_count == 0:
                raise ValueError("training loader produced no supervised tokens")
            validation = (
                self.evaluate(model, validation_loader, settings, device, execution_mode, autocast_dtype)
                if validation_loader is not None
                else None
            )
            final_validation = validation
            answer_loss = epoch_metrics.mean_answer_loss
            validation_answer_loss = validation.mean_answer_loss if validation else None
            self.logger.info(
                "epoch_completed epoch=%s loss=%.6f task_loss=%.6f thinking_loss=%.6f answer_loss=%s "
                "answer_bpb=%s surprise=%.4f validation_loss=%s validation_perplexity=%s "
                "validation_answer_loss=%s validation_answer_bpb=%s learning_rate=%.8f optimizer_steps=%s "
                "supervised_tokens=%s tokens=%s expert_activations=%s precision=%s",
                epoch_index,
                epoch_metrics.mean_loss,
                epoch_metrics.mean_task_loss,
                epoch_metrics.mean_thinking_loss,
                self.format_metric(answer_loss),
                self.format_metric(self.to_bits_per_byte(answer_loss)),
                epoch_metrics.mean_surprise,
                f"{validation.mean_loss:.6f}" if validation else "none",
                f"{math.exp(min(validation.mean_loss, 80.0)):.6f}" if validation else "none",
                self.format_metric(validation_answer_loss),
                self.format_metric(self.to_bits_per_byte(validation_answer_loss)),
                optimizer.param_groups[0]["lr"],
                optimizer_steps,
                epoch_metrics.supervised_token_count,
                epoch_metrics.token_count,
                epoch_metrics.expert_activation_counts,
                precision,
            )
            accumulator.merge(epoch_metrics)
        elapsed_seconds = time.perf_counter() - start_time
        return accumulator.to_result(
            elapsed_seconds,
            final_validation,
            optimizer_steps,
            optimizer.param_groups[0]["lr"],
            precision,
        )

    def evaluate(
        self,
        model: KoemiModel,
        loader: DataLoader[dict[str, Tensor]],
        settings: TrainingSettings,
        device: torch.device,
        execution_mode: ExecutionMode,
        autocast_dtype: torch.dtype | None,
    ) -> MetricAccumulator:
        metrics = MetricAccumulator()
        model.eval()
        with torch.inference_mode():
            for batch in loader:
                supervised_token_count, thinking_token_count = self.count_supervision(batch)
                if supervised_token_count == 0:
                    continue
                input_ids, target_ids, thinking_mask = self.move_batch(batch, device, settings.pin_memory)
                with self.autocast_context(device, autocast_dtype):
                    output = model(input_ids, execution_mode=execution_mode)
                    objective = calculate_training_objective(
                        output,
                        target_ids,
                        thinking_mask,
                        settings.thinking_loss_weight,
                        settings.label_smoothing,
                        supervised_token_count=supervised_token_count,
                        thinking_token_count=thinking_token_count,
                    )
                metrics.add(output, objective, supervised_token_count, thinking_token_count)
        model.train()
        if metrics.supervised_token_count == 0:
            raise ValueError("validation loader produced no supervised tokens")
        return metrics

    @staticmethod
    def resolve_device(device_name: str) -> torch.device:
        device = torch.device(device_name)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA training was requested but CUDA is unavailable")
        return device

    @staticmethod
    def resolve_precision(device: torch.device, requested: str) -> tuple[str, torch.dtype | None]:
        precision = requested
        if requested == "auto":
            if device.type != "cuda":
                precision = "fp32"
            else:
                precision = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
        if precision == "fp16" and device.type != "cuda":
            raise ValueError("fp16 training requires CUDA")
        if precision == "bf16" and device.type not in {"cpu", "cuda"}:
            raise ValueError("bf16 training requires a CPU or CUDA device")
        return precision, {"bf16": torch.bfloat16, "fp16": torch.float16}.get(precision)

    @staticmethod
    def autocast_context(device: torch.device, dtype: torch.dtype | None):
        return nullcontext() if dtype is None else torch.autocast(device_type=device.type, dtype=dtype)

    @staticmethod
    def move_batch(
        batch: dict[str, Tensor], device: torch.device, pin_memory: bool
    ) -> tuple[Tensor, Tensor, Tensor]:
        non_blocking = pin_memory and device.type == "cuda"
        return (
            batch["input_ids"].to(device, non_blocking=non_blocking),
            batch["target_ids"].to(device, non_blocking=non_blocking),
            batch["thinking_mask"].to(device, non_blocking=non_blocking),
        )

    @staticmethod
    def count_supervision(batch: dict[str, Tensor]) -> tuple[int, int]:
        target_ids = batch["target_ids"]
        thinking_mask = batch["thinking_mask"]
        if target_ids.shape != thinking_mask.shape:
            raise ValueError("thinking_mask must have the same shape as target_ids")
        supervised_positions = target_ids != IGNORE_TARGET_ID
        thinking_positions = supervised_positions & thinking_mask
        return int(supervised_positions.sum().item()), int(thinking_positions.sum().item())

    @staticmethod
    def to_bits_per_byte(loss: float | None) -> float | None:
        return loss / math.log(2.0) if loss is not None else None

    @staticmethod
    def format_metric(value: float | None) -> str:
        return f"{value:.6f}" if value is not None else "none"

    @staticmethod
    def learning_rate_factor(step: int, warmup_steps: int, total_steps: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(1, step + 1) / warmup_steps
        decay_steps = max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    @staticmethod
    def optimizer_step(
        model: KoemiModel,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        scaler: torch.amp.GradScaler,
        settings: TrainingSettings,
        accumulated_batches: int,
    ) -> None:
        scaler.unscale_(optimizer)
        if accumulated_batches < settings.gradient_accumulation_steps:
            correction = settings.gradient_accumulation_steps / accumulated_batches
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(correction)
        nn.utils.clip_grad_norm_(model.parameters(), settings.gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)


class MetricAccumulator:
    def __init__(self) -> None:
        self.weighted_loss: Tensor | None = None
        self.weighted_task_loss: Tensor | None = None
        self.weighted_thinking_loss: Tensor | None = None
        self.weighted_answer_loss: Tensor | None = None
        self.surprise_total: Tensor | None = None
        self.effective_token_weight = 0.0
        self.supervised_token_count = 0
        self.thinking_token_count = 0
        self.answer_token_count = 0
        self.token_count = 0
        self.expert_activation_totals: Tensor | None = None

    def add(
        self,
        output: KoemiOutput,
        objective: TrainingObjective,
        supervised_token_count: int,
        thinking_token_count: int,
    ) -> None:
        if objective.answer_loss is None:
            raise ValueError("training objective must provide answer_loss")
        if thinking_token_count < 0 or thinking_token_count > supervised_token_count:
            raise ValueError("thinking token count must be between zero and the supervised token count")
        answer_token_count = supervised_token_count - thinking_token_count
        self.weighted_loss = self.add_weighted(
            self.weighted_loss, objective.total_loss, objective.effective_token_weight
        )
        self.weighted_task_loss = self.add_weighted(
            self.weighted_task_loss, objective.task_loss, supervised_token_count
        )
        self.weighted_thinking_loss = self.add_weighted(
            self.weighted_thinking_loss, objective.thinking_loss, thinking_token_count
        )
        self.weighted_answer_loss = self.add_weighted(
            self.weighted_answer_loss, objective.answer_loss, answer_token_count
        )
        surprise_total = output.surprise_values.masked_select(output.valid_positions).sum()
        self.surprise_total = self.add_tensor(self.surprise_total, surprise_total)
        self.effective_token_weight += objective.effective_token_weight
        self.supervised_token_count += supervised_token_count
        self.thinking_token_count += thinking_token_count
        self.answer_token_count += answer_token_count
        self.token_count += output.token_count
        self.accumulate_expert_activations(output)

    @staticmethod
    def add_tensor(total: Tensor | None, value: Tensor) -> Tensor:
        value = value.detach()
        return value if total is None else total + value

    @classmethod
    def add_weighted(cls, total: Tensor | None, value: Tensor, weight: float) -> Tensor:
        return cls.add_tensor(total, value * weight)

    def accumulate_expert_activations(self, output: KoemiOutput) -> None:
        if output.expert_count == 0:
            return
        assignments = output.expert_indices.masked_select(output.valid_positions)
        activation_counts = torch.bincount(assignments, minlength=output.expert_count)
        self.expert_activation_totals = self.add_tensor(self.expert_activation_totals, activation_counts)

    def merge(self, other: MetricAccumulator) -> None:
        self.weighted_loss = self.merge_tensor(self.weighted_loss, other.weighted_loss)
        self.weighted_task_loss = self.merge_tensor(self.weighted_task_loss, other.weighted_task_loss)
        self.weighted_thinking_loss = self.merge_tensor(self.weighted_thinking_loss, other.weighted_thinking_loss)
        self.weighted_answer_loss = self.merge_tensor(self.weighted_answer_loss, other.weighted_answer_loss)
        self.surprise_total = self.merge_tensor(self.surprise_total, other.surprise_total)
        self.expert_activation_totals = self.merge_tensor(
            self.expert_activation_totals, other.expert_activation_totals
        )
        self.effective_token_weight += other.effective_token_weight
        self.supervised_token_count += other.supervised_token_count
        self.thinking_token_count += other.thinking_token_count
        self.answer_token_count += other.answer_token_count
        self.token_count += other.token_count

    @classmethod
    def merge_tensor(cls, total: Tensor | None, value: Tensor | None) -> Tensor | None:
        return total if value is None else cls.add_tensor(total, value)

    @staticmethod
    def average(weighted_value: Tensor | None, token_count: float) -> float:
        if weighted_value is None or token_count == 0:
            return 0.0
        return float((weighted_value / token_count).item())

    @property
    def mean_loss(self) -> float:
        return self.average(self.weighted_loss, self.effective_token_weight)

    @property
    def mean_task_loss(self) -> float:
        return self.average(self.weighted_task_loss, self.supervised_token_count)

    @property
    def mean_thinking_loss(self) -> float:
        return self.average(self.weighted_thinking_loss, self.thinking_token_count)

    @property
    def mean_answer_loss(self) -> float | None:
        if self.answer_token_count == 0:
            return None
        return self.average(self.weighted_answer_loss, self.answer_token_count)

    @property
    def mean_surprise(self) -> float:
        return self.average(self.surprise_total, self.token_count)

    @property
    def expert_activation_counts(self) -> tuple[int, ...]:
        if self.expert_activation_totals is None:
            return ()
        return tuple(int(count) for count in self.expert_activation_totals.tolist())

    def to_result(
        self,
        elapsed_seconds: float,
        validation: MetricAccumulator | None,
        optimizer_steps: int,
        final_learning_rate: float,
        precision: str,
    ) -> TrainingResult:
        validation_loss = validation.mean_loss if validation else None
        mean_answer_loss = self.mean_answer_loss
        validation_answer_loss = validation.mean_answer_loss if validation else None
        return TrainingResult(
            self.mean_loss,
            self.mean_task_loss,
            self.mean_thinking_loss,
            self.mean_surprise,
            self.supervised_token_count,
            self.token_count,
            tuple(self.expert_activation_counts),
            elapsed_seconds,
            validation_loss,
            math.exp(min(validation_loss, 80.0)) if validation_loss is not None else None,
            optimizer_steps,
            self.supervised_token_count / elapsed_seconds,
            final_learning_rate,
            precision,
            mean_answer_loss,
            Trainer.to_bits_per_byte(mean_answer_loss),
            validation_answer_loss,
            Trainer.to_bits_per_byte(validation_answer_loss),
        )
