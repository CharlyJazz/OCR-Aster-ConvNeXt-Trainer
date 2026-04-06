"""
Main training loop.

Entry point: train(config_path)
Called by ocr_aster/train/run.py CLI.

Flow:
  1. Load & validate YAML config
  2. Build model (ConvNeXt → BiLSTM → ASTER v2)
  3. Start HFPublisher (background data thread)
  4. Optionally resume from checkpoint
  5. Training loop:
       - forward_pass with AMP + scheduled teacher forcing
       - gradient clip + AdamW step
       - every valInterval: run_validation + log to MLflow + save report
       - save best_accuracy and best_norm_ed models separately
       - save periodic checkpoints
  6. Save final model
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW

from ocr_aster.config.loader import load_config
from ocr_aster.config.schema import TrainingConfig
from ocr_aster.data.collate import AlignCollate
from ocr_aster.data.dataset import HFValDataset
from ocr_aster.data.publisher import HFPublisher
from ocr_aster.model.model import AsterConvNeXt
from ocr_aster.monitoring.tracker import ExperimentTracker
from ocr_aster.train.forward_pass import forward_pass
from ocr_aster.train.metrics import ValidationResult
from ocr_aster.train.utils import AttnLabelConverter, Averager
from ocr_aster.train.validation import run_validation
from ocr_aster.train.validation_logger import write_report

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_model(config: TrainingConfig, device: torch.device) -> AsterConvNeXt:
    model = AsterConvNeXt(num_classes=config.num_class)

    # Kaiming init
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    return model.to(device)


def _resume_from_checkpoint(
    path: str,
    model: AsterConvNeXt,
    optimizer: AdamW,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> int:
    """Load checkpoint and return the starting iteration."""
    logger.info(f"Resuming from {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scaler.load_state_dict(ckpt["scaler"])

    # Try to extract iteration from filename first (more reliable)
    match = re.search(r"iter_(\d+)\.pth", path)
    if match:
        return int(match.group(1))
    return ckpt.get("iteration", 0)


def _save_checkpoint(
    path: Path,
    model: AsterConvNeXt,
    optimizer: AdamW,
    scaler: torch.amp.GradScaler,
    iteration: int,
    best_accuracy: float,
    best_norm_ed: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "iteration": iteration,
            "best_accuracy": best_accuracy,
            "best_norm_ed": best_norm_ed,
        },
        path,
    )
    logger.info(f"Checkpoint saved → {path}")


def _count_parameters(model: nn.Module) -> int:
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {n:,}")
    return n


def _build_val_dataloader(config: TrainingConfig) -> torch.utils.data.DataLoader:
    collate = AlignCollate(imgH=config.imgH, imgW=config.imgW)
    ds = HFValDataset(
        src=config.val_dataset,
        imgH=config.imgH,
        imgW=config.imgW,
    )
    return torch.utils.data.DataLoader(
        ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
    )


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(config_path: str) -> None:
    """
    Full training run from a YAML config path.

    Args:
        config_path: path to a TrainingConfig YAML file
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── 1. Config ──────────────────────────────────────────────────────────
    config = load_config(config_path)
    char_set = config.build_character_set()
    converter = AttnLabelConverter(char_set)
    config.num_class = converter.num_class

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    log_dir = Path(config.checkpoints_dir) / config.experiment_name
    log_dir.mkdir(parents=True, exist_ok=True)
    val_log_path = log_dir / "validation_log.txt"

    # ── 2. Model ───────────────────────────────────────────────────────────
    model = _build_model(config, device)
    num_params = _count_parameters(model)

    # ── 3. Optimiser + scaler ──────────────────────────────────────────────
    optimizer = AdamW(
        model.parameters(),
        lr=config.lr,
        betas=(config.beta1, config.beta2),
        eps=config.eps,
        weight_decay=config.weight_decay,
    )
    scaler = torch.amp.GradScaler(device=device.type)

    # ── 4. Resume ──────────────────────────────────────────────────────────
    start_iter = 0
    best_accuracy = 0.0
    best_norm_ed = 0.0

    if config.saved_model and Path(config.saved_model).exists():
        start_iter = _resume_from_checkpoint(
            config.saved_model, model, optimizer, scaler, device
        )
        logger.info(f"Resuming from iteration {start_iter:,}")
    elif config.saved_model:
        logger.warning(f"Checkpoint not found: {config.saved_model} — training from scratch")

    # ── 5. MLflow ──────────────────────────────────────────────────────────
    tracker: ExperimentTracker | None = None
    if config.mlflow.enabled:
        tracker = ExperimentTracker.start(
            experiment_name=config.mlflow.experiment_name or config.experiment_name,
            run_name=config.mlflow.run_name,
            tracking_uri=config.mlflow.tracking_uri or None,
            params={
                "imgH": config.imgH, "imgW": config.imgW,
                "output_channel": config.output_channel,
                "hidden_size": config.hidden_size,
                "batch_max_length": config.batch_max_length,
                "batch_size": config.batch_size,
                "lr": config.lr, "weight_decay": config.weight_decay,
                "tf_start": config.tf_start, "tf_end": config.tf_end,
                "tf_decay_iters": config.tf_decay_iters,
                "num_iter": config.num_iter,
                "num_params": num_params,
            },
        )

    # ── 6. Data ────────────────────────────────────────────────────────────
    publisher = HFPublisher(
        config=config,
        batch_size=config.batch_size,
        augment=True,
    )
    val_loader = _build_val_dataloader(config)
    collate = AlignCollate(imgH=config.imgH, imgW=config.imgW)
    loss_avg = Averager()

    # ── 7. Training loop ───────────────────────────────────────────────────
    logger.info(
        f"Training {config.experiment_name} "
        f"[{start_iter:,} → {config.num_iter:,} iters]"
    )
    model.train()

    _prev_phase_name: str | None = None

    for iteration in range(start_iter, config.num_iter + 1):

        # Phase transition
        phase = config.active_phase(iteration)
        phase_name = phase.name if phase else None
        if phase_name != _prev_phase_name:
            if phase is not None:
                logger.info(
                    f"→ Phase transition: {phase.name} "
                    f"[{phase.from_iter:,}–{phase.to_iter:,}] "
                    f"lr={phase.lr} batch_size={phase.batch_size}"
                )
                # Update LR
                for pg in optimizer.param_groups:
                    pg["lr"] = phase.lr
                # Rebuild augmentation pipeline for this phase
                publisher.update_phase(iteration)
            _prev_phase_name = phase_name

        # Batch from publisher
        raw_batch = publisher.get_batch()
        images, label_strings = zip(*raw_batch)
        batch_collated = collate(list(zip(images, label_strings)))
        images_tensor, label_strings = batch_collated
        images_tensor = images_tensor.to(device)

        # Encode
        text_for_pred, text_for_loss, lengths = converter.encode(
            list(label_strings), config.batch_max_length
        )
        text_for_pred = text_for_pred.to(device)
        text_for_loss = text_for_loss.to(device)

        # Forward
        tf_ratio = config.teacher_forcing_ratio(iteration)
        loss, _ = forward_pass(
            model, images_tensor, text_for_pred, text_for_loss, lengths,
            converter, tf_ratio, device,
        )

        # Backward
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        loss_avg.add(loss.item())

        # ── Logging ────────────────────────────────────────────────────────
        if iteration % config.save_log_gradient_every == 0:
            logger.info(
                f"[{iteration:>7,}/{config.num_iter:,}] "
                f"loss={loss_avg.val():.4f}  τ={tf_ratio:.3f}  "
                f"lr={optimizer.param_groups[0]['lr']:.2e}"
            )
            if tracker:
                tracker.log_train_step(
                    iteration=iteration,
                    loss=loss_avg.val(),
                    tf_ratio=tf_ratio,
                )
            loss_avg.reset()

        # ── Validation ────────────────────────────────────────────────────
        if iteration % config.valInterval == 0 and iteration > 0:
            model.eval()
            result: ValidationResult = run_validation(
                model=model,
                dataloader=val_loader,
                converter=converter,
                batch_max_length=config.batch_max_length,
                iteration=iteration,
                device=device,
            )
            model.train()

            logger.info(result.summary())
            write_report(result, val_log_path)

            if tracker:
                tracker.log_validation(result)

            # Save best models
            if result.accuracy > best_accuracy:
                best_accuracy = result.accuracy
                _save_checkpoint(
                    log_dir / "best_accuracy_model.pth",
                    model, optimizer, scaler, iteration,
                    best_accuracy, best_norm_ed,
                )
                if tracker:
                    tracker.set_tag("best_accuracy_iter", str(iteration))

            if result.norm_edit_distance > best_norm_ed:
                best_norm_ed = result.norm_edit_distance
                _save_checkpoint(
                    log_dir / "best_norm_ed_model.pth",
                    model, optimizer, scaler, iteration,
                    best_accuracy, best_norm_ed,
                )

        # ── Periodic checkpoint ───────────────────────────────────────────
        if iteration % config.save_every_n_iterations == 0 and iteration > 0:
            _save_checkpoint(
                log_dir / f"checkpoint_iter_{iteration}.pth",
                model, optimizer, scaler, iteration,
                best_accuracy, best_norm_ed,
            )

    # ── 8. Final save ──────────────────────────────────────────────────────
    publisher.stop()
    _save_checkpoint(
        log_dir / "final_model.pth",
        model, optimizer, scaler, config.num_iter,
        best_accuracy, best_norm_ed,
    )
    logger.info("Training complete.")

    if tracker:
        tracker.finish()
