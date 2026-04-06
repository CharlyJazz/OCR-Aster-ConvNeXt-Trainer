"""
Main training loop.

Entry point: train(config_path)
Called by ocr_aster/train/run.py CLI.

Flow:
  1. Load & validate YAML config
  2. Build model (ConvNeXt → BiLSTM → ASTER v2)
  3. Start HFPublisher (background data thread)
  4. Optionally resume from checkpoint
  5. Register SIGINT handler — Ctrl+C saves an interrupt checkpoint cleanly
  6. Training loop:
       - forward_pass with AMP + scheduled teacher forcing
       - gradient clip + AdamW step
       - every valInterval: run_validation + log to MLflow + save report
       - save best_accuracy and best_norm_ed models separately
       - save periodic checkpoints
  7. Save final model

Checkpoint format (all saves use the same schema via save_checkpoint()):
    {
        "model":          model.state_dict(),
        "optimizer":      optimizer.state_dict(),
        "scaler":         scaler.state_dict(),
        "iteration":      int,
        "best_accuracy":  float,
        "best_norm_ed":   float,
        "experiment_name": str,
        "num_class":      int,
        "character":      str,
        "checkpoint_type": "periodic" | "best_accuracy" | "best_norm_ed"
                         | "interrupt" | "final",
    }

Every .pth produced by this loop has identical keys and sizes — no surprises
when resuming or comparing checkpoints.
"""

from __future__ import annotations

import csv
import logging
import re
import signal
import sys
from pathlib import Path
from typing import Literal

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

CheckpointType = Literal["periodic", "best_accuracy", "best_norm_ed", "interrupt", "final"]


# ---------------------------------------------------------------------------
# Centralised checkpoint — ALL saves go through this one function
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: Path,
    model: AsterConvNeXt,
    optimizer: AdamW,
    scaler: torch.amp.GradScaler,
    iteration: int,
    best_accuracy: float,
    best_norm_ed: float,
    config: TrainingConfig,
    checkpoint_type: CheckpointType = "periodic",
) -> None:
    """
    Save a training checkpoint with unified metadata.

    Every .pth file written during a training run has the exact same set of
    keys, so file sizes are comparable and resuming never fails due to missing
    fields.

    Args:
        path:            destination .pth file path
        model:           AsterConvNeXt instance
        optimizer:       AdamW optimiser
        scaler:          AMP GradScaler
        iteration:       current training iteration
        best_accuracy:   best exact-match accuracy seen so far
        best_norm_ed:    best normalised edit distance seen so far
        config:          full TrainingConfig (provides experiment metadata)
        checkpoint_type: label stored in the file — useful for inspection
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            # ── Resumable state ──────────────────────────────────────────
            "model":          model.state_dict(),
            "optimizer":      optimizer.state_dict(),
            "scaler":         scaler.state_dict(),
            "iteration":      iteration,
            # ── Best metrics (so resume can restore the tracker values) ──
            "best_accuracy":  best_accuracy,
            "best_norm_ed":   best_norm_ed,
            # ── Experiment metadata (same for every ckpt in a run) ───────
            "experiment_name": config.experiment_name,
            "num_class":       config.num_class,
            "character":       config.character,
            "imgH":            config.imgH,
            "imgW":            config.imgW,
            "batch_max_length": config.batch_max_length,
            # ── Provenance ───────────────────────────────────────────────
            "checkpoint_type": checkpoint_type,
        },
        path,
    )
    logger.info(f"[{checkpoint_type}] checkpoint → {path.name}  (iter {iteration:,})")


# ---------------------------------------------------------------------------
# Other helpers
# ---------------------------------------------------------------------------

def _build_model(config: TrainingConfig, device: torch.device) -> AsterConvNeXt:
    model = AsterConvNeXt(num_classes=config.num_class)
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
) -> tuple[int, float, float]:
    """
    Load checkpoint state into model/optimizer/scaler.

    Returns:
        (start_iteration, best_accuracy, best_norm_ed)
    """
    logger.info(f"Resuming from {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scaler.load_state_dict(ckpt["scaler"])

    # Iteration: filename is canonical (safer than the stored value if file
    # was manually renamed), fall back to stored value.
    match = re.search(r"iter_(\d+)\.pth", path)
    iteration = int(match.group(1)) if match else ckpt.get("iteration", 0)

    best_accuracy = ckpt.get("best_accuracy", 0.0)
    best_norm_ed  = ckpt.get("best_norm_ed", 0.0)
    return iteration, best_accuracy, best_norm_ed


def _count_parameters(model: nn.Module) -> int:
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {n:,}")
    return n


def _build_val_dataloader(config: TrainingConfig) -> torch.utils.data.DataLoader:
    collate = AlignCollate(imgH=config.imgH, imgW=config.imgW)
    ds = HFValDataset(src=config.val_dataset, imgH=config.imgH, imgW=config.imgW)
    return torch.utils.data.DataLoader(
        ds, batch_size=config.batch_size, shuffle=False,
        num_workers=0, collate_fn=collate,
    )


def _grad_norm(model: nn.Module) -> float:
    """Compute total L2 gradient norm across all parameters."""
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return total ** 0.5


def _init_metrics_csv(path: Path) -> None:
    """Create metrics CSV with header if it doesn't exist yet."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "iteration", "train_loss", "grad_norm", "learning_rate",
            "tf_ratio",
            "val_accuracy", "val_norm_ed", "val_cer", "val_loss",
            "avg_conf_correct", "avg_conf_incorrect", "calibration_gap",
            "acc_1_5", "acc_6_10", "acc_11_20", "acc_21plus",
        ])


def _append_train_row(
    path: Path, iteration: int, loss: float, grad_norm: float,
    lr: float, tf_ratio: float,
) -> None:
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            iteration, f"{loss:.6f}", f"{grad_norm:.6f}", f"{lr:.8f}",
            f"{tf_ratio:.4f}",
            "", "", "", "", "", "", "", "", "", "", "",
        ])


def _append_val_row(
    path: Path, iteration: int, loss: float, grad_norm: float,
    lr: float, tf_ratio: float, result: "ValidationResult",
) -> None:
    abl = result.accuracy_by_length
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            iteration,
            f"{loss:.6f}", f"{grad_norm:.6f}", f"{lr:.8f}", f"{tf_ratio:.4f}",
            f"{result.accuracy:.6f}", f"{result.norm_edit_distance:.6f}",
            f"{result.cer:.6f}", f"{result.val_loss:.6f}",
            f"{result.avg_conf_correct:.6f}", f"{result.avg_conf_incorrect:.6f}",
            f"{result.calibration_gap:.6f}",
            f"{abl.get('1-5', 0.0):.4f}", f"{abl.get('6-10', 0.0):.4f}",
            f"{abl.get('11-20', 0.0):.4f}", f"{abl.get('21+', 0.0):.4f}",
        ])


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
    val_log_path    = log_dir / "validation_log.txt"
    metrics_csv_path = log_dir / f"training_metrics_{config.experiment_name}.csv"
    _init_metrics_csv(metrics_csv_path)

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
    start_iter    = 0
    best_accuracy = 0.0
    best_norm_ed  = 0.0

    if config.saved_model and Path(config.saved_model).exists():
        start_iter, best_accuracy, best_norm_ed = _resume_from_checkpoint(
            config.saved_model, model, optimizer, scaler, device
        )
        logger.info(
            f"Resumed at iter {start_iter:,}  "
            f"best_acc={best_accuracy:.4f}  best_ned={best_norm_ed:.4f}"
        )
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
    # publisher is re-assigned on batch_size phase changes — keep reference via list
    publisher = HFPublisher(config=config, batch_size=config.batch_size, augment=True)
    _pub = [publisher]  # mutable container so SIGINT handler always sees current ref
    val_loader = _build_val_dataloader(config)
    collate    = AlignCollate(imgH=config.imgH, imgW=config.imgW)
    loss_avg   = Averager()

    # ── 7. SIGINT handler (Ctrl+C → interrupt checkpoint) ──────────────────
    # Mutable container so the nested function can write to it
    _state = {"iteration": start_iter, "interrupted": False}

    def _handle_sigint(sig, frame):  # noqa: ANN001
        if _state["interrupted"]:
            logger.warning("Second interrupt — forcing exit without saving.")
            sys.exit(1)
        logger.warning("Interrupt received — saving checkpoint before exit...")
        _state["interrupted"] = True
        save_checkpoint(
            path=log_dir / f"interrupt_iter_{_state['iteration']}.pth",
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            iteration=_state["iteration"],
            best_accuracy=best_accuracy,
            best_norm_ed=best_norm_ed,
            config=config,
            checkpoint_type="interrupt",
        )
        _pub[0].stop()
        if tracker:
            tracker.finish()
        logger.info("Interrupt checkpoint saved. Exiting.")
        sys.exit(0)

    signal.signal(signal.SIGINT, _handle_sigint)

    # ── 8. Training loop ───────────────────────────────────────────────────
    logger.info(
        f"Training '{config.experiment_name}' "
        f"[{start_iter:,} → {config.num_iter:,} iters]"
    )
    model.train()
    _prev_phase_name: str | None = None

    for iteration in range(start_iter, config.num_iter + 1):
        _state["iteration"] = iteration

        # ── Phase transition ───────────────────────────────────────────────
        phase = config.active_phase(iteration)
        phase_name = phase.name if phase else None
        if phase_name != _prev_phase_name:
            if phase is not None:
                logger.info(
                    f"→ Phase: {phase.name}  "
                    f"iters [{phase.from_iter:,}–{phase.to_iter:,}]  "
                    f"lr={phase.lr}  batch_size={phase.batch_size}"
                )
                # Update learning rate
                for pg in optimizer.param_groups:
                    pg["lr"] = phase.lr
                # Rebuild augmentation pipeline for this phase
                publisher.update_phase(iteration)
                # Update batch size: restart publisher if it changed
                if phase.batch_size != publisher._batch_size:
                    logger.info(
                        f"Batch size change: {publisher._batch_size} → {phase.batch_size}"
                    )
                    publisher.stop()
                    publisher = HFPublisher(
                        config=config,
                        batch_size=phase.batch_size,
                        augment=True,
                    )
                    publisher.update_phase(iteration)
                    _pub[0] = publisher  # keep SIGINT handler in sync
            _prev_phase_name = phase_name

        # ── Forward + backward ────────────────────────────────────────────
        raw_batch = publisher.get_batch()
        images_pil, label_strings_raw = zip(*raw_batch)
        images_tensor, label_strings = collate(list(zip(images_pil, label_strings_raw)))
        images_tensor = images_tensor.to(device)

        text_for_pred, text_for_loss, lengths = converter.encode(
            list(label_strings), config.batch_max_length
        )
        text_for_pred = text_for_pred.to(device)
        text_for_loss = text_for_loss.to(device)

        tf_ratio = config.teacher_forcing_ratio(iteration)
        loss, _ = forward_pass(
            model, images_tensor, text_for_pred, text_for_loss,
            lengths, converter, tf_ratio, device,
            label_smoothing=config.label_smoothing,
        )

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = _grad_norm(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        loss_avg.add(loss.item())
        current_lr = optimizer.param_groups[0]["lr"]

        # ── Step logging ──────────────────────────────────────────────────
        if iteration % config.save_log_gradient_every == 0:
            logger.info(
                f"[{iteration:>7,}/{config.num_iter:,}] "
                f"loss={loss_avg.val():.4f}  grad={grad_norm:.3f}  "
                f"τ={tf_ratio:.3f}  lr={current_lr:.2e}"
            )
            if tracker:
                tracker.log_train_step(iteration=iteration, loss=loss_avg.val(), tf_ratio=tf_ratio)
            _append_train_row(
                metrics_csv_path, iteration, loss_avg.val(),
                grad_norm, current_lr, tf_ratio,
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
            _append_val_row(
                metrics_csv_path, iteration, result.val_loss,
                grad_norm, current_lr, tf_ratio, result,
            )
            if tracker:
                tracker.log_validation(result)

            if result.accuracy > best_accuracy:
                best_accuracy = result.accuracy
                save_checkpoint(
                    log_dir / "best_accuracy_model.pth",
                    model, optimizer, scaler, iteration,
                    best_accuracy, best_norm_ed, config,
                    checkpoint_type="best_accuracy",
                )
                if tracker:
                    tracker.set_tag("best_accuracy_iter", str(iteration))

            if result.norm_edit_distance > best_norm_ed:
                best_norm_ed = result.norm_edit_distance
                save_checkpoint(
                    log_dir / "best_norm_ed_model.pth",
                    model, optimizer, scaler, iteration,
                    best_accuracy, best_norm_ed, config,
                    checkpoint_type="best_norm_ed",
                )

        # ── Periodic checkpoint ───────────────────────────────────────────
        if iteration % config.save_every_n_iterations == 0 and iteration > 0:
            save_checkpoint(
                log_dir / f"checkpoint_iter_{iteration}.pth",
                model, optimizer, scaler, iteration,
                best_accuracy, best_norm_ed, config,
                checkpoint_type="periodic",
            )

    # ── 9. Final save ──────────────────────────────────────────────────────
    _pub[0].stop()
    save_checkpoint(
        log_dir / "final_model.pth",
        model, optimizer, scaler, config.num_iter,
        best_accuracy, best_norm_ed, config,
        checkpoint_type="final",
    )
    logger.info("Training complete.")
    if tracker:
        tracker.finish()
