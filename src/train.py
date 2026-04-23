import os
import argparse
import logging
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "0"

import torch
from transformers import AutoTokenizer, LongT5ForConditionalGeneration
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.data import DataLoader
from datasets import load_from_disk
from rouge_score.rouge_scorer import RougeScorer

from kg_embedder import KGEncoder
from base_model import KATSum
from utils import SummarizationDataset, collate_fn, train_one_epoch, evaluate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("katsum_train.log"),
    ],
)
log = logging.getLogger(__name__)


def save_checkpoint(
    model, optimizer, scheduler, epoch, best_val_loss, val_results, path, is_best=False
):
    checkpoint = {
        "epoch": epoch,
        "kg_sidecar_state_dict": model.kg_sidecar_layers.state_dict(),
        "kg_embedder_state_dict": model.kg_embedder.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_loss": best_val_loss,
        "val_loss": val_results["val_loss"],
        "val_rouge1": val_results["rouge1"],
        "val_rouge2": val_results["rouge2"],
        "val_rougeL": val_results["rougeL"],
    }
    torch.save(checkpoint, path)
    log.info(f"{'[BEST] ' if is_best else ''}Checkpoint saved -> {path}")


def load_checkpoint(path, model, optimizer, scheduler, device):
    log.info(f"Resuming from checkpoint: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)

    model.kg_sidecar_layers.load_state_dict(ckpt["kg_sidecar_state_dict"])
    model.kg_embedder.load_state_dict(ckpt["kg_embedder_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    start_epoch = ckpt["epoch"] + 1
    best_val_loss = ckpt["best_val_loss"]
    log.info(f"Resumed at epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")
    return start_epoch, best_val_loss


def build_optimizer_and_scheduler(model, lr, total_steps, warmup_steps):
    optimizer = AdamW(model.trainable_parameters(), lr=lr, weight_decay=0.01)
    warmup = LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps
    )
    decay = CosineAnnealingLR(
        optimizer, T_max=max(total_steps - warmup_steps, 1), eta_min=1e-6
    )
    scheduler = SequentialLR(
        optimizer, schedulers=[warmup, decay], milestones=[warmup_steps]
    )

    return optimizer, scheduler


def run_phase(
    *,
    phase_name,
    model,
    train_loader,
    val_loader,
    tokenizer,
    rouge_scorer,
    epochs,
    start_epoch,
    lr,
    grad_accum,
    max_grad_norm,
    best_val_loss,
    best_ckpt_path,
    last_ckpt_path,
    device,
    resume,
):
    steps_per_epoch = len(train_loader) // grad_accum
    total_steps = steps_per_epoch * epochs
    warmup_steps = total_steps // 10

    optimizer, scheduler = build_optimizer_and_scheduler(
        model, lr, total_steps, warmup_steps
    )

    # Try resuming within this phase
    if resume and last_ckpt_path.exists():
        start_epoch, best_val_loss = load_checkpoint(
            last_ckpt_path, model, optimizer, scheduler, device
        )

    log.info(f"\n{'='*60}")
    log.info(
        f"  PHASE: {phase_name}  |  epochs {start_epoch+1} -> {epochs}  |  lr={lr}"
    )
    log.info(
        f"  Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )
    log.info(f"{'='*60}")

    for epoch in range(start_epoch, epochs):
        log.info(f"\n--- {phase_name} | Epoch {epoch+1}/{epochs} ---")

        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            logger=log,
            device=device,
            grad_accumulation_steps=grad_accum,
        )
        log.info(f"Train loss: {train_loss:.4f}")

        log.info(f"Validating {len(val_loader)} batches...")
        val_results = evaluate(
            model=model,
            dataloader=val_loader,
            tokenizer=tokenizer,
            device=device,
            rouge_scorer=rouge_scorer,
            max_new_tokens=256,
        )
        log.info(
            f"Val loss: {val_results['val_loss']:.4f}  "
            f"R-1: {val_results['rouge1']:.4f}  "
            f"R-2: {val_results['rouge2']:.4f}  "
            f"R-L: {val_results['rougeL']:.4f}"
        )

        is_best = val_results["val_loss"] < best_val_loss
        if is_best:
            best_val_loss = val_results["val_loss"]
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                best_val_loss,
                val_results,
                best_ckpt_path,
                is_best=True,
            )

        save_checkpoint(
            model,
            optimizer,
            scheduler,
            epoch,
            best_val_loss,
            val_results,
            last_ckpt_path,
        )

    return best_val_loss


def main(args):
    # ---- Config ------------------------------------------------------------
    MODEL_NAME = "google/long-t5-tglobal-base"
    SAVE_DIR = Path("checkpoints")
    BATCH_SIZE = 4
    GRAD_ACCUM = 16
    MAX_GRAD_NORM = 1.0  # gradient clipping
    SRC_MAX_LEN = 4096
    TGT_MAX_LEN = 512
    NUM_SIDECAR_LAYERS = 12

    # Phase 1: freeze base, train only KG components
    PHASE1_EPOCHS = args.phase1_epochs
    PHASE1_LR = 3e-4


    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Using device: {DEVICE}")

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    P1_BEST = SAVE_DIR / "phase1_best.pt"
    P1_LAST = SAVE_DIR / "phase1_last.pt"

    NUM_VAL_SAMPLES = 1000
    NUM_TRAIN_SAMPLES = 15000

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

  
    log.info("Loading dataset...")
    _train = load_from_disk("./dataset/pubmed_with_triples_v/train")
    _val = load_from_disk("./dataset/pubmed_with_triples_v/validation")
    _test = load_from_disk("./dataset/pubmed_with_triples_v/test")
    log.info(f"Splits  train:{len(_train)}  val:{len(_val)}  test:{len(_test)}")

    def make_dataset(split, val):
        if args.trial:
            return SummarizationDataset(
                articles=split["article"][:2],
                summaries=split["abstract"][:2],
                triples=split["rebel_triples"][:2],
                tokenizer=tokenizer,
                src_max_len=SRC_MAX_LEN,
                tgt_max_len=TGT_MAX_LEN,
            )
        else:
            if val:
                return SummarizationDataset(
                    articles=split["article"][:NUM_VAL_SAMPLES],
                    summaries=split["abstract"][:NUM_VAL_SAMPLES],
                    triples=split["rebel_triples"][:NUM_VAL_SAMPLES],
                    tokenizer=tokenizer,
                    src_max_len=SRC_MAX_LEN,
                    tgt_max_len=TGT_MAX_LEN,
                )
            else:
                return SummarizationDataset(
                    articles=split["article"][:NUM_TRAIN_SAMPLES],
                    summaries=split["abstract"][:NUM_TRAIN_SAMPLES],
                    triples=split["rebel_triples"][:NUM_TRAIN_SAMPLES],
                    tokenizer=tokenizer,
                    src_max_len=SRC_MAX_LEN,
                    tgt_max_len=TGT_MAX_LEN,
                )

    def make_loader(split, shuffle, val):
        return DataLoader(
            make_dataset(split, val),
            batch_size=BATCH_SIZE,
            shuffle=shuffle,
            collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
            num_workers=2,
            pin_memory=(DEVICE == "cuda"),
        )

    train_loader = make_loader(_train, shuffle=True, val=False)
    val_loader = make_loader(_val, shuffle=False, val=True)

    # Model
    base_model = LongT5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    kg_embedder = KGEncoder(
        encoder=base_model.encoder,
        tokenizer=tokenizer,
        hidden_dim=base_model.config.d_model,
        device=DEVICE,
    )

    rouge_scorer = RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    # PHASE 1 — base model frozen, only KG sidecar layers train
    if PHASE1_EPOCHS > 0:
        model = KATSum(
            base_model=base_model,
            kg_embedder=kg_embedder,
            num_sidecar_layers=NUM_SIDECAR_LAYERS,
            freeze_base=True,  # base frozen
            device=DEVICE,
        )
        model.parameter_count()

        best_val_loss = run_phase(
            phase_name="Phase 1 (KG-only)",
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            tokenizer=tokenizer,
            rouge_scorer=rouge_scorer,
            epochs=PHASE1_EPOCHS,
            start_epoch=0,
            lr=PHASE1_LR,
            grad_accum=GRAD_ACCUM,
            max_grad_norm=MAX_GRAD_NORM,
            best_val_loss=float("inf"),
            best_ckpt_path=P1_BEST,
            last_ckpt_path=P1_LAST,
            device=DEVICE,
            resume=args.resume,
        )
        log.info(f"Phase 1 complete. Best val loss: {best_val_loss:.4f}")
    else:
        log.info("Skipping Phase 1 (phase1_epochs=0)")

    log.info("Training complete.")


# Entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KATSum one-phase training")

    parser.add_argument(
        "--phase1_epochs",
        type=int,
        default=3,
        help="Epochs for Phase 1 (base frozen, KG layers only). Set 0 to skip.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint if available.",
    )
    parser.add_argument(
        "--trial",
        action="store_true",
        help="Run on 2 examples per split for quick sanity check.",
    )

    args = parser.parse_args()
    main(args)
