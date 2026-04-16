import os

os.environ["TOKENIZERS_PARALLELISM"] = "0"

from transformers import AutoTokenizer
from transformers import LongT5ForConditionalGeneration
from utils import *
from kg_embedder import KGEncoder
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR,CosineAnnealingLR,SequentialLR
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from base_model import KATSum
from datasets import load_from_disk
from rouge_score.rouge_scorer import RougeScorer
import logging
import argparse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("longt5_train.log"),
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
    log.info(f"{'[BEST] ' if is_best else ''}Checkpoint saved to {path}")


def load_checkpoint(path, model, optimizer, scheduler, device):
    log.info(f"Resuming from checkpoint: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    model.kg_sidecar_layers.load_state_dict(checkpoint["kg_sidecar_state_dict"])
    model.kg_embedder.load_state_dict(checkpoint["kg_embedder_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    start_epoch = checkpoint["epoch"] + 1  # resume from next epoch
    best_val_loss = checkpoint["best_val_loss"]

    log.info(f"Resumed at epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")
    return start_epoch, best_val_loss


def main(args):
    # CONFIG
    MODEL_NAME = "google/long-t5-tglobal-base"
    SAVE_DIR = "checkpoints/"
    BATCH_SIZE = 4
    GRAD_ACCUM = 16
    EPOCHS = args.epoch
    LR = 3e-4
    SRC_MAX_LEN = 4096
    TGT_MAX_LEN = 512
    NUM_SIDECAR_LAYERS = 4
    FREEZE_BASE = True
    RESUME = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BEST_CKPT_PATH = Path(SAVE_DIR) / "best_checkpoint.pt"
    LAST_CKPT_PATH = Path(SAVE_DIR) / "last_checkpoint.pt"

    log.info(f"Using device: {DEVICE}")
    Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Load data
    log.info("Loading dataset...")
    _train = load_from_disk("./dataset/pubmed_with_triples_rebalanced/train")
    _val = load_from_disk("./dataset/pubmed_with_triples_rebalanced/validation")
    _test = load_from_disk("./dataset/pubmed_with_triples_rebalanced/test")
    log.info(f"Splits train:{len(_train)}  val:{len(_val)}  test:{len(_test)}")

    # Build model
    base_model = LongT5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    kg_embedder = KGEncoder(
        encoder=base_model.encoder,
        tokenizer=tokenizer,
        hidden_dim=base_model.config.d_model,
        device=DEVICE,
    )

    model = KATSum(
        base_model=base_model,
        kg_embedder=kg_embedder,
        num_sidecar_layers=NUM_SIDECAR_LAYERS,
        freeze_base=FREEZE_BASE,
        device=DEVICE,
    )
    model.parameter_count()

    # Datasets and loaders
    def make_dataset(split):
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
            return SummarizationDataset(
                articles=split["article"],
                summaries=split["abstract"],
                triples=split["rebel_triples"],
                tokenizer=tokenizer,
                src_max_len=SRC_MAX_LEN,
                tgt_max_len=TGT_MAX_LEN,
            )

    train_loader = DataLoader(
        make_dataset(_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
        num_workers=2,
    )
    val_loader = DataLoader(
        make_dataset(_val),
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
        num_workers=2,
    )

    # Optimizer and scheduler
    optimizer = AdamW(model.trainable_parameters(), lr=LR, weight_decay=0.01)
    steps_per_epoch = len(train_loader) // GRAD_ACCUM
    total_steps = steps_per_epoch * EPOCHS
    warmup_steps = total_steps // 10

    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps
    )
    decay_scheduler = CosineAnnealingLR(
        optimizer, T_max=max(total_steps - warmup_steps, 1), eta_min=1e-6
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, decay_scheduler],
        milestones=[warmup_steps],
    )
    
    rouge_scorer = RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    # Resume logic
    start_epoch = 0
    best_val_loss = float("inf")

    if RESUME and LAST_CKPT_PATH.exists():
        start_epoch, best_val_loss = load_checkpoint(
            LAST_CKPT_PATH, model, optimizer, scheduler, DEVICE
        )
    elif RESUME and BEST_CKPT_PATH.exists():
        start_epoch, best_val_loss = load_checkpoint(
            BEST_CKPT_PATH, model, optimizer, scheduler, DEVICE
        )
    else:
        log.info("Starting fresh training run.")

    # Training loop
    log.info(f"Training begin from {start_epoch} to {EPOCHS}")
    for epoch in range(start_epoch, EPOCHS):
        log.info(f"\n{'='*60}")
        log.info(f"Epoch {epoch+1}/{EPOCHS}")
        log.info(f"\n{'='*60}")

        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            logger=log,
            device=DEVICE,
            grad_accumulation_steps=GRAD_ACCUM,
        )

        log.info(f"Train loss: {train_loss:.4f}")

        log.info(f"Validating {len(val_loader)} batches from validation set...")
        val_results = evaluate(
            model=model,
            dataloader=val_loader,
            tokenizer=tokenizer,
            device=DEVICE,
            rouge_scorer=rouge_scorer,
            max_new_tokens=512,
        )

        log.info(
            f"Val loss: {val_results['val_loss']:.4f}  "
            f"ROUGE-1: {val_results['rouge1']:.4f}  "
            f"ROUGE-2: {val_results['rouge2']:.4f}  "
            f"ROUGE-L: {val_results['rougeL']:.4f}  "
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
                BEST_CKPT_PATH,
                is_best=True,
            )

        save_checkpoint(
            model,
            optimizer,
            scheduler,
            epoch,
            best_val_loss,
            val_results,
            LAST_CKPT_PATH,
            is_best=False,
        )

    log.info("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LongT5 fine-tuning with Knowledge Graph Triples"
    )

    parser.add_argument(
        "--trial",
        action="store_true",
        help="Run on 1 example per split for quick testing.",
    )

    parser.add_argument(
        "--epoch", type=int, default=1, help="Absolute number of epochs to run for"
    )

    args = parser.parse_args()

    main(args)
