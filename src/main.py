from transformers import AutoTokenizer
from transformers import LongT5ForConditionalGeneration
from utils import *
from kg_extractor import KGExtractor
from kg_embedder import KGEncoder
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from base_model import KATSum

def main():
    """
    Main function for training run.
    """

    # CONFIG
    MODEL_NAME = "google/long-t5-tglobal-base"
    SAVE_DIR = "checkpoints/"
    BATCH_SIZE = 2
    GRAD_ACCUM = 8
    EPOCHS = 3
    LR = 3e-4
    SRC_MAX_LEN = 4096
    TGT_MAX_LEN = 256
    NUM_SIDECAR_LAYERS = 3
    FREEZE_BASE = True
    DEVICE = "cpu"

    print(f"Using device: {DEVICE}")
    Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Load data
    print(f"Loading dataset...")
    dataset = load_dataset("EdinburghNLP/xsum")

    train_articles = [d["document"] for d in dataset["train"]][:5]
    train_summaries = [d["summary"] for d in dataset["train"]][:5]

    val_articles = [d["document"] for d in dataset["validation"]][:2]
    val_summaries = [d["summary"] for d in dataset["validation"]][:2]

    test_articles = [d["document"] for d in dataset["test"]][:2]
    test_summaries = [d["summary"] for d in dataset["test"]][:2]

    # Build KG extractor
    extractor = KGExtractor(device=DEVICE)

    # Build model and KG embedder
    base_model = LongT5ForConditionalGeneration.from_pretrained(MODEL_NAME)

    kg_embedder = KGEncoder(
        encoder=base_model.encoder,
        tokenizer=tokenizer,
        hidden_dim=base_model.config.d_model,
        device=DEVICE,
    )
    
    # Build the KATSum model
    model = KATSum(
        base_model_name=MODEL_NAME,
        kg_embedder=kg_embedder,
        num_sidecar_layers=NUM_SIDECAR_LAYERS,
        freeze_base=FREEZE_BASE,
        device=DEVICE,
    )

    model.parameter_count()  # prints frozen vs trainable summary

    # Build datasets
    train_dataset = SummarizationDataset(
        articles=train_articles,
        summaries=train_summaries,
        tokenizer=tokenizer,
        extractor=extractor,
        src_max_len=SRC_MAX_LEN,
        tgt_max_len=TGT_MAX_LEN,
        cache_triples=True,
    )

    val_dataset = SummarizationDataset(
        articles=val_articles,
        summaries=val_summaries,
        tokenizer=tokenizer,
        extractor=extractor,
        src_max_len=SRC_MAX_LEN,
        tgt_max_len=TGT_MAX_LEN,
        cache_triples=True,
    )

    test_dataset = SummarizationDataset(
        articles=test_articles,
        summaries=test_summaries,
        tokenizer=tokenizer,
        extractor=extractor,
        src_max_len=SRC_MAX_LEN,
        tgt_max_len=TGT_MAX_LEN,
        cache_triples=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
        num_workers=2,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
        num_workers=2,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
        num_workers=2,
    )

    # Optimizer and scheduler
    optimizer = AdamW(model.trainable_parameters(), lr=LR, weight_decay=0.01)

    # Linear warmup
    # Warmup: LR rises from 0 to LR over first 10% of steps.
    total_steps = len(train_loader) * EPOCHS // GRAD_ACCUM
    warmup_steps = total_steps // 10
    scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_steps,
    )

    best_val_loss = float("inf")

    # Training loop
    for epoch in range(EPOCHS):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"{'='*60}")

        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=DEVICE,
            grad_accumulation_steps=GRAD_ACCUM,
        )

        print(f"Train loss: {train_loss:.4f}")

        val_results = evaluate(
            model=model,
            dataloader=val_loader,
            tokenizer=tokenizer,
            device=DEVICE,
        )

        print(
            f"Val loss: {val_results['val_loss']:.4f}  "
            f"ROUGE-1: {val_results['rouge1']:.4f}  "
            f"ROUGE-2: {val_results['rouge2']:.4f}  "
            f"ROUGE-L: {val_results['rougeL']:.4f}  "
            f"Bert Score: {val_results['bert']:.4f}  "
            f"SummaC Score: {val_results['summaC']:.4f}  "
        )

        # Save best model
        if val_results["val_loss"] < best_val_loss:
            best_val_loss = val_results["val_loss"]

            # Save only the new KG layers as the base model can be reloaded from HuggingFace
            checkpoint = {
                "epoch": epoch,
                "kg_sidecar_state_dict": model.kg_sidecar_layers.state_dict(),
                "kg_embedder_state_dict": model.kg_embedder.state_dict(),
                "val_loss": best_val_loss,
                "val_rouge1": val_results["rouge1"],
            }

            save_path = Path(SAVE_DIR) / f"best_checkpoint.pt"
            torch.save(checkpoint, save_path)
            print(f"Saved best model to {save_path}")
            
    print(f"\n{'='*60}")
    print(f"Testing ON TEST Set...")
    print(f"{'='*60}")
    
    # Test on dataset
    test_results = evaluate(
        model=model,
        dataloader=test_loader,
        tokenizer=tokenizer,
        device=DEVICE,
    )

    print(
        f"Test loss:{test_results['val_loss']:.4f}  "
        f"ROUGE-1: {test_results['rouge1']:.4f}  "
        f"ROUGE-2: {test_results['rouge2']:.4f}  "
        f"ROUGE-L: {test_results['rougeL']:.4f}  " 
        f"Bert Score: {test_results['bert']:.4f}  "
        f"SummaC Score: {test_results['summaC']:.4f}"
    )


if __name__ == "__main__":
    main()