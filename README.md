# TEAM DETAILS

**Team No.**: 3  
**Course Name**: Natural Language Processing  (under **Prof. Swapnil Hingmire**)

| **S. No.** | **Member Name**          | **Roll Number** |
| ---------- | ----------------- | --------------- |
| 1          | Simeon K Sonar    | 112301031       |
| 2          | Shreesh Amit      | 112301046       |
| 3          | Devapriya Pradeep | 112301006       |
| 4          | Babblu G          | 142504001       |





# Custom KATSum — Knowledge-Augmented Text Summarization

Custom KATSum is a knowledge-graph-augmented summarization model built on top of [LongT5](https://huggingface.co/google/long-t5-tglobal-base). It enriches the decoder with structured relational knowledge extracted from source articles via [REBEL](https://huggingface.co/Babelscape/rebel-large), improving factual grounding in generated summaries.

---

## How It Works

1. **Triple Extraction** — REBEL extracts `(subj, predicate, obj)` triples from each source article.
2. **KG Encoding** — Triples are serialized to text and encoded using the LongT5 encoder (via `KGEncoder`), producing one dense vector per triple.
3. **Sidecar Injection** — `KGSidecarLayer` modules are inserted into the last N decoder blocks. Each sidecar performs cross-attention between the decoder's hidden state and the KG embeddings, then blends the result via a learned fusion gate.
4. **Training** — The base LongT5 weights are frozen; only the KG sidecar layers and KG encoder are trained.
5. **Evaluation** — Summaries generated with and without KG augmentation are compared using three Gemini-powered pipelines: A/B preference judging, QuestEval factuality scoring and reference-based dimensional scoring.

---

## File Manifest

```
.
├── src/
│   ├── base_model.py                     # KATSum model: KGSidecarLayer, KATSum
│   ├── kg_embedder.py                    # KGEncoder: encodes triples via T5 encoder
│   ├── kg_extractor.py                   # KGExtractor: REBEL-based triple extraction
│   ├── rebel_triple_extraction.py        # Pre-extraction pipeline for PubMed dataset
│   ├── train.py                          # One-phase training script
│   ├── base_summary_generator.py         # Inference script for the vanilla LongT5 baseline
│   ├── custom_phase_1_summary_generator.py # Inference script for KATSum model
│   ├── utils.py                          # Dataset class, collate_fn, train/eval loops
│   ├── main.ipynb                        # Exploratory notebook
│   ├── checkpoints/
│   │   ├── phase1_best.pt                # Best Phase 1 checkpoint
│   ├── dataset/
│   │   └── pubmed_with_triples_v/        # Pre-extracted PubMed dataset with REBEL triples
│   └── results/
│       ├── base_model_test_results_20.json
│       └── katsum_phase_1_model_test_results_20.json
│
├── evaluation/
│   ├── merger.py                         # Merges base + KATSum outputs into summaries_N.json
│   ├── summaries_20.json                 # Merged summaries for evaluation
│   ├── gemini/
│   │   ├── pipeline_ab.py               # A/B preference judging via Gemini
│   │   ├── pipeline_questeval.py        # QuestEval factuality scoring via Gemini
│   │   └── pipeline_reference.py        # Reference-based dimensional scoring via Gemini
│   └── results/
│       ├── evaluation_results_20.json
│       ├── questeval_results_20.json
│       └── scoring_results_20.json
│
├── future-work/                         # No involvement with project at this point (can ignore)
│   ├── llm-kg-extraction.ipynb          # LLM-based triple extraction experiments
│   └── ollama/                          # Ollama-based local evaluation pipelines
│
└── requirements.txt
```

---

## Installation & Setup


```bash
git clone git@github.com:Simeonk1307/nlp-kg-summarization.git
cd nlp-kg-summarization
```

```bash
bash setup.sh
```

#### For evaluation pipelines, create a `.env` file in the `evaluation/` directory (refer `.env.example`):

```env
GEMINI_API_KEY=your_api_key_here
```

#### For running summary generators, main.ipynb or train.py `src/dataset/pubmed_with_triples_v` must exist 
NOTE: As running `rebel_triple_extraction.py` is lengthy process you may do this

```bash
Take `pubmed_with_triples_v.zip` from the drive and put in project root
unzip pubmed_with_triples_v.zip -d ./src/dataset/
```

#### For running custom summary generator `src/checkpoints/phase1_best.pt` or equivalent must be there. 
NOTE: As runnning `train.py` is lengthy process you may do this

```bash
Take `phase1_best.pt` from the drive and put in project root
cp phase1_best.pt ./src/checkpoints/
```



---

## Dataset Preparation (To skip look at `Installation & Setup`)

Pre-extract REBEL triples from the [PubMed summarization](https://huggingface.co/datasets/ccdv/pubmed-summarization)  when you use dataset and save an enriched copy to disk:

** NOTE: This is also quite slow so advised would be to use the pre-processed dataset from drive **

```bash
cd src
python3 rebel_triple_extraction.py \
    --output_path ./dataset/pubmed_with_triples_v \
    --batch_size 16 \
    --long_strategy chunk
```

Add `--trial` to run on a single example per split for a quick sanity check.

NOTE: It is loaded using `datasets library` in this file (so if you want to use then do `load_dataset(...)`)  

---

## Training (To skip look at `Installation & Setup`)

** NOTE: This is also quite slow so advised would be to use the checkpoint from drive **

```bash
python train.py --phase1_epochs 3
```

Key options:

| Flag | Default | Description |
|---|---|---|
| `--phase1_epochs` | `3` | Epochs to train KG sidecar layers (base model frozen) |
| `--resume` | `False` | Resume from the last saved checkpoint |
| `--trial` | `False` | Run on 2 examples for a quick sanity check |

Checkpoints are saved to `src/checkpoints/`. The best validation checkpoint is saved as `phase1_best.pt`.

---

## Inference

** NOTE: Summary generation is a slow process so GPU would be better to speed up the process **

**Baseline (vanilla LongT5):**

```bash
python base_summary_generator.py \
    --num_samples 20 \
    --result_dir ./results/
```

**KATSum (with KG augmentation):**

```bash
python custom_phase_1_summary_generator.py \
    --num_samples 20 \
    --checkpoint_path ./checkpoints/phase1_best.pt \
    --result_dir ./results/
```

---

## Evaluation

**Step 1 — Merge outputs:** (FAST)

```bash
cd ../evaluation
python merge.py \
  --base ../src/results/base_model_test_results_20.json \
  --other ../src/results/katsum_phase_1_model_test_results_20.json \
  --output summaries_20.json
```

This reads from `src/results/` and writes `summaries_20.json`.

**Step 2 — Run evaluation pipelines:** (FAST but be careful)

** NOTE: Be very mindful while running this. Track your current usage and check if you have billing account or not.
if you do then ensure to cap the amount O/w Try to be within the token and requests limit or change models**

```bash

# A/B preference judging
python gemini/pipeline_ab.py \
    --input summaries_20.json \
    --output results/evaluation_results_20.json

# QuestEval factuality scoring
python gemini/pipeline_questeval.py \
    --input summaries_20.json \
    --output results/questeval_results_20.json

# Reference-based dimensional scoring
python gemini/pipeline_reference.py \
    --input summaries_20.json \
    --output results/scoring_results_20.json
```

### Evaluation Metrics

| Pipeline | What it measures |
|---|---|
| **A/B judging** | Which summary (KG vs. no-KG) is preferred overall, with Cohen's h effect size |
| **QuestEval** | Factuality: can questions generated from the reference be answered from each summary? |
| **Reference scoring** | Dimensional scores (1–5) for faithfulness, coverage, reference alignment, coherence, hallucination, and overall quality |

---

## Architecture Details

### KGSidecarLayer

Each sidecar layer is inserted after the cross-attention sublayer in a decoder block. It:

- Runs cross-attention between the decoder hidden state (query) and KG embeddings (key/value), sharing weights with the block's existing cross-attention layer.
- Applies a sigmoid **fusion gate** to control how much KG context is blended into the decoder state:
  - `gate_bias = 0` → 50% KG / 50% decoder
  - `gate_bias = -1` → ~27% KG / ~73% decoder *(default)*
  - `gate_bias = -2` → ~12% KG / ~88% decoder
- Passes the fused representation through the block's shared FFN.

### KGEncoder

Converts triples `(subj, predicate, obj)` → `"subj predicate obj"` → tokenizes → runs through the frozen T5 encoder → mean-pools token embeddings → returns `(1, num_triples, hidden_dim)`.

### Hook-based injection

Rather than rewriting LongT5's forward pass, sidecar layers are applied via PyTorch `register_forward_hook` on the target decoder blocks. Hooks are registered before each forward/generate call and removed immediately after.

---

## Requirements

Core dependencies:

```
torch
transformers
datasets
rouge-score
spacy
google-generativeai
python-dotenv
tqdm
```

See `requirements.txt` for pinned versions.

---

## Copyright and License

See [LICENSE](LICENSE).

---

## Contact

For inquiries, please email:
[simeon13072005@gmail.com](mailto:simeon13072005@gmail.com) ·
[shreesh2005@gmail.com](mailto:shreesh2005@gmail.com)

---

## Credits

We gratefully acknowledge **Prof. Swapnil Hingmire** for his invaluable guidance, insightful instruction and continued support throughout the course. His mentorship was instrumental in helping us navigate challenges during this project.
We also thank our **teammates** for their collaboration, dedication, and collective effort in bringing this work to completion.

