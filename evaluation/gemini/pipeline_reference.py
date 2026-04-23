import json
import re
import random
import time
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Tuple
import google.generativeai as genai
import google.api_core.exceptions
import os
from dotenv import load_dotenv, find_dotenv
from tqdm import tqdm

load_dotenv(find_dotenv())

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
model = genai.GenerativeModel(
    model_name="gemini-2.5-flash-lite",
    system_instruction=(
        "You are a JSON-only responder. "
        "Output raw JSON with no markdown, no backticks, no explanation. "
        "Just the JSON object."
    )
)

# Config
MAX_RETRIES       = 5
BASE_BACKOFF      = 2
REQUEST_TIMEOUT   = 120
MAX_WORKERS       = 5
MAX_SOURCE_WORDS  = 10000
MAX_TRIPLES_COUNT = 500

RETRYABLE = (
    google.api_core.exceptions.DeadlineExceeded,
    google.api_core.exceptions.ServiceUnavailable,
    google.api_core.exceptions.InternalServerError,
    google.api_core.exceptions.ResourceExhausted,
)

SCORE_KEYS = [
    "faithfulness", "coverage", "reference_alignment",
    "coherence", "hallucination", "overall",
]



def call_judge(prompt: str) -> dict:
    last_exc = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = model.generate_content(
                prompt,
                request_options={"timeout": REQUEST_TIMEOUT}
            )
            if not response.candidates:
                raise ValueError("Response has no candidates (likely safety-blocked).")

            text = response.text.strip()
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)

            match = re.search(r'\{.*\}', text, re.DOTALL)
            if not match:
                raise ValueError(f"No JSON object found in response: {text[:200]!r}...")

            result = json.loads(match.group())

            missing = [k for k in SCORE_KEYS if k not in result]
            if missing:
                raise ValueError(f"Missing score keys: {missing} in {result}")

            for k in SCORE_KEYS:
                if not isinstance(result[k], (int, float)) or not (1 <= result[k] <= 5):
                    raise ValueError(
                        f"Score '{k}' out of range or wrong type: {result[k]}"
                    )

            return result  

        except RETRYABLE as e:
            wait = BASE_BACKOFF ** attempt + random.uniform(0, 1)
            logger.warning(
                f"Attempt {attempt}/{MAX_RETRIES} failed ({type(e).__name__}). "
                f"Retrying in {wait:.1f}s..."
            )
            last_exc = e
            time.sleep(wait)

        except (ValueError, json.JSONDecodeError) as e:
            wait = BASE_BACKOFF + random.uniform(0, 1)
            logger.warning(
                f"Bad model output on attempt {attempt}/{MAX_RETRIES}: {e}. "
                f"Retrying in {wait:.1f}s..."
            )
            last_exc = e
            time.sleep(wait)

        except Exception as e:
            logger.error(f"Non-retryable error: {type(e).__name__}: {e}")
            raise

    raise RuntimeError(
        f"All {MAX_RETRIES} attempts failed. Last error: {last_exc}"
    ) from last_exc


def _truncate_text(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text

    truncated  = " ".join(words[:max_words])
    last_period = truncated.rfind('.')
    if last_period > len(truncated) * 0.8:
        return truncated[:last_period + 1]

    return truncated + "..."


def _build_prompt(
    reference_summary: str,
    summary:           str,
    source_text:       Optional[str] = None,
    triples:           Optional[List[Tuple]] = None,
) -> str:
    
    sections = [
        "You are an expert summarization evaluator.\n",
        "=" * 80,
    ]

    if source_text and source_text.strip():
        truncated = _truncate_text(source_text.strip(), MAX_SOURCE_WORDS)
        sections.append(
            "\n[SOURCE TEXT]\n" +
            "-" * 80 + "\n" +
            truncated + "\n" +
            "-" * 80
        )

    if triples:
        selected = triples[:MAX_TRIPLES_COUNT]
        lines = [f"  • ({h}, {r}, {t})" for h, r, t in selected]
        sections.append(
            "\n[KNOWLEDGE GRAPH TRIPLES]\n" +
            "-" * 80 + "\n" +
            "\n".join(lines) + "\n" +
            "-" * 80
        )

    sections.append(
        "\n[REFERENCE SUMMARY]\n" +
        "-" * 80 + "\n" +
        reference_summary + "\n" +
        "-" * 80
    )

    sections.append(
        "\n[GENERATED SUMMARY]\n" +
        "-" * 80 + "\n" +
        summary + "\n" +
        "-" * 80
    )

    sections.append(
        "\n[EVALUATION TASK]\n" +
        "-" * 80 + "\n"
        "Rate the generated summary on each dimension from 1 to 5:\n\n"
        "  • faithfulness (1-5): Does it stay true to the source? No contradictions?\n"
        "  • coverage (1-5): Are key facts from the source captured?\n"
        "  • reference_alignment (1-5): How well does it match the reference?\n"
        "  • coherence (1-5): Is it fluent and logically consistent?\n"
        "  • hallucination (1-5): Does it avoid adding unsupported information? "
        "(5=no hallucination)\n"
        "  • overall (1-5): Overall quality of the summary\n\n"
        "Respond ONLY in JSON format:\n"
        '{"faithfulness":0,"coverage":0,"reference_alignment":0,'
        '"coherence":0,"hallucination":0,"overall":0,"reasoning":""}\n' +
        "-" * 80
    )

    return "".join(sections)  




def score_against_reference(
    reference_summary: str,
    summary:           str,
    source_text:       Optional[str] = None,
    triples:           Optional[List[Tuple]] = None,
) -> dict:
    prompt = _build_prompt(
        reference_summary=reference_summary,
        summary=summary,
        source_text=source_text,
        triples=triples,
    )
    return call_judge(prompt)




def _score_one(args: Tuple[int, Dict]) -> Dict:
    i, sample = args
    try:
       
        score_no_kg = score_against_reference(
            reference_summary=sample["reference_summary"],
            summary=sample["summary_without_kg"],
            source_text=sample.get("source_text"),
            triples=sample.get("triples"),
        )

        score_with_kg = score_against_reference(
            reference_summary=sample["reference_summary"],
            summary=sample["summary_with_kg"],
            source_text=sample.get("source_text"),
            triples=sample.get("triples"),
        )

        return {
            "index":   i,
            "no_kg":   score_no_kg,
            "with_kg": score_with_kg,
            "error":   None,
        }

    except Exception as e:
        logger.error(f"Sample {i + 1} failed permanently: {e}")
        return {
            "index":   i,
            "no_kg":   None,
            "with_kg": None,
            "error":   str(e),
        }


def run_scoring_pipeline(
    samples:     List[Dict],
    max_workers: int = MAX_WORKERS,
    output_file: Optional[str] = None,
) -> Dict:

    logger.info(f"Starting scoring evaluation of {len(samples)} samples")

    tasks              = [(i, s) for i, s in enumerate(samples)]
    all_scores_no_kg   = [None] * len(samples)
    all_scores_with_kg = [None] * len(samples)

    errors      = 0
    errors_lock = threading.Lock()
    start_time  = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_score_one, t): t[0] for t in tasks}

        with tqdm(total=len(samples), desc="Scoring", unit="sample") as pbar:
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    r = future.result()
                except Exception as e:
                    logger.error(f"Unexpected error for sample {idx + 1}: {e}")
                    r = {"index": idx, "no_kg": None, "with_kg": None, "error": str(e)}

                i = r["index"]
                if r["error"]:
                    with errors_lock:
                        errors += 1
                        err_snapshot = errors
                else:
                    all_scores_no_kg[i]   = r["no_kg"]
                    all_scores_with_kg[i] = r["with_kg"]
                    with errors_lock:
                        err_snapshot = errors

                pbar.update(1)
                pbar.set_postfix({"errors": err_snapshot})

    elapsed = time.time() - start_time

    valid_no_kg   = [s for s in all_scores_no_kg   if s is not None]
    valid_with_kg = [s for s in all_scores_with_kg if s is not None]
    n_valid       = len(valid_no_kg)

    print(f"\n{'=' * 70}")
    print(f"{'SCORING RESULTS':^70}")
    print(f"{'=' * 70}")
    print(f"  Total Samples    : {len(samples):>6}")
    print(f"  Valid Scores     : {n_valid:>6}")
    print(f"  Errors           : {errors:>6}")
    print(f"  Time Elapsed     : {elapsed:>6.1f}s")
    print(f"  Avg Time/Sample  : {elapsed / len(samples):>6.2f}s")
    print(f"{'-' * 70}")
    print(f"\n{'Dimension':<25} {'NoKG':>8} {'WithKG':>8} {'Delta':>8}")
    print("-" * 70)

    dimension_stats = {}
    for d in SCORE_KEYS:
        avg_no   = sum(r[d] for r in valid_no_kg)   / n_valid if n_valid else 0
        avg_with = sum(r[d] for r in valid_with_kg) / n_valid if n_valid else 0
        delta    = avg_with - avg_no
        arrow    = "▲" if delta > 0.05 else ("▼" if delta < -0.05 else "≈")

        dimension_stats[d] = {
            "no_kg_avg":   avg_no,
            "with_kg_avg": avg_with,
            "delta":       delta,
        }

        print(f"  {d:<23} {avg_no:>8.2f} {avg_with:>8.2f} {arrow}{abs(delta):>6.2f}")

    print(f"{'=' * 70}\n")

    
    if output_file:
        with open(output_file, 'w') as f:
            json.dump({
                "metadata": {
                    "total_samples":   len(samples),
                    "valid_samples":   n_valid,
                    "errors":          errors,
                    "elapsed_seconds": elapsed,
                },
                "dimension_stats": dimension_stats,
                "all_scores": {
                    "no_kg":   all_scores_no_kg,
                    "with_kg": all_scores_with_kg,
                },
            }, f, indent=2)
        logger.info(f"Results saved to {output_file}")

    return {
        "no_kg":   all_scores_no_kg,
        "with_kg": all_scores_with_kg,
        "stats":   dimension_stats,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Score summaries against reference")
    parser.add_argument("--input",   default="summaries.json",    help="Input file")
    parser.add_argument("--output",  default="./results/scoring_results.json", help="Output file")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS,  help="Parallel workers")

    args = parser.parse_args()

    logger.info(f"Loading samples from {args.input}")
    with open(args.input) as f:
        content = f.read().strip()

    
    content = re.sub(r"^\s*//.*$", "", content, flags=re.MULTILINE)

    if content.startswith('['):
        try:
            samples = json.loads(content)
            logger.info("Parsed input as JSON array.")
        except json.JSONDecodeError as e:
            logger.warning(
                f"Input starts with '[' but failed JSON array parse ({e}). "
                "Falling back to JSONL."
            )
            samples = None
    else:
        samples = None

    if samples is None:
        lines  = [ln.strip() for ln in content.splitlines() if ln.strip()]
        errors = []
        samples = []
        for lineno, line in enumerate(lines, 1):
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as e:
                errors.append((lineno, e))
        for lineno, e in errors:
            logger.warning(f"  Skipped line {lineno}: {e}")
        if not samples:
            raise SystemExit(
                f"Could not load any samples from {args.input!r}. "
                "Check that it is a valid JSON array or JSONL file."
            )

    logger.info(f"Loaded {len(samples)} samples")

    results = run_scoring_pipeline(
        samples=samples,
        max_workers=args.workers,
        output_file=args.output,
    )

    logger.info("Scoring complete!")