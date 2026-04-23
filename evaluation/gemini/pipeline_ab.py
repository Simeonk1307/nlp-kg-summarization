import json
import re
import random
import time
import logging
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict, List, Tuple, Any
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
MAX_RETRIES     = 5
BASE_BACKOFF    = 2
REQUEST_TIMEOUT = 120
MAX_WORKERS     = 5
RANDOM_SEED     = 42

MAX_SOURCE_WORDS  = 10000
MAX_TRIPLES_COUNT = 500

RETRYABLE = (
    google.api_core.exceptions.DeadlineExceeded,
    google.api_core.exceptions.ServiceUnavailable,
    google.api_core.exceptions.InternalServerError,
    google.api_core.exceptions.ResourceExhausted,
)

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

            # strip markdown fences BEFORE searching for JSON
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)

            match = re.search(r'\{.*\}', text, re.DOTALL)
            if not match:
                raise ValueError(f"No JSON object found in response: {text[:200]!r}...")

            result = json.loads(match.group())

            if "winner" not in result or "reasoning" not in result:
                raise ValueError(f"Missing required keys in JSON: {result}")

            result["winner"] = result["winner"].strip().upper()
            if result["winner"] not in ("A", "B", "TIE"):
                raise ValueError(f"Unexpected winner value: {result['winner']!r}")

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

    truncated = " ".join(words[:max_words])
    last_period = truncated.rfind('.')
    if last_period > len(truncated) * 0.8:
        return truncated[:last_period + 1]

    return truncated + "..."


def _build_prompt(
    source_text:       Optional[str],
    triples:           Optional[List[Tuple]],
    reference_summary: str,
    summary_a:         str,
    summary_b:         str,
) -> str:
    
    
    sections = [
        "You are an expert summarization evaluator. "
        "Compare the two summaries and determine which is better overall.\n"
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
        lines = [f"  ({h}, {r}, {t})" for h, r, t in selected]
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
        "\n[SUMMARY A]\n" +
        "-" * 80 + "\n" +
        summary_a + "\n" +
        "-" * 80
    )
    sections.append(
        "\n[SUMMARY B]\n" +
        "-" * 80 + "\n" +
        summary_b + "\n" +
        "-" * 80
    )

    sections.append(
        "\n[EVALUATION TASK]\n" +
        "-" * 80 + "\n" +
        "Evaluate which summary is better by considering ALL of these criteria:\n"
        "  1. Faithfulness   — does it contradict the source / triples?\n"
        "  2. Coverage       — are the key facts from the source captured?\n"
        "  3. Reference alignment — how well does it match the reference?\n"
        "  4. Coherence      — is it fluent and logically consistent?\n"
        "  5. Conciseness    — is it appropriately concise?\n\n"
        "Respond ONLY in JSON with no extra text:\n"
        '{"winner":"A or B or TIE","reasoning":"detailed explanation"}\n' 
    )

    
    return "".join(sections)



def pairwise_judge(
    reference_summary: str,
    summary_a:         str,
    summary_b:         str,
    label_a:           str,
    label_b:           str,
    source_text:       Optional[str] = None,
    triples:           Optional[List[Tuple]] = None,
) -> dict:
    prompt = _build_prompt(
        source_text=source_text,
        triples=triples,
        reference_summary=reference_summary,
        summary_a=summary_a,
        summary_b=summary_b,
    )

    result = call_judge(prompt)
    winner_raw = result["winner"]

    winner_label = (
        label_a if winner_raw == "A"
        else label_b if winner_raw == "B"
        else "tie"
    )

    return {
        "winner":    winner_label,
        "reasoning": result["reasoning"],
    }



def _judge_one(args: Tuple[int, Dict, bool]) -> Dict[str, Any]:
    i, sample, flip = args

    if flip:
        summary_a, label_a = sample["summary_with_kg"],    "with_kg"
        summary_b, label_b = sample["summary_without_kg"], "no_kg"
    else:
        summary_a, label_a = sample["summary_without_kg"], "no_kg"
        summary_b, label_b = sample["summary_with_kg"],    "with_kg"

    try:
        result = pairwise_judge(
            reference_summary=sample["reference_summary"],
            summary_a=summary_a,
            summary_b=summary_b,
            label_a=label_a,
            label_b=label_b,
            source_text=sample.get("source_text"),
            triples=sample.get("triples"),
        )

        return {
            "index":     i,
            "flipped":   flip,
            "winner":    result["winner"],
            "reasoning": result["reasoning"],
            "error":     None,
        }

    except Exception as e:
        logger.error(f"Sample {i + 1} failed permanently: {e}")
        return {
            "index":     i,
            "flipped":   flip,
            "winner":    "error",
            "reasoning": "",
            "error":     str(e),
        }



def run_judge_pipeline(
    samples:     List[Dict],
    seed:        int = RANDOM_SEED,
    max_workers: int = MAX_WORKERS,
    output_file: Optional[str] = None,
) -> List[Dict]:

    random.seed(seed)
    logger.info(f"Starting evaluation of {len(samples)} samples (seed={seed})")

    flips  = [random.random() > 0.5 for _ in samples]
    tasks  = [(i, s, flips[i]) for i, s in enumerate(samples)]

    
    counts      = defaultdict(int)
    counts_lock = threading.Lock()
    results     = [None] * len(samples)
    start_time  = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_judge_one, t): t[0] for t in tasks}

        with tqdm(total=len(samples), desc="Evaluating", unit="sample") as pbar:
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    r = future.result()
                except Exception as e:
                    logger.error(f"Unexpected error for sample {idx + 1}: {e}")
                    r = {
                        "index":     idx,
                        "flipped":   flips[idx],
                        "winner":    "error",
                        "reasoning": "",
                        "error":     str(e),
                    }

                i = r["index"]

          
                with counts_lock:
                    counts[r["winner"]] += 1
                    snapshot = dict(counts)

                results[i] = {
                    "sample": i + 1,
                    **{k: v for k, v in r.items() if k != "index"},
                }

                pbar.update(1)
                pbar.set_postfix({
                    "with_kg": snapshot.get("with_kg", 0),
                    "no_kg":   snapshot.get("no_kg", 0),
                    "tie":     snapshot.get("tie", 0),
                    "err":     snapshot.get("error", 0),
                })

    elapsed = time.time() - start_time

    
    n      = len(samples)
    errors = counts["error"]
    valid  = n - errors

    print(f"\n{'=' * 70}")
    print(f"{'EVALUATION RESULTS':^70}")
    print(f"{'=' * 70}")
    print(f"  Total Samples    : {n:>6}")
    print(f"  Valid Judgments  : {valid:>6}")
    print(f"  Errors           : {errors:>6}")
    print(f"  Time Elapsed     : {elapsed:>6.1f}s")
    print(f"  Avg Time/Sample  : {elapsed / n:>6.2f}s")
    print(f"{'-' * 70}")

    if valid > 0:
        print(f"  NoKG Wins        : {counts['no_kg']:>6}   "
              f"({100 * counts['no_kg'] / valid:>5.1f}%)")
        print(f"  WithKG Wins      : {counts['with_kg']:>6}   "
              f"({100 * counts['with_kg'] / valid:>5.1f}%)")
        print(f"  Ties             : {counts['tie']:>6}   "
              f"({100 * counts['tie'] / valid:>5.1f}%)")

        diff = abs(counts['with_kg'] - counts['no_kg'])
        print(f"{'-' * 70}")
        print(f"  Win Difference   : {diff:>6}   ({100 * diff / valid:>5.1f}%)")

      
        p1       = counts['with_kg'] / valid
        p2       = counts['no_kg']   / valid
        cohens_h = 2 * abs(p1 ** 0.5 - p2 ** 0.5)

        effect = (
            "negligible" if cohens_h < 0.2
            else "small"  if cohens_h < 0.5
            else "medium" if cohens_h < 0.8
            else "large"
        )
        print(f"  Effect Size (h)  : {cohens_h:>6.3f}   ({effect})")

    print(f"{'=' * 70}\n")

    
    if output_file:
        with open(output_file, 'w') as f:
            json.dump({
                "metadata": {
                    "total_samples":   n,
                    "seed":            seed,
                    "max_workers":     max_workers,
                    "elapsed_seconds": elapsed,
                },
                "counts":  dict(counts),
                "results": results,
            }, f, indent=2)
        logger.info(f"Results saved to {output_file}")

    return results



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="A/B test summarization with KG")
    parser.add_argument("--input",   default="summaries.json",      help="Input file")
    parser.add_argument("--output",  default="./results/evaluation_results.json", help="Output file")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS,    help="Parallel workers")
    parser.add_argument("--seed",    type=int, default=RANDOM_SEED,    help="Random seed")

    args = parser.parse_args()

    logger.info(f"Loading samples from {args.input}")
    with open(args.input) as f:
        content = f.read().strip()
        if content.startswith('['):
            samples = json.loads(content)
        else:
            samples = [json.loads(line) for line in content.split('\n') if line.strip()]

    logger.info(f"Loaded {len(samples)} samples")

    results = run_judge_pipeline(
        samples=samples,
        seed=args.seed,
        max_workers=args.workers,
        output_file=args.output,
    )

    logger.info("Evaluation complete!")