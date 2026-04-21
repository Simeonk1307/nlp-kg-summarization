import json
import re
import random
import time
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import google.generativeai as genai
import google.api_core.exceptions
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
model = genai.GenerativeModel(
    model_name="gemini-2.5-flash-lite",
    system_instruction="You are a JSON-only responder. Output raw JSON with no markdown, no backticks, no explanation. Just the JSON object."
)

# ── Retry config ───────────────────────────────────────────────────────────────
MAX_RETRIES     = 5
BASE_BACKOFF    = 2    # seconds (doubled each retry)
REQUEST_TIMEOUT = 120  # seconds per API call

RETRYABLE = (
    google.api_core.exceptions.DeadlineExceeded,
    google.api_core.exceptions.ServiceUnavailable,
    google.api_core.exceptions.InternalServerError,
    google.api_core.exceptions.ResourceExhausted,
)

SCORE_KEYS = ["faithfulness", "coverage", "reference_alignment", "coherence", "hallucination", "overall"]

# ── Shared call_judge ──────────────────────────────────────────────────────────
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
                raise ValueError(f"No JSON object found in response: {text!r}")

            return json.loads(match.group())

        except RETRYABLE as e:
            wait = BASE_BACKOFF ** attempt
            logger.warning(f"Attempt {attempt}/{MAX_RETRIES} failed ({type(e).__name__}). Retrying in {wait}s...")
            last_exc = e
            time.sleep(wait)

        except (ValueError, json.JSONDecodeError) as e:
            logger.warning(f"Bad model output on attempt {attempt}/{MAX_RETRIES}: {e}. Retrying in {BASE_BACKOFF}s...")
            last_exc = e
            time.sleep(BASE_BACKOFF)

        except Exception as e:
            logger.error(f"Non-retryable error: {type(e).__name__}: {e}")
            raise

    raise RuntimeError(f"All {MAX_RETRIES} attempts failed. Last error: {last_exc}") from last_exc


# ── Pipeline 1: per-dimension scoring ─────────────────────────────────────────
def score_against_reference(reference: str, summary: str) -> dict:
    prompt = f"""You are a summarization evaluator.

Reference Summary:
{reference}

Generated Summary:
{summary}

Rate each dimension 1-5. Respond ONLY in JSON with no extra text:
{{"faithfulness":0,"coverage":0,"reference_alignment":0,"coherence":0,"hallucination":0,"overall":0,"reasoning":""}}"""

    result = call_judge(prompt)

    missing = [k for k in SCORE_KEYS if k not in result]
    if missing:
        raise ValueError(f"Missing score keys: {missing} in {result}")

    for k in SCORE_KEYS:
        if not isinstance(result[k], (int, float)) or not (1 <= result[k] <= 5):
            raise ValueError(f"Score '{k}' out of range or wrong type: {result[k]}")

    return result


def _score_one(args: tuple) -> dict:
    """Worker: scores both no_kg and with_kg for a single sample."""
    i, s = args
    try:
        score_no_kg   = score_against_reference(s["reference_summary"], s["summary_without_kg"])
        score_with_kg = score_against_reference(s["reference_summary"], s["summary_with_kg"])
        return {"index": i, "no_kg": score_no_kg, "with_kg": score_with_kg, "error": None}
    except Exception as e:
        logger.error(f"Sample {i+1} failed permanently: {e}")
        return {"index": i, "no_kg": None, "with_kg": None, "error": str(e)}


def run_pipeline1(samples: list, max_workers: int = 5) -> dict:
    tasks = [(i, s) for i, s in enumerate(samples)]

    all_scores_no_kg   = [None] * len(samples)
    all_scores_with_kg = [None] * len(samples)
    errors = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_score_one, t): t[0] for t in tasks}

        for future in as_completed(futures):
            try:
                r = future.result()
            except Exception as e:
                idx = futures[future]
                logger.error(f"Unexpected error for sample {idx+1}: {e}")
                r = {"index": idx, "no_kg": None, "with_kg": None, "error": str(e)}

            i = r["index"]
            if r["error"]:
                errors += 1
                status = "❌ ERROR"
                print(f"  Sample {i+1:>3} → {status} | {r['error']}")
            else:
                all_scores_no_kg[i]   = r["no_kg"]
                all_scores_with_kg[i] = r["with_kg"]
                status = "✅ done"
                print(f"  Sample {i+1:>3} → {status} | overall no_kg={r['no_kg']['overall']} with_kg={r['with_kg']['overall']}")

    # Only average over valid (non-error) samples
    valid_no_kg   = [s for s in all_scores_no_kg   if s is not None]
    valid_with_kg = [s for s in all_scores_with_kg if s is not None]
    n_valid = len(valid_no_kg)

    print(f"\n{'='*50}")
    print(f"Scoring Results (n={len(samples)}, valid={n_valid}, errors={errors}):")
    print(f"\n{'Dimension':<25} {'NoKG':>8} {'WithKG':>8} {'Delta':>8}")
    print("-" * 50)
    for d in SCORE_KEYS:
        avg_no   = sum(r[d] for r in valid_no_kg)   / n_valid if n_valid else 0
        avg_with = sum(r[d] for r in valid_with_kg) / n_valid if n_valid else 0
        delta    = avg_with - avg_no
        arrow    = "▲" if delta > 0 else ("▼" if delta < 0 else "=")
        print(f"  {d:<23} {avg_no:>8.2f} {avg_with:>8.2f} {arrow}{abs(delta):>6.2f}")
    print(f"{'='*50}")

    return {"no_kg": all_scores_no_kg, "with_kg": all_scores_with_kg}


samples = [
    {
        "triples": [
            ("Obama", "born_in", "Hawaii"),
            ("Obama", "was", "44th President"),
            ("Obama", "party", "Democratic")
        ],
        "reference": "Barack Obama, a Democrat born in Hawaii, served as the 44th President of the United States.",
        "output_no_kg": "Obama was the 44th president and a member of the Democratic party.",
        "output_with_kg": "Barack Obama, born in Hawaii, served as the 44th US President representing the Democratic party."
    }
]

start = time.time()
results = run_pipeline1(samples)
end = time.time()

print(end-start)

with open('summaries_5.jsonl') as f:
    samples = json.load(f)


start = time.time()
results = run_pipeline1(samples)
end = time.time()

print(end-start)