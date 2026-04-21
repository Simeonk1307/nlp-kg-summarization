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

# ── Retry config ──────────────────────────────────────────────────────────────
MAX_RETRIES  = 5
BASE_BACKOFF = 2      # seconds (doubled each retry)
REQUEST_TIMEOUT = 120 # seconds per API call

RETRYABLE = (
    google.api_core.exceptions.DeadlineExceeded,
    google.api_core.exceptions.ServiceUnavailable,
    google.api_core.exceptions.InternalServerError,
    google.api_core.exceptions.ResourceExhausted,  # 429 rate limit
)

def call_judge(prompt: str) -> dict:
    """Call Gemini with retries, backoff, and timeout. Returns parsed JSON dict."""
    last_exc = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = model.generate_content(
                prompt,
                request_options={"timeout": REQUEST_TIMEOUT}
            )

            # Guard: blocked / empty response
            if not response.candidates:
                raise ValueError("Response has no candidates (likely safety-blocked).")

            text = response.text.strip()

            # Strip markdown fences if model misbehaves
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)

            match = re.search(r'\{.*\}', text, re.DOTALL)
            if not match:
                raise ValueError(f"No JSON object found in response: {text!r}")

            result = json.loads(match.group())

            # Validate expected keys
            if "winner" not in result or "reasoning" not in result:
                raise ValueError(f"Missing keys in JSON: {result}")

            # Normalize winner field
            result["winner"] = result["winner"].strip().upper()
            if result["winner"] not in ("A", "B", "TIE"):
                raise ValueError(f"Unexpected winner value: {result['winner']!r}")

            return result

        except RETRYABLE as e:
            wait = BASE_BACKOFF ** attempt
            logger.warning(f"Attempt {attempt}/{MAX_RETRIES} failed ({type(e).__name__}). Retrying in {wait}s...")
            last_exc = e
            time.sleep(wait)

        except (ValueError, json.JSONDecodeError) as e:
            # Bad output — retry once, but don't keep hammering
            wait = BASE_BACKOFF
            logger.warning(f"Bad model output on attempt {attempt}/{MAX_RETRIES}: {e}. Retrying in {wait}s...")
            last_exc = e
            time.sleep(wait)

        except Exception as e:
            # Non-retryable (auth error, bad request, etc.) — fail fast
            logger.error(f"Non-retryable error: {type(e).__name__}: {e}")
            raise

    raise RuntimeError(f"All {MAX_RETRIES} attempts failed. Last error: {last_exc}") from last_exc


def pairwise_judge(reference: str, summary_a: str, summary_b: str,
                   label_a: str, label_b: str) -> dict:
    prompt = f"""You are a summarization evaluator.

Reference Summary:
{reference}

Summary A:
{summary_a}

Summary B:
{summary_b}

Which summary is better overall? Consider faithfulness, reference alignment, coherence, coverage.
Respond ONLY in JSON with no extra text:
{{"winner":"A or B or tie","reasoning":""}}"""

    result = call_judge(prompt)
    winner_raw = result["winner"]
    winner_label = (
        label_a if winner_raw == "A"
        else label_b if winner_raw == "B"
        else "tie"
    )
    return {"winner": winner_label, "reasoning": result["reasoning"]}


def _judge_one(args: tuple) -> dict:
    """Worker for a single sample — used by the thread pool."""
    i, s, flip = args

    # FIX: actually swap positions when flipping, not just labels
    if flip:
        summary_a, label_a = s["summary_with_kg"],    "with_kg"
        summary_b, label_b = s["summary_without_kg"], "no_kg"
    else:
        summary_a, label_a = s["summary_without_kg"], "no_kg"
        summary_b, label_b = s["summary_with_kg"],    "with_kg"

    try:
        result = pairwise_judge(s["reference_summary"], summary_a, summary_b, label_a, label_b)
        return {"index": i, "flipped": flip, "error": None, **result}
    except Exception as e:
        logger.error(f"Sample {i+1} failed permanently: {e}")
        return {"index": i, "flipped": flip, "winner": "error", "reasoning": str(e), "error": str(e)}


def run_pipeline2(samples: list, seed: int = 42, max_workers: int = 5) -> list:
    random.seed(seed)
    flips = [random.random() > 0.5 for _ in samples]
    tasks = [(i, s, flips[i]) for i, s in enumerate(samples)]

    counts  = defaultdict(int)
    results = [None] * len(samples)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_judge_one, t): t[0] for t in tasks}

        for future in as_completed(futures):
            try:
                r = future.result()
            except Exception as e:
                # Should not reach here (errors are caught in _judge_one), but just in case
                idx = futures[future]
                logger.error(f"Unexpected error for sample {idx+1}: {e}")
                r = {"index": idx, "flipped": flips[idx], "winner": "error",
                     "reasoning": str(e), "error": str(e)}

            i = r["index"]
            counts[r["winner"]] += 1
            results[i] = {"sample": i + 1, **{k: v for k, v in r.items() if k != "index"}}

            status = "❌ ERROR" if r["winner"] == "error" else f"✅ {r['winner']}"
            preview = r["reasoning"][:500] if r["reasoning"] else ""
            print(f"  Sample {i+1:>3} → {status} | {preview}...")

    n = len(samples)
    errors = counts.get("error", 0)
    valid  = n - errors

    print(f"\n{'='*50}")
    print(f"A/B Results (n={n}, valid={valid}, errors={errors}):")
    print(f"  NoKG wins   : {counts['no_kg']}   ({100*counts['no_kg']/n:.1f}%)")
    print(f"  WithKG wins : {counts['with_kg']}  ({100*counts['with_kg']/n:.1f}%)")
    print(f"  Ties        : {counts['tie']}   ({100*counts['tie']/n:.1f}%)")
    if errors:
        print(f"  Errors      : {errors}   ({100*errors/n:.1f}%)")
    print(f"{'='*50}")

    return results


samples = [
    {
        "reference_summary": "Barack Obama, a Democrat born in Hawaii, served as the 44th President of the United States.",
        "summary_without_kg": "Obama was the 44th president and a member of the Democratic party.",
        "summary_with_kg": "Barack Obama, born in Hawaii, served as the 44th US President representing the Democratic party."
    }
]

start = time.time()
results = run_pipeline2(samples)
end = time.time()

print(end-start)


with open('summaries_5.jsonl') as f:
    samples = json.load(f)


start = time.time()
results = run_pipeline2(samples)
end = time.time()

print(end-start)