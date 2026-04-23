import json
import re
import random
import time
import logging
import threading
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


MAX_RETRIES       = 5
BASE_BACKOFF      = 2
REQUEST_TIMEOUT   = 120
MAX_WORKERS       = 5
RANDOM_SEED       = 42
NUM_QUESTIONS     = 5
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
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)

            match = re.search(r'\{.*\}', text, re.DOTALL)
            if not match:
                raise ValueError(f"No JSON object found in response: {text[:200]!r}...")

            return json.loads(match.group())

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



def _get_field(sample: Dict, *keys) -> Any:
    """Get first available field from sample, return empty default if none exist."""
    for key in keys:
        if key in sample and sample[key]:
            return sample[key]
    return None


def _get_triples(sample: Dict) -> List[Tuple]:
    """Extract triples from sample, default to empty list."""
    triples = _get_field(sample, "triples", "kg_triples", "knowledge_graph")
    if isinstance(triples, list):
        return triples
    return []


def _get_reference(sample: Dict) -> str:
    """Extract reference summary from sample."""
    ref = _get_field(sample, "reference_summary", "reference", "gold_summary")
    return ref if ref else ""


def _get_summary_no_kg(sample: Dict) -> str:
    """Extract no-KG summary from sample."""
    summary = _get_field(sample, "summary_without_kg", "summary_no_kg", "output_no_kg")
    return summary if summary else ""


def _get_summary_with_kg(sample: Dict) -> str:
    """Extract with-KG summary from sample."""
    summary = _get_field(sample, "summary_with_kg", "summary_kg", "output_with_kg")
    return summary if summary else ""



def _build_prompt(
    task: str,
    triples: List[Tuple] = None,
    reference: str = "",
    summary_a: Optional[str] = None,
    summary_b: Optional[str] = None,
    questions: Optional[List[Dict]] = None,
) -> str:
    """Build prompts for question generation, answering, and scoring."""
    
    if triples is None:
        triples = []
    
    sections = []

    if task == "generate_questions":
        sections.append(
            "You are a QA generation expert.\n"
        )
        
        if triples:
            sections.append(
                "\n[KG TRIPLES]\n" +
                "-" * 80 + "\n" +
                "\n".join([f"  ({h}, {r}, {t})" for h, r, t in triples[:MAX_TRIPLES_COUNT]]) + "\n" +
                "-" * 80
            )
        
        if reference:
            sections.append(
                "\n[REFERENCE SUMMARY]\n" +
                "-" * 80 + "\n" +
                reference + "\n" +
                "-" * 80
            )
        
        sections.append(
            f"\n[TASK]\n" +
            "-" * 80 + "\n" +
            f"Generate {NUM_QUESTIONS} factual questions answerable from the reference above.\n"
            "Each question must have a clear short answer.\n"
            'Respond ONLY in JSON:\n'
            '{"questions":[{"q":"","a":""}]}\n' +
            "-" * 80
        )

    elif task == "answer_questions":
        sections.append(
            "You are a QA system. Answer each question using ONLY the summary below.\n"
            'If not answerable, say "unanswerable".\n'
        )
        
        if summary_a:
            sections.append(
                "\n[SUMMARY]\n" +
                "-" * 80 + "\n" +
                summary_a + "\n" +
                "-" * 80
            )
        
        if questions:
            questions_text = "\n".join(
                [f"{i + 1}. {q.get('q', '')}" for i, q in enumerate(questions)]
            )
            sections.append(
                "\n[QUESTIONS]\n" +
                "-" * 80 + "\n" +
                questions_text + "\n" +
                "-" * 80
            )
        
        sections.append(
            "\n[TASK]\n" +
            "-" * 80 + "\n" +
            'Respond ONLY in JSON:\n'
            '{"answers":[""]}\n' +
            "-" * 80
        )

    elif task == "score_answers":
        sections.append(
            "You are an answer evaluator.\n"
            "Score 1 if the predicted answer is correct or semantically equivalent to expected, else 0.\n"
        )
        
        if questions and summary_b:
            qa_pairs = "\n".join([
                f"Q: {q.get('q', '')}\nExpected: {q.get('a', '')}\nPredicted: {summary_b[i] if i < len(summary_b) else ''}"
                for i, q in enumerate(questions)
            ])
            
            sections.append(
                "\n[QA PAIRS]\n" +
                "-" * 80 + "\n" +
                qa_pairs + "\n" +
                "-" * 80
            )
        
        sections.append(
            "\n[TASK]\n" +
            "-" * 80 + "\n" +
            'Respond ONLY in JSON:\n'
            '{"scores":[0]}\n' +
            "-" * 80
        )

    return "".join(sections)



def _generate_questions(triples: List[Tuple], reference: str) -> List[Dict]:
    if not reference and not triples:
        logger.warning("No reference or triples provided for question generation")
        return []
    
    prompt = _build_prompt("generate_questions", triples, reference)
    result = call_judge(prompt)
    questions = result.get("questions", [])
    if not isinstance(questions, list):
        questions = []
    return questions


def _answer_questions(summary: str, questions: List[Dict]) -> List[str]:
    if not summary or not questions:
        return [""] * len(questions) if questions else []
    
    prompt = _build_prompt("answer_questions", summary_a=summary, questions=questions)
    result = call_judge(prompt)
    answers = result.get("answers", [])
    if not isinstance(answers, list):
        answers = []
    return answers


def _score_answers(questions: List[Dict], answers: List[str]) -> List[int]:
    if not questions or not answers:
        return [0] * len(questions) if questions else []
    
    prompt = _build_prompt("score_answers", summary_b=answers, questions=questions)
    result = call_judge(prompt)
    scores = result.get("scores", [])
    if not isinstance(scores, list):
        scores = []
    return scores


def questeval_sample(sample: Dict) -> Dict[str, Any]:
    """Run QuestEval on a single sample with flexible field extraction."""
    
    triples = _get_triples(sample)
    reference = _get_reference(sample)
    summary_no_kg = _get_summary_no_kg(sample)
    summary_with_kg = _get_summary_with_kg(sample)
    
    questions = _generate_questions(triples, reference)

    if not questions:
        logger.warning("No questions generated, skipping evaluation")
        return {
            "questions": [],
            "no_kg":   {"answers": [], "scores": [], "factuality": 0.0},
            "with_kg": {"answers": [], "scores": [], "factuality": 0.0},
        }

    answers_no_kg   = _answer_questions(summary_no_kg, questions)
    answers_with_kg = _answer_questions(summary_with_kg, questions)

    scores_no_kg   = _score_answers(questions, answers_no_kg)
    scores_with_kg = _score_answers(questions, answers_with_kg)

    factuality_no_kg   = sum(scores_no_kg) / len(scores_no_kg) if scores_no_kg else 0.0
    factuality_with_kg = sum(scores_with_kg) / len(scores_with_kg) if scores_with_kg else 0.0

    return {
        "questions": questions,
        "no_kg":   {"answers": answers_no_kg,   "scores": scores_no_kg,   "factuality": factuality_no_kg},
        "with_kg": {"answers": answers_with_kg, "scores": scores_with_kg, "factuality": factuality_with_kg},
    }



def _evaluate_one(args: Tuple[int, Dict]) -> Dict[str, Any]:
    i, sample = args
    
    try:
        result = questeval_sample(sample)
        
        return {
            "index":      i,
            "no_kg":      result["no_kg"],
            "with_kg":    result["with_kg"],
            "questions":  result["questions"],
            "error":      None,
        }
        
    except Exception as e:
        logger.error(f"Sample {i + 1} failed: {e}")
        return {
            "index":     i,
            "no_kg":     {"factuality": 0.0, "answers": [], "scores": []},
            "with_kg":   {"factuality": 0.0, "answers": [], "scores": []},
            "questions": [],
            "error":     str(e),
        }



def run_qeval_pipeline(
    samples:     List[Dict],
    seed:        int = RANDOM_SEED,
    max_workers: int = MAX_WORKERS,
    output_file: Optional[str] = None,
) -> List[Dict]:
    
    random.seed(seed)
    logger.info(f"Starting QuestEval evaluation of {len(samples)} samples (seed={seed})")

    tasks = [(i, s) for i, s in enumerate(samples)]

    total_no_kg    = 0.0
    total_with_kg  = 0.0
    error_count    = 0
    totals_lock    = threading.Lock()
    results        = [None] * len(samples)
    start_time     = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_evaluate_one, t): t[0] for t in tasks}

        with tqdm(total=len(samples), desc="Evaluating", unit="sample") as pbar:
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    r = future.result()
                except Exception as e:
                    logger.error(f"Unexpected error for sample {idx + 1}: {e}")
                    r = {
                        "index":     idx,
                        "no_kg":     {"factuality": 0.0, "answers": [], "scores": []},
                        "with_kg":   {"factuality": 0.0, "answers": [], "scores": []},
                        "questions": [],
                        "error":     str(e),
                    }

                i = r["index"]
                
                with totals_lock:
                    if r["error"] is None:
                        total_no_kg   += r["no_kg"]["factuality"]
                        total_with_kg += r["with_kg"]["factuality"]
                    else:
                        error_count += 1
                    snapshot_no_kg   = total_no_kg
                    snapshot_with_kg = total_with_kg
                    snapshot_errors  = error_count

                results[i] = {
                    "sample": i + 1,
                    **{k: v for k, v in r.items() if k != "index"},
                }

                pbar.update(1)
                valid = (i + 1) - snapshot_errors
                pbar.set_postfix({
                    "no_kg":   f"{snapshot_no_kg  / max(valid, 1):.3f}",
                    "with_kg": f"{snapshot_with_kg / max(valid, 1):.3f}",
                    "err":     snapshot_errors,
                })

    elapsed = time.time() - start_time

    n       = len(samples)
    valid   = n - error_count
    avg_no_kg   = total_no_kg   / max(valid, 1)
    avg_with_kg = total_with_kg / max(valid, 1)

    print(f"\n{'=' * 70}")
    print(f"{'QUESTEVAL RESULTS':^70}")
    print(f"{'=' * 70}")
    print(f"  Total Samples    : {n:>6}")
    print(f"  Valid Samples    : {valid:>6}")
    print(f"  Errors           : {error_count:>6}")
    print(f"  Time Elapsed     : {elapsed:>6.1f}s")
    print(f"  Avg Time/Sample  : {elapsed / n:>6.2f}s")
    print(f"{'-' * 70}")

    if valid > 0:
        print(f"  NoKG   Factuality: {avg_no_kg:>6.3f}   ({avg_no_kg * 100:>5.1f}%)")
        print(f"  WithKG Factuality: {avg_with_kg:>6.3f}   ({avg_with_kg * 100:>5.1f}%)")
        diff = avg_with_kg - avg_no_kg
        sign = "+" if diff >= 0 else ""
        print(f"{'-' * 70}")
        print(f"  Delta (WithKG-NoKG): {sign}{diff:>6.3f}   ({sign}{diff * 100:>5.1f}%)")

    print(f"{'=' * 70}\n")

    if output_file:
        with open(output_file, 'w') as f:
            json.dump({
                "metadata": {
                    "total_samples":   n,
                    "valid_samples":   valid,
                    "seed":            seed,
                    "max_workers":     max_workers,
                    "elapsed_seconds": elapsed,
                },
                "aggregate": {
                    "avg_no_kg_factuality":   avg_no_kg,
                    "avg_with_kg_factuality": avg_with_kg,
                    "delta":                  avg_with_kg - avg_no_kg,
                },
                "results": results,
            }, f, indent=2)
        logger.info(f"Results saved to {output_file}")

    return results



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="QuestEval factuality scorer: NoKG vs WithKG")
    parser.add_argument("--input",   default="summaries.json",        help="Input JSONL or JSON file")
    parser.add_argument("--output",  default="./results/questeval_results.json",   help="Output JSON file")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS,      help="Parallel workers")
    parser.add_argument("--seed",    type=int, default=RANDOM_SEED,      help="Random seed")
    args = parser.parse_args()

    logger.info(f"Loading samples from {args.input}")
    with open(args.input) as f:
        content = f.read().strip()
        if content.startswith('['):
            samples = json.loads(content)
        else:
            samples = [json.loads(line) for line in content.split('\n') if line.strip()]
    logger.info(f"Loaded {len(samples)} samples")

    run_qeval_pipeline(
        samples=samples,
        seed=args.seed,
        max_workers=args.workers,
        output_file=args.output,
    )

    logger.info("QuestEval evaluation complete!")