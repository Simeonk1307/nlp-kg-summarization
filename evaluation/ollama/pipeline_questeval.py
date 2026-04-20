import json
import re
import requests
from collections import defaultdict
import time

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "mistral"

def call_judge(prompt):
    response = requests.post(OLLAMA_URL, json={
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0}
    })
    text = response.json()["response"].strip()
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        return json.loads(match.group())
    raise ValueError(f"No JSON found: {text}")

def format_triples(triples):
    return "\n".join([f"({h}, {r}, {t})" for h, r, t in triples])

def generate_questions(triples, reference):
    prompt = f"""You are a QA generation expert.

KG Triples:
{format_triples(triples)}

Reference Summary:
{reference}

Generate 5 factual questions that can be answered from the triples and reference above.
Each question must have a clear short answer from the triples.
Respond ONLY in JSON with no extra text:
{{"questions":[{{"q":"","a":""}}]}}"""

    return call_judge(prompt)["questions"]

def answer_from_summary(summary, questions):
    questions_text = "\n".join([f"{i+1}. {q['q']}" for i, q in enumerate(questions)])
    prompt = f"""You are a QA system. Answer each question using ONLY the summary below. If not answerable, say "unanswerable".

Summary:
{summary}

Questions:
{questions_text}

Respond ONLY in JSON with no extra text:
{{"answers":[""]}}"""

    return call_judge(prompt)["answers"]

def score_answers(questions, predicted_answers):
    qa_pairs = "\n".join([
        f"Q: {q['q']}\nExpected: {q['a']}\nPredicted: {pred}"
        for q, pred in zip(questions, predicted_answers)
    ])
    prompt = f"""You are an answer evaluator. For each QA pair, score 1 if predicted answer is correct or semantically equivalent to expected, else 0.

{qa_pairs}

Respond ONLY in JSON with no extra text:
{{"scores":[0]}}"""

    return call_judge(prompt)["scores"]

def questeval_sample(triples, reference, output_no_kg, output_with_kg):
    questions = generate_questions(triples, reference)

    answers_no_kg = answer_from_summary(output_no_kg, questions)
    answers_with_kg = answer_from_summary(output_with_kg, questions)

    scores_no_kg = score_answers(questions, answers_no_kg)
    scores_with_kg = score_answers(questions, answers_with_kg)

    factuality_no_kg = sum(scores_no_kg) / len(scores_no_kg)
    factuality_with_kg = sum(scores_with_kg) / len(scores_with_kg)

    return {
        "questions": questions,
        "no_kg": {"answers": answers_no_kg, "scores": scores_no_kg, "factuality": factuality_no_kg},
        "with_kg": {"answers": answers_with_kg, "scores": scores_with_kg, "factuality": factuality_with_kg}
    }

def run_pipeline3(samples):
    all_results = []
    total_no_kg = 0
    total_with_kg = 0

    for i, s in enumerate(samples):
        print(f"Sample {i+1}/{len(samples)}")
        result = questeval_sample(s["triples"], s["reference"], s["output_no_kg"], s["output_with_kg"])
        all_results.append(result)
        total_no_kg += result["no_kg"]["factuality"]
        total_with_kg += result["with_kg"]["factuality"]

    n = len(samples)
    print(f"\nQuestEval Factuality Scores (avg over {n} samples):")
    print(f"  NoKG   : {total_no_kg/n:.3f}")
    print(f"  WithKG : {total_with_kg/n:.3f}")

    return all_results


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
results = run_pipeline3(samples)
end = time.time()

print(end-start)