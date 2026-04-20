import json
import re
import random
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

def pairwise_judge(triples, reference, summary_a, summary_b, label_a, label_b):
    prompt = f"""You are a summarization evaluator.

KG Triples:
{format_triples(triples)}

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
    winner_label = label_a if result["winner"] == "A" else (label_b if result["winner"] == "B" else "tie")
    return {"winner": winner_label, "reasoning": result["reasoning"]}

def run_pipeline2(samples, seed=42):
    random.seed(seed)
    counts = defaultdict(int)
    all_results = []

    for i, s in enumerate(samples):
        print(f"Sample {i+1}/{len(samples)}")
        flip = random.random() > 0.5

        if flip:
            summary_a, label_a = s["output_with_kg"], "with_kg"
            summary_b, label_b = s["output_no_kg"], "no_kg"
        else:
            summary_a, label_a = s["output_no_kg"], "no_kg"
            summary_b, label_b = s["output_with_kg"], "with_kg"

        result = pairwise_judge(s["triples"], s["reference"], summary_a, summary_b, label_a, label_b)
        counts[result["winner"]] += 1
        all_results.append({"sample": i+1, "flipped": flip, **result})

    n = len(samples)
    print(f"\nA/B Results (n={n}):")
    print(f"  NoKG wins   : {counts['no_kg']} ({100*counts['no_kg']/n:.1f}%)")
    print(f"  WithKG wins : {counts['with_kg']} ({100*counts['with_kg']/n:.1f}%)")
    print(f"  Ties        : {counts['tie']} ({100*counts['tie']/n:.1f}%)")

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
results = run_pipeline2(samples)
end = time.time()

print(end-start)