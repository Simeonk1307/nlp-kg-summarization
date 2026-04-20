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

def score_against_reference(triples, reference, summary):
    prompt = f"""You are a summarization evaluator.

KG Triples:
{format_triples(triples)}

Reference Summary:
{reference}

Generated Summary:
{summary}

Rate 1-5 only. Respond ONLY in JSON with no extra text:
{{"faithfulness":0,"coverage":0,"reference_alignment":0,"coherence":0,"hallucination":0,"overall":0,"reasoning":""}}"""

    return call_judge(prompt)

def run_pipeline1(samples):
    dims = ["faithfulness", "coverage", "reference_alignment", "coherence", "hallucination", "overall"]
    all_scores_no_kg = []
    all_scores_with_kg = []

    for i, s in enumerate(samples):
        print(f"Sample {i+1}/{len(samples)}")
        score_no_kg = score_against_reference(s["triples"], s["reference"], s["output_no_kg"])
        score_with_kg = score_against_reference(s["triples"], s["reference"], s["output_with_kg"])
        all_scores_no_kg.append(score_no_kg)
        all_scores_with_kg.append(score_with_kg)

    n = len(samples)
    print(f"\n{'Dimension':<25} {'NoKG':>8} {'WithKG':>8}")
    print("-" * 43)
    for d in dims:
        avg_no = sum(r[d] for r in all_scores_no_kg) / n
        avg_with = sum(r[d] for r in all_scores_with_kg) / n
        print(f"{d:<25} {avg_no:>8.2f} {avg_with:>8.2f}")

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