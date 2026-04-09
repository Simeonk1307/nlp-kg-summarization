"""
Knowledge Graph Triples Extractor Helper

We use REBEL from hugging face (https://huggingface.co/Babelscape/rebel-large)

REBEL is a seq2seq model. We feed it the raw text and it outputs
a special markup string like:
<triplet> Fed <subj> raised <rel> interest rates <obj>
We then parse that markup into clean tuples.
"""

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import re
from typing import List, Tuple

# Triple is an alias for 3 variable tuple (head enitity,relation,tail entity)
Triple = Tuple[str, str, str]


class KGExtractor:

    # REBEL is ~1.6GB
    model_name = "Babelscape/rebel-large"

    def __init__(self, device: str = "cpu"):
        """
        Args:
            device: "cuda", "cpu", or None (auto-detect)
        """

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Loading REBEL extractor on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.model.eval()
        self.model.to(self.device)
        print("REBEL loaded")

    def extract(
        self,
        text: str,
        max_triples: int = 30,
        option: str | None = None,
        deduplicate: bool = False,
    ):
        """
        Main function to extract triples from a single string.

        Args:
            text: Input text. Can be a sentence or a paragraph. Long texts are chunked automatically.
            max_triples: Upper cap on how many triples to return.
            option:
                'str' - output is list[str]
                'dict' - output is list[dict]
                 None - output is list[Triple]
            deduplicate:
                whether to dedeuplicate the triples or not
                It is expsensive hence the default is False
        """

        # REBEL has a 512-token limit, so we chunk long texts
        chunks = self._chunk_text(text)

        all_triples: List[Triple] = []
        for chunk in chunks:
            raw_output = self._run_model(chunk)
            triples = self._parse_rebel_output(raw_output, option)
            all_triples.extend(triples)

        if not deduplicate:
            return all_triples[:max_triples]

        def normalize_triplet(t):
            if isinstance(t, tuple):
                return t

            elif isinstance(t, dict):
                return (t["head"], t["type"], t["tail"])

            elif isinstance(t, str):
                parts = t.split()
                if len(parts) < 3:
                    return None
                # parts[0] is subject, parts[-1] is obj and remaining are relations
                return (parts[0], " ".join(parts[1:-1]), parts[-1])

            else:
                return None

        # Naive Deduplication of triples done here
        seen = set()
        deduped = []

        for t in all_triples:
            norm = normalize_triplet(t)
            if not norm:
                continue

            key = tuple(x.lower() for x in norm)

            if key not in seen:
                seen.add(key)
                deduped.append(t)

        return deduped[:max_triples]

    def _run_model(self, text: str) -> str:
        """
        Run REBEL on a single chunk, return raw output string
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=256,
                num_beams=3,
            )

        # Decode the first output token ID back to a string
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=False)

    def _chunk_text(self, text: str, chunk_size: int = 512) -> List[str]:
        """
        Split text into chunks of ~chunk_size characters, splitting on
        sentence boundaries / full stops (.) wherever possible.
        """
        if len(text) <= chunk_size:
            return [text]

        # In this regex ?<= is look behind assertion opeator
        # Here it splits at any of the puncutation [.!?] if the
        # puncutation is followed by atleast one space character
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks = []
        current = ""

        for sent in sentences:
            if len(current) + len(sent) < chunk_size:
                current += " " + sent
            else:
                if current:
                    chunks.append(current.strip())
                current = sent

        if current:
            chunks.append(current.strip())

        return chunks if chunks else [text]

    def _parse_rebel_output(self, text: str, option: str | None = None):
        """
        Function taken from hugging face https://huggingface.co/Babelscape/rebel-large
        and modified
        Args:
            text: decoded text from REBEL
            option:
                'str' - output is list[str]
                'dict' - output is list[dict]
                None - output is list[Triple]
        """

        def format_triplet(subject, relation, object_):
            subject, relation, object_ = (
                subject.strip(),
                relation.strip(),
                object_.strip(),
            )

            if not (subject and relation and object_):
                return None

            if option == "dict":
                return {"head": subject, "type": relation, "tail": object_}
            elif option == "str":
                return f"{subject} {relation} {object_}"
            elif option is None:
                return (subject, relation, object_)
            else:
                raise KeyError(
                    f"Wrong option {option}! should be 'str', 'dict' or None"
                )

        triplets = []
        subject, relation, object_ = "", "", ""
        text = text.strip()
        current = None

        tokens = (
            text.replace("<s>", "").replace("</s>", "").replace("<pad>", "").split()
        )

        for token in tokens:
            if token == "<triplet>":
                triplet = format_triplet(subject, relation, object_)
                if triplet:
                    triplets.append(triplet)

                subject, relation, object_ = "", "", ""
                current = "subject"

            elif token == "<subj>":
                triplet = format_triplet(subject, relation, object_)
                if triplet:
                    triplets.append(triplet)

                object_ = ""
                current = "object"

            elif token == "<obj>":
                relation = ""
                current = "relation"

            else:
                if current == "subject":
                    subject += " " + token
                elif current == "object":
                    object_ += " " + token
                elif current == "relation":
                    relation += " " + token

        triplet = format_triplet(subject, relation, object_)
        if triplet:
            triplets.append(triplet)

        return triplets


# Example use case
if __name__ == "__main__":
    extractor = KGExtractor()

    article1 = """
    The Federal Reserve raised interest rates by 0.25 percentage points on Wednesday,
    the tenth consecutive increase, as it continued its battle against inflation.
    Fed Chair Jerome Powell said the committee remained committed to bringing inflation
    down to its 2% target. Markets fell sharply following the announcement,
    with the S&P 500 dropping 1.8%.
    """

    article2 = """
    Javokhir Sindarov missed a clear opportunity to secure victory this afternoon against Matthias Bluebaum, allowing Anish Giri to narrow the gap 
    after defeating Fabiano Caruana with the black pieces. With five rounds remaining, Sindarov still leads by 1.5 points.Vaishali and Zhu Jiner 
    share the lead at the FIDE Women's Candidates with 5.5/9, following victories over Divya Deshmukh and Kateryna Lagno, respectively. 
    Meanwhile, Anna Muzychuk squandered a highly promising endgame, slipping behind the leaders.
    Let's take a closer look at how the afternoon unfolded. The ceremonial opening move was played by Paris Klerides, General Secretary of 
    the Cyprus Chess Federation and FIDE Delegate for Cyprus, who made the symbolic 1.e4 on behalf of Matthias Bluebaum. However, 
    Bluebaum opted for 1.d4 instead. Javokhir Sindarov replied with a very rare line 
    the Harrwitz Attack in the Queen's Gambit Decline
    """

    triples = extractor.extract(article2, 30)
    print("\nExtracted triples:")
    print(triples)
