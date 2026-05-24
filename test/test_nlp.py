"""Local + competition NLP scorer.

Originally written for the GCP Workbench layout with `/home/jupyter/{TEAM_TRACK}/`
paths. Refactored so it works locally too:

- Paths are resolved via env vars (or CLI flags) and fall back to the GCP layout
  only when nothing else is set.
- The equivalence model is lazily loaded the first time `get_evaluator()` runs,
  so `import test_nlp` doesn't crash when the weights are missing (e.g. local
  dev box without `nlp_eval_512.zip`).

Env overrides (each also has a CLI flag, see `--help`):

  NLP_DATA_DIR       directory containing nlp.jsonl + documents/   (default: GCP layout)
  NLP_RESULTS_DIR    directory to write nlp_results.json to       (default: GCP layout)
  NLP_MODEL_DIR      directory containing the AE model checkpoint (default: ./test/models/nlp_eval_512)
  NLP_SERVER_URL     model server URL                              (default: http://localhost:5004/nlp)
  NLP_LIMIT          run only the first N eval rows                (default: all)

Examples:

  # Run against a local repo layout
  NLP_DATA_DIR=$(pwd)/nlp/src NLP_RESULTS_DIR=$(pwd)/test/results python test/test_nlp.py

  # Score a saved preds file without spinning up the server
  python test/test_nlp.py --score-only preds.json --gt nlp/src/nlp.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import zipfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from string import printable
from time import sleep
from typing import Any

import requests
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from test_utils import batched
from tqdm import tqdm, trange
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

load_dotenv()
TEAM_NAME = os.getenv("TEAM_NAME")
TEAM_TRACK = os.getenv("TEAM_TRACK")

BATCH_SIZE = 4
RETRIEVAL_ONLY_SCORE = 0.4
MAX_CANDIDATE_TOKEN_LENGTH = 64
DEFAULT_MODEL_DIR = "./test/models/nlp_eval_512"
DEFAULT_SERVER_URL = "http://localhost:5004/nlp"


def _gcp_data_dir() -> Path | None:
    return Path(f"/home/jupyter/{TEAM_TRACK}/nlp") if TEAM_TRACK else None


def _gcp_results_dir() -> Path | None:
    return Path(f"/home/jupyter/{TEAM_NAME}") if TEAM_NAME else None


def resolve_data_dir(override: str | None = None) -> Path:
    if override:
        return Path(override).expanduser().resolve()
    env = os.getenv("NLP_DATA_DIR")
    if env:
        return Path(env).expanduser().resolve()
    gcp = _gcp_data_dir()
    if gcp:
        return gcp
    raise RuntimeError(
        "No data directory configured. Set NLP_DATA_DIR or TEAM_TRACK, or pass --data-dir."
    )


def resolve_results_dir(override: str | None = None) -> Path:
    if override:
        return Path(override).expanduser().resolve()
    env = os.getenv("NLP_RESULTS_DIR")
    if env:
        return Path(env).expanduser().resolve()
    gcp = _gcp_results_dir()
    if gcp:
        return gcp
    raise RuntimeError(
        "No results directory configured. Set NLP_RESULTS_DIR or TEAM_NAME, or pass --results-dir."
    )


def resolve_model_dir(override: str | None = None) -> Path:
    if override:
        return Path(override).expanduser().resolve()
    return Path(os.getenv("NLP_MODEL_DIR", DEFAULT_MODEL_DIR)).expanduser().resolve()


def resolve_server_url(override: str | None = None) -> str:
    return override or os.getenv("NLP_SERVER_URL", DEFAULT_SERVER_URL)


@dataclass
class AEResult:
    index: int
    score: float
    equivalent: bool
    prob_equivalent: float

    def to_dict(self) -> dict:
        return {
            "score": round(self.score, 4),
            "equivalent": self.equivalent,
            "prob_equivalent": round(self.prob_equivalent, 4),
        }


class AnswerEquivalenceEvaluator:
    """Wraps a fine-tuned encoder for answer-equivalence inference."""

    def __init__(
        self,
        model_path: str | Path,
        threshold: float = 0.5,
        device: str | None = None,
        max_length: int = 128,
    ):
        self.threshold = threshold
        self.max_length = max_length

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"Loading model from {model_path} on {self.device}")
        if not Path(model_path).exists():
            if not TEAM_TRACK:
                raise FileNotFoundError(
                    f"Model not found at {model_path} and TEAM_TRACK is unset, "
                    f"so the GCP fallback path is unavailable. "
                    f"Set NLP_MODEL_DIR or pass --model-dir."
                )
            logger.info(
                f"Model path {model_path} does not exist, copying from /home/jupyter/{TEAM_TRACK}/nlp/models"
            )
            existing_model_path = (
                Path(f"/home/jupyter/{TEAM_TRACK}/nlp/models")
                / f"{Path(model_path).name}.zip"
            )
            if not existing_model_path.exists():
                raise FileNotFoundError(
                    f"Model not found at {model_path} or {existing_model_path}"
                )
            with zipfile.ZipFile(existing_model_path, "r") as zip_ref:
                zip_ref.extractall(Path(model_path).parent)
            logger.info(f"Extracted model to {Path(model_path).parent}")

        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(model_path)
        ).to(self.device)
        self.model.eval()

        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model loaded: {n_params:,} parameters")

    def _format_input(self, question: str, reference: str, candidate: str) -> str:
        _printable = "".join(filter(lambda x: x in printable, candidate))
        tokens = self.tokenizer.tokenize(
            _printable, max_length=MAX_CANDIDATE_TOKEN_LENGTH, truncation=True
        )
        reconstructed_candidate = self.tokenizer.convert_tokens_to_string(tokens)
        return (
            f"Question: {question} "
            f"Reference: {reference} "
            f"Candidate: {reconstructed_candidate}"
        )

    @torch.no_grad()
    def batch_evaluate(
        self,
        data: list[tuple[list[str], list[str], str, str, str]],
        batch_size: int = 64,
    ) -> list[AEResult]:
        empty_str_results = []
        non_empty_indexed_triples = []

        for i, (docs, pred_docs, q, r, c) in enumerate(data):
            overlap_docs = len(set(docs).intersection(set(pred_docs))) >= 1
            if len(docs) == 0 and len(pred_docs) == 0 and r == "" and c == "":
                empty_str_results.append(
                    AEResult(index=i, score=1.0, equivalent=True, prob_equivalent=1.0)
                )
            elif (r == "" or c == "") and overlap_docs:
                _equivalent = r == c
                empty_str_results.append(
                    AEResult(
                        index=i,
                        score=1.0 if _equivalent else RETRIEVAL_ONLY_SCORE,
                        equivalent=_equivalent,
                        prob_equivalent=1.0 if _equivalent else 0.0,
                    )
                )
            elif overlap_docs:
                non_empty_indexed_triples.append((i, q, r, c))
            else:
                empty_str_results.append(
                    AEResult(index=i, score=0.0, equivalent=False, prob_equivalent=0.0)
                )

        texts = [
            (i, self._format_input(q, r, c)) for i, q, r, c in non_empty_indexed_triples
        ]
        all_results = []

        for i in trange(0, len(texts), batch_size):
            batch_indices, batch_texts = zip(*texts[i : i + batch_size])
            encoding = self.tokenizer(
                batch_texts,
                max_length=self.max_length,
                padding="longest",
                truncation=True,
                return_tensors="pt",
            ).to(self.device)

            logits = self.model(**encoding).logits
            probs = F.softmax(logits, dim=-1)

            for prob_idx, prob in enumerate(probs):
                prob_eq = prob[1].item()
                _equivalent = prob_eq >= self.threshold
                all_results.append(
                    AEResult(
                        index=batch_indices[prob_idx],
                        score=1.0 if _equivalent else RETRIEVAL_ONLY_SCORE,
                        equivalent=_equivalent,
                        prob_equivalent=prob_eq,
                    )
                )

        all_results.extend(empty_str_results)
        all_results.sort(key=lambda r: r.index)
        return all_results

    def aggregate_score(self, results: list[AEResult]) -> dict:
        n = len(results)
        if n == 0:
            return {
                "n": 0,
                "equiv_rate": 0.0,
                "mean_prob": 0.0,
                "equivalent_count": 0,
                "not_equivalent_count": 0,
            }
        equiv_count = sum(r.score for r in results)
        mean_prob = sum(r.prob_equivalent for r in results) / n
        return {
            "n": n,
            "equiv_rate": round(equiv_count / n, 3),
            "mean_prob": round(mean_prob, 3),
            "equivalent_count": equiv_count,
            "not_equivalent_count": n - equiv_count,
        }


_EVAL_SINGLETON: AnswerEquivalenceEvaluator | None = None


def get_evaluator(model_dir: str | Path | None = None) -> AnswerEquivalenceEvaluator:
    """Lazily build (and cache) the equivalence model. Caller passes a path,
    or we fall back to NLP_MODEL_DIR / the GCP default."""
    global _EVAL_SINGLETON
    if _EVAL_SINGLETON is not None:
        return _EVAL_SINGLETON
    path = Path(model_dir) if model_dir else resolve_model_dir()
    _EVAL_SINGLETON = AnswerEquivalenceEvaluator(
        model_path=path, threshold=0.9, device=None, max_length=512
    )
    return _EVAL_SINGLETON


def poll_endpoint_for_loading(server_url: str, max_retries=None, delay_sec=10):
    retry_num = 0
    while max_retries is None or retry_num < max_retries:
        try:
            response = requests.post(
                server_url, data=json.dumps({"instances": [{"poll": "true"}]})
            ).json()["predictions"]
            if len(response) == 1 and response[0].get("status") == "loaded":
                print("Model server is loaded.")
                return True
            elif len(response) == 1 and response[0].get("status") == "error":
                print("Model server is reporting an error.")
                return False
            elif len(response) == 1 and response[0].get("status") == "loading":
                print(f"Retry {retry_num}: Model server is still loading the corpus.")
        except Exception as e:
            print(f"Error occurred while polling endpoint: {e}")
        sleep(delay_sec)
        retry_num += 1
    return False


def sample_generator(
    instances: Sequence[Mapping[str, Any]],
) -> Iterator[Mapping[str, Any]]:
    for instance in instances:
        yield {"question": instance["question"]}


def score_nlp(
    preds: Sequence[dict[str, list[str] | str]],
    ground_truth: Sequence[Mapping[str, Any]],
    model_dir: str | Path | None = None,
) -> float:
    data = []
    for pred, gt in zip(preds, ground_truth):
        data.append(
            (
                gt["source_docs"],
                pred["documents"][:3],
                gt["question"],
                gt["answer"] if gt["answer"] is not None else "",
                pred["answer"],
            )
        )
    evaluator = get_evaluator(model_dir)
    results = evaluator.batch_evaluate(data)
    summary = evaluator.aggregate_score(results)
    logger.info(f"Answer Equivalence Evaluation Summary: {summary}")
    return summary["equiv_rate"]


def _load_eval(path: Path, limit: int | None) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if limit:
        rows = rows[:limit]
    return rows


def _load_documents(documents_dir: Path) -> list[dict[str, str]]:
    doc_contents = []
    for doc_file in sorted(documents_dir.glob("*.txt")):
        doc_contents.append(
            {"id": doc_file.stem, "document": doc_file.read_text()}
        )
    return doc_contents


def _query_server(
    server_url: str, instances: list[dict], batch_size: int
) -> list[dict]:
    results: list[dict] = []
    batch_generator = batched(sample_generator(instances), n=batch_size)
    for batch in tqdm(batch_generator, total=math.ceil(len(instances) / batch_size)):
        response = requests.post(
            server_url, data=json.dumps({"instances": list(batch)})
        )
        results.extend(response.json()["predictions"])
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", help="Override NLP_DATA_DIR")
    parser.add_argument("--results-dir", help="Override NLP_RESULTS_DIR")
    parser.add_argument("--model-dir", help="Override NLP_MODEL_DIR")
    parser.add_argument("--server-url", help="Override NLP_SERVER_URL")
    parser.add_argument(
        "--limit",
        type=int,
        default=int(os.getenv("NLP_LIMIT", "0")) or None,
        help="Run only the first N eval rows",
    )
    parser.add_argument(
        "--skip-load",
        action="store_true",
        help="Assume the server already has the corpus loaded.",
    )
    parser.add_argument(
        "--score-only",
        metavar="PREDS_JSON",
        help="Skip the server entirely; just score this predictions file against --gt.",
    )
    parser.add_argument("--gt", help="Ground-truth jsonl (used with --score-only)")
    args = parser.parse_args()

    if args.score_only:
        gt_path = Path(args.gt) if args.gt else resolve_data_dir(args.data_dir) / "nlp.jsonl"
        instances = _load_eval(gt_path, args.limit)
        preds = json.loads(Path(args.score_only).read_text())
        if args.limit:
            preds = preds[: args.limit]
        score = score_nlp(preds, instances, model_dir=args.model_dir)
        print("NLP RAG QA Accuracy:", score)
        return

    data_dir = resolve_data_dir(args.data_dir)
    results_dir = resolve_results_dir(args.results_dir)
    server_url = resolve_server_url(args.server_url)
    results_dir.mkdir(parents=True, exist_ok=True)

    instances = _load_eval(data_dir / "nlp.jsonl", args.limit)
    documents = _load_documents(data_dir / "documents")

    if not args.skip_load:
        response = requests.post(
            server_url,
            data=json.dumps({"instances": [{"documents": documents}]}),
        )
        if (
            response.status_code != 200
            or response.json()["predictions"][0].get("status") == "error"
        ):
            logger.error(f"Failed to load corpus: {response.text}")
            return
        logger.info("Corpus load initiated, polling for completion...")
        if not poll_endpoint_for_loading(server_url, max_retries=30, delay_sec=10):
            logger.error("Corpus failed to load within expected time.")
            return

    results = _query_server(server_url, instances, BATCH_SIZE)

    results_path = results_dir / "nlp_results.json"
    print(f"Saving test results to {results_path}")
    results_path.write_text(json.dumps(results))

    score = score_nlp(results, instances, model_dir=args.model_dir)
    print("NLP RAG QA Accuracy:", score)


if __name__ == "__main__":
    main()
