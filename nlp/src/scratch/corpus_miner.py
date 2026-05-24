"""Corpus pattern miner — frequency analysis of high-signal regex patterns
across all 296 documents. Drives Phase 2 bucket regex refinement.

Stdlib only. Outputs to `nlp/src/scratch/miner_report.md`.

What it counts:
- Money / percent / date / codename / designation / count patterns (per current
  regex catalog), plus candidate variants that might be more precise.
- Top-N most-frequent capitalised entity phrases (proxy for high-signal tokens).
- Top-N high-IDF content n-grams (bigrams + trigrams).
- Section headers (markdown-style + ALL-CAPS labels).
- Pre-answer structural cues (e.g. lines ending in ":").

Run: python corpus_miner.py
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
DOCS = sorted((HERE / "documents").glob("DOC-*.txt"))
OUT = Path(__file__).resolve().parent / "miner_report.md"

# --- candidate patterns (existing + alternative formulations) ---------------

CANDIDATES: dict[str, list[tuple[str, str]]] = {
    "money": [
        ("strict_credits",
         r"\d[\d,]*(?:\.\d+)?\s*(?:million|billion|thousand|trillion)?\s*(?:Phi\s+)?Credits?"),
        ("verbose_credits",
         r"(?:approximately\s+|approx\.\s+|over\s+|nearly\s+|about\s+)?"
         r"\d[\d,]*(?:\.\d+)?\s*"
         r"(?:million|billion|thousand|trillion)?\s*"
         r"(?:Phi\s+)?Credits?"),
        ("dollar_amount",
         r"\$\d[\d,]*(?:\.\d+)?\s*(?:million|billion|thousand|trillion)?"),
        ("financial_penalty_label",
         r"(?i)(?:financial\s+penalty|monetary\s+penalty|fine\s+of|funding\s+of|"
         r"budget\s+of|cost\s+of|revenue\s+of|payment\s+of)[^\n]{0,80}"),
    ],
    "percent": [
        ("digit_pct", r"\d+(?:\.\d+)?\s*%"),
        ("digit_word_percent",
         r"(?:approximately\s+|about\s+|nearly\s+|over\s+)?"
         r"\d+(?:\.\d+)?\s*(?:percent(?:age)?(?:\s+points?)?|%)"),
        ("share_of", r"(?i)\b\d+(?:\.\d+)?\s*%?\s+(?:share|fraction|proportion)\b"),
    ],
    "date_pce": [
        ("iso_pce", r"\d{2,4}-\d{2}-\d{2}(?:\s+PCE)?"),
        ("quarter_pce", r"Q[1-4]\s+\d{2,4}\s+PCE"),
        ("year_pce", r"\d{2,4}\s+PCE"),
        ("year_ce", r"\d{4}\s+CE"),
        ("year_dual", r"\d{2,4}\s+PCE\s*\(\d{4}\s+CE\)"),
        ("hour_time", r"\b\d{1,2}:\d{2}(?:\s+(?:AM|PM|hrs))?\b"),
    ],
    "codename": [
        ("all_caps_word", r"\b[A-Z]{4,}\b"),
        ("designation", r"\b[A-Z]{2,}-\d+(?:-\d+)*\b"),
        ("title_case_list",
         r"\b[A-Z][a-z]{3,}(?:,\s+[A-Z][a-z]{3,}){1,4}(?:,?\s+and\s+[A-Z][a-z]{3,})?"),
        ("seastitch_like", r"\b[A-Z]{6,}\b"),
        ("quoted_string", r"\"([^\"]{3,40})\""),
    ],
    "count": [
        ("plain_int", r"\b\d+\b"),
        ("number_word",
         r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
         r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|"
         r"thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million)\b"),
        ("approximately_n",
         r"(?:approximately\s+|approx\.\s+|about\s+|nearly\s+|over\s+)?"
         r"\d[\d,]*(?:\.\d+)?\s*(?:million|billion|thousand|trillion)?"),
        ("every_n_period",
         r"(?i)\bevery\s+\d+\s+(?:year|month|week|day|hour|minute|second)s?\b"),
    ],
    "entity": [
        ("title_phrase", r"\b(?:[A-Z][a-z]+(?:\s+(?:[A-Z][a-z]+|of|the|and)){1,4})\b"),
        ("doctoral_credential",
         r"(?i)(?:doctoral|doctorate|PhD|master'?s|bachelor'?s)\s+(?:degree\s+)?"
         r"(?:in|of)?\s*[a-z\s-]{0,40}"),
    ],
    "structural": [
        ("md_header", r"^#+\s+.+$"),
        ("md_bold_label", r"\*\*[A-Z][A-Za-z ]{2,40}:\*\*"),
        ("caps_label", r"^[A-Z][A-Z ]{4,40}:"),
        ("colon_line", r"^[A-Za-z][^\n]{0,60}:"),
        ("bullet", r"^\s*[-*]\s+"),
        ("classification",
         r"(?i)classification:\s*(?:open|restricted|confidential|secret|"
         r"l[1-5]|public)"),
    ],
}

STOPWORDS = frozenset((
    "the a an of and or to in for on with by is are was were be as at from "
    "that this it its their there they them which who whom whose what when "
    "where why how been being have has had do does did but if not no nor so "
    "than then into about over under between within against during through "
    "before after above below up down out off again further once each any "
    "all some most more many much few several other another such own same "
    "very can could should would may might must will shall did done").split())

RE_WORD = re.compile(r"\b[A-Za-z][A-Za-z'-]+\b")
RE_CAPS_PHRASE = re.compile(r"\b(?:[A-Z][A-Za-z]+)(?:\s+(?:[A-Z][A-Za-z]+|of|the|and)){0,3}\b")


def load_docs() -> list[tuple[str, str]]:
    return [(p.stem, p.read_text(encoding="utf-8")) for p in DOCS]


def count_patterns(docs: list[tuple[str, str]]) -> dict:
    out: dict[str, list[dict]] = {}
    for cat, pats in CANDIDATES.items():
        out[cat] = []
        for name, pat in pats:
            try:
                rx = re.compile(pat, re.MULTILINE)
            except re.error as e:
                out[cat].append({"name": name, "error": str(e)})
                continue
            total = 0
            doc_hits = 0
            sample: list[str] = []
            for _, text in docs:
                ms = rx.findall(text)
                if ms:
                    doc_hits += 1
                    total += len(ms)
                    if len(sample) < 5:
                        # findall returns either str or tuple of groups
                        for m in ms[:2]:
                            if isinstance(m, tuple):
                                m = next((x for x in m if x), "") or ""
                            if m and m not in sample:
                                sample.append(m if len(m) <= 80 else m[:77] + "...")
                                if len(sample) >= 5:
                                    break
            out[cat].append({
                "name": name,
                "regex": pat,
                "total_matches": total,
                "docs_with_match": doc_hits,
                "docs_with_match_pct": round(doc_hits / len(docs), 3),
                "matches_per_doc": round(total / max(1, doc_hits), 2),
                "samples": sample,
            })
    return out


def top_entities(docs: list[tuple[str, str]], k: int = 60) -> list[tuple[str, int]]:
    c: Counter[str] = Counter()
    for _, text in docs:
        for m in RE_CAPS_PHRASE.findall(text):
            m = m.strip()
            if not m:
                continue
            words = m.split()
            if all(w.lower() in STOPWORDS for w in words):
                continue
            c[m] += 1
    return c.most_common(k)


def top_ngrams(docs: list[tuple[str, str]], n: int, k: int = 50,
               min_df: int = 5, max_df_frac: float = 0.5
               ) -> list[tuple[str, int, int]]:
    N = len(docs)
    df: Counter[str] = Counter()
    tf: Counter[str] = Counter()
    for _, text in docs:
        words = [w.lower() for w in RE_WORD.findall(text)]
        words = [w for w in words if w not in STOPWORDS and len(w) >= 3]
        seen: set[str] = set()
        for i in range(len(words) - n + 1):
            g = " ".join(words[i:i + n])
            tf[g] += 1
            seen.add(g)
        for g in seen:
            df[g] += 1
    out: list[tuple[str, int, int]] = []
    max_df = int(N * max_df_frac)
    for g, c in tf.most_common():
        d = df[g]
        if d < min_df or d > max_df:
            continue
        out.append((g, c, d))
        if len(out) >= k:
            break
    return out


def md_table(rows: list[dict], cols: list[str]) -> str:
    lines = ["| " + " | ".join(cols) + " |"]
    lines.append("|" + "|".join("---" for _ in cols) + "|")
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c, "")
            if isinstance(v, list):
                v = "; ".join(str(x) for x in v[:3])
            v = str(v).replace("\n", " ").replace("|", "\\|")
            if len(v) > 80:
                v = v[:77] + "..."
            cells.append(v)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> int:
    docs = load_docs()
    print(f"Loaded {len(docs)} docs.")

    rep = ["# Corpus Pattern Miner Report\n",
           f"Docs analysed: **{len(docs)}**\n"]

    pat_stats = count_patterns(docs)
    for cat, rows in pat_stats.items():
        rep.append(f"\n## {cat}\n")
        rep.append(md_table(rows,
                            ["name", "total_matches", "docs_with_match",
                             "docs_with_match_pct", "matches_per_doc", "samples"]))
        rep.append("")

    rep.append("\n## Top capitalised entity phrases (top 60)\n")
    ents = top_entities(docs)
    rep.append("```")
    for e, c in ents:
        rep.append(f"  {c:5}  {e}")
    rep.append("```")

    rep.append("\n## Top content bigrams (df 5..150)\n```")
    for g, tf, df in top_ngrams(docs, 2, 40, min_df=5, max_df_frac=0.5):
        rep.append(f"  tf={tf:5} df={df:3}  {g}")
    rep.append("```\n")

    rep.append("\n## Top content trigrams (df 3..80)\n```")
    for g, tf, df in top_ngrams(docs, 3, 30, min_df=3, max_df_frac=0.27):
        rep.append(f"  tf={tf:5} df={df:3}  {g}")
    rep.append("```\n")

    OUT.write_text("\n".join(rep))
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
