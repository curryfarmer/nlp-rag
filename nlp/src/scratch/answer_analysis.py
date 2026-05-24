#!/usr/bin/env python3
"""
Deep characterization of answers and source documents for Phase 2 extraction design.

Analyzes dev split (706 questions) to characterize:
1. Verbatim-presence breakdown (exact match, all words present, 70%+ words, derivation-only)
2. Answer shape histogram (number, number+unit, date, codename, proper noun, sentence, multi-clause)
3. Question-type bucket assignments (mutually exclusive)
4. Per-bucket verbatim-extractability
5. L2-specific breakdown (arithmetic_subtract, arithmetic_divide, count_aggregate, comparison, multi_fact)
"""

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


BASE_DIR = Path(__file__).parent.parent
DOCS_DIR = BASE_DIR / "documents"
NLPJSONL_PATH = BASE_DIR / "nlp.jsonl"
EVAL_SPLIT_PATH = BASE_DIR / "eval_split.json"


@dataclass
class Row:
    key: int
    difficulty: str  # "L1", "L2", etc.
    question: str
    answer: str
    source_docs: list[str]
    answerable: bool


def load_data() -> tuple[list[Row], set[int]]:
    """Load nlp.jsonl and eval_split.json; return (all_rows, dev_indices)."""
    rows = []
    with open(NLPJSONL_PATH) as f:
        for line in f:
            data = json.loads(line)
            rows.append(Row(
                key=data["key"],
                difficulty=data["difficulty"],
                question=data["question"],
                answer=data["answer"],
                source_docs=data["source_docs"],
                answerable=data["answerable"],
            ))
    
    with open(EVAL_SPLIT_PATH) as f:
        split = json.load(f)
        dev_indices = set(split["dev"])
    
    return rows, dev_indices


def load_doc(doc_id: str) -> str:
    """Load document text from documents/ dir."""
    path = DOCS_DIR / f"{doc_id}.txt"
    if not path.exists():
        return ""
    return path.read_text(errors="replace")


def verbatim_check(gold: str, doc_text: str) -> dict:
    """
    Breakdown of verbatim presence:
    - 'exact': case-insensitive substring match
    - 'all_words': all gold words appear in doc (any order, case-insensitive)
    - 'pct_70': >= 70% of gold words present
    - 'none': none of the above (derivation-only)
    
    Returns: {"type": "exact" | "all_words" | "pct_70" | "none", ...}
    """
    doc_lower = doc_text.lower()
    gold_lower = gold.lower()
    
    # 1. Exact substring match
    if gold_lower in doc_lower:
        return {"type": "exact", "gold": gold, "in_doc": gold_lower}
    
    # 2. All gold words present (extract content words)
    gold_words = set(re.findall(r"\b[a-z0-9]+\b", gold_lower))
    # Filter out common stopwords
    stopwords = {
        "the", "a", "an", "and", "or", "of", "to", "in", "for", "is", "are",
        "was", "were", "be", "been", "as", "at", "by", "on", "with", "from",
        "that", "this", "it", "its", "than", "more", "less", "other"
    }
    gold_words_filtered = gold_words - stopwords
    
    if not gold_words_filtered:
        # All-stopword answer (rare); treat as present if any word matches
        return {"type": "all_words", "gold": gold}
    
    doc_words = set(re.findall(r"\b[a-z0-9]+\b", doc_lower))
    gold_present = gold_words_filtered & doc_words
    pct = len(gold_present) / len(gold_words_filtered) if gold_words_filtered else 0
    
    if pct == 1.0:
        return {"type": "all_words", "gold": gold, "pct": 1.0}
    elif pct >= 0.7:
        return {"type": "pct_70", "gold": gold, "pct": pct}
    else:
        return {"type": "none", "gold": gold, "pct": pct}


def classify_answer_shape(answer: str) -> str:
    """
    Classify answer into one of:
    - "pure_number": \d+ optionally with unit (e.g., "123", "45 billion")
    - "date_pce": Q[1-4] \d{2,4} PCE or \d{2,4}-\d{2}-\d{2} or \d{2,4} PCE
    - "codename": all-caps with hyphens (e.g., "PROJECT-NEXUS")
    - "proper_noun": capitalized phrase(s)
    - "sentence": normal sentence (subject + verb, period/no period)
    - "multi_clause": 2+ independent/dependent clauses
    """
    answer_clean = answer.strip()
    
    # Pure number / number with unit
    if re.match(r"^\d+(\.\d+)?\s*(million|billion|thousand|credits|phi|dollars|years|percent|%)?$", answer_clean, re.I):
        return "pure_number"
    
    # Date patterns
    if re.search(r"Q[1-4]\s+\d{2,4}\s+PCE|\d{2,4}-\d{2}-\d{2}|\d{2,4}\s+PCE", answer_clean):
        return "date_pce"
    
    # Codename (all caps with hyphens)
    if re.match(r"^[A-Z][A-Z0-9-]*$", answer_clean) and "-" in answer_clean:
        return "codename"
    
    # All-caps single word
    if re.match(r"^[A-Z]+$", answer_clean):
        return "codename"
    
    # Proper noun phrase (starts with cap, may have multiple words)
    if re.match(r"^[A-Z][a-zA-Z\s'-]*$", answer_clean):
        if " " not in answer_clean:
            return "proper_noun"
        # Multi-word: check if it's mostly capitalized (entity name)
        cap_words = sum(1 for w in answer_clean.split() if w[0].isupper())
        if cap_words >= len(answer_clean.split()) - 1:
            return "proper_noun"
    
    # Multi-clause: look for conjunctions at top level
    # Rough heuristic: contains "and" or "but" or semicolon, and seems like full statement
    if re.search(r"\band\b.*\band\b|\bbut\b|;", answer_clean, re.I):
        return "multi_clause"
    
    # Sentence: contains verb-like structure or ends with period/question mark
    if re.search(r"\b(is|are|was|were|be|been|have|has|do|does|did|can|could|will|would|should|may|might)\b", answer_clean, re.I):
        return "sentence"
    
    # Default to sentence if length > 4 words
    if len(answer_clean.split()) > 4:
        return "sentence"
    
    return "proper_noun"


def question_to_bucket(question: str) -> str:
    """
    Classify question into EXACTLY ONE bucket (first-match wins):
    - money: Credits, Phi, dollar, penalty/cost/revenue/fine/budget/funding, "how much"
    - percent: %, percentage, percent, share of
    - date_pce: when, deadline, "on what date", "in what year"
    - years_delta: "how many years", "how long after/between", interval
    - count: "how many" (not years), "how many of"
    - codename: codename, designation, "code name", "name of the program"
    - entity_who: who, "by whom", "which person", "which official"
    - entity_what: "which company", "which division", "which sector", "what industry"
    - proportion_other: "what proportion", "what fraction", "what ratio"
    - fallback: none of above
    """
    q_lower = question.lower()
    
    # money
    if any(x in q_lower for x in ["credits", "phi credits", "phi credit", "dollar"]):
        return "money"
    if any(x in q_lower for x in ["penalty", "cost", "revenue", "fine", "budget", "funding"]):
        return "money"
    if "how much" in q_lower:
        return "money"
    
    # percent
    if any(x in q_lower for x in ["%", "percentage", "percent", "share of"]):
        return "percent"
    
    # date_pce
    if any(x in q_lower for x in ["when", "deadline", "by what deadline", "on what date", "in what year"]):
        return "date_pce"
    
    # years_delta
    if any(x in q_lower for x in ["how many years", "how long after", "how long between", "interval"]):
        return "years_delta"
    
    # count
    if "how many of" in q_lower:
        return "count"
    if "how many" in q_lower:
        return "count"
    
    # codename
    if any(x in q_lower for x in ["codename", "designation", "code name", "name of the program"]):
        return "codename"
    
    # entity_who
    if any(x in q_lower for x in ["who ", "by whom", "which person", "which official"]):
        return "entity_who"
    
    # entity_what
    if any(x in q_lower for x in ["which company", "which division", "which sector", "what industry"]):
        return "entity_what"
    
    # proportion_other
    if any(x in q_lower for x in ["what proportion", "what fraction", "what ratio"]):
        return "proportion_other"
    
    return "fallback"


def classify_l2_type(question: str, answer: str) -> Optional[str]:
    """
    For L2 questions only, classify into:
    - arithmetic_subtract: "how many years between/after X and Y"
    - arithmetic_divide: "given X and Y, how many … to recoup/cover/produce …"
    - count_aggregate: "how many of X are Y"
    - comparison: "by how many … did X change between A and B"
    - multi_fact: combining 2+ literal facts into one statement
    
    Returns None for L1 questions.
    """
    q_lower = question.lower()
    
    # arithmetic_subtract
    if any(x in q_lower for x in ["how many years between", "how many years after", "years passed between"]):
        return "arithmetic_subtract"
    
    # arithmetic_divide
    if any(x in q_lower for x in ["recoup", "cover", "produce", "given"]) and "how many" in q_lower:
        return "arithmetic_divide"
    
    # count_aggregate
    if "how many of" in q_lower and "are" in q_lower:
        return "count_aggregate"
    
    # comparison
    if any(x in q_lower for x in ["by how many", "change between"]):
        return "comparison"
    
    # multi_fact (combining 2+ facts)
    # Heuristic: long answer with multiple clauses
    if len(answer.split()) > 5 and any(x in answer.lower() for x in [" and ", ", and ", "; "]):
        return "multi_fact"
    
    # Default fallback for L2
    return "multi_fact"


def extract_snippet(doc_text: str, answer: str, context_words: int = 15) -> str:
    """Extract answer context from doc."""
    doc_lower = doc_text.lower()
    answer_lower = answer.lower()
    
    # Try exact substring match first
    idx = doc_lower.find(answer_lower)
    if idx == -1:
        # Try finding key words from answer
        words = re.findall(r"\b[a-z0-9]+\b", answer_lower)
        for word in words:
            if len(word) > 4:
                idx = doc_lower.find(word)
                if idx != -1:
                    break
    
    if idx == -1:
        return "[no match]"
    
    # Extract window
    start = max(0, idx - context_words * 5)
    end = min(len(doc_text), idx + len(answer) + context_words * 5)
    snippet = doc_text[start:end].replace("\n", " ")
    
    if start > 0:
        snippet = "..." + snippet
    if end < len(doc_text):
        snippet = snippet + "..."
    
    return snippet.strip()


def main():
    print("=" * 80)
    print("PHASE 2 ANSWER CHARACTERIZATION ANALYSIS")
    print("=" * 80)
    print()
    
    # Load data
    all_rows, dev_indices = load_data()
    dev_rows = [r for r in all_rows if r.key in dev_indices]
    
    print(f"Total rows: {len(all_rows)}")
    print(f"Dev set: {len(dev_rows)}")
    print(f"Held-out: {len(all_rows) - len(dev_rows)}")
    print()
    
    # ========== 1. VERBATIM-PRESENCE BREAKDOWN ==========
    print("=" * 80)
    print("1. VERBATIM-PRESENCE BREAKDOWN (by difficulty)")
    print("=" * 80)
    print()
    
    verbatim_stats = defaultdict(lambda: {"exact": 0, "all_words": 0, "pct_70": 0, "none": 0})
    breakdown_examples = defaultdict(list)  # {(difficulty, type): [examples]}
    
    for row in dev_rows:
        if not row.answerable or not row.source_docs:
            continue
        
        doc_id = row.source_docs[0]
        doc_text = load_doc(doc_id)
        if not doc_text:
            continue
        
        result = verbatim_check(row.answer, doc_text)
        vtype = result["type"]
        verbatim_stats[row.difficulty][vtype] += 1
        
        # Store example
        if len(breakdown_examples[(row.difficulty, vtype)]) < 3:
            breakdown_examples[(row.difficulty, vtype)].append({
                "question": row.question,
                "answer": row.answer,
                "snippet": extract_snippet(doc_text, row.answer),
            })
    
    for difficulty in sorted(verbatim_stats.keys()):
        stats = verbatim_stats[difficulty]
        total = sum(stats.values())
        if total == 0:
            continue
        
        print(f"\n{difficulty} ({total} questions):")
        for vtype in ["exact", "all_words", "pct_70", "none"]:
            count = stats[vtype]
            pct = 100 * count / total if total else 0
            print(f"  {vtype:15s}: {count:3d} ({pct:5.1f}%)")
        
        # Show examples
        for vtype in ["exact", "all_words", "pct_70", "none"]:
            examples = breakdown_examples.get((difficulty, vtype), [])
            if examples:
                print(f"\n  {vtype} examples:")
                for i, ex in enumerate(examples[:3], 1):
                    print(f"    [{i}] Q: {ex['question'][:70]}...")
                    print(f"        A: {ex['answer']}")
                    print(f"        Doc: {ex['snippet'][:100]}...")
    
    print()
    
    # ========== 2. ANSWER SHAPE HISTOGRAM ==========
    print("=" * 80)
    print("2. ANSWER SHAPE HISTOGRAM")
    print("=" * 80)
    print()
    
    shape_stats = defaultdict(int)
    shape_words = defaultdict(list)
    shape_examples = defaultdict(list)
    
    for row in dev_rows:
        if not row.answerable:
            continue
        
        shape = classify_answer_shape(row.answer)
        shape_stats[shape] += 1
        
        word_count = len(row.answer.split())
        shape_words[shape].append(word_count)
        
        if len(shape_examples[shape]) < 3:
            doc_text = load_doc(row.source_docs[0]) if row.source_docs else ""
            shape_examples[shape].append({
                "question": row.question,
                "answer": row.answer,
                "snippet": extract_snippet(doc_text, row.answer) if doc_text else "[no doc]",
            })
    
    for shape in sorted(shape_stats.keys()):
        count = shape_stats[shape]
        pct = 100 * count / len(dev_rows) if dev_rows else 0
        words = shape_words[shape]
        median_words = sorted(words)[len(words) // 2] if words else 0
        
        print(f"\n{shape} ({count} answers, {pct:.1f}%, median {median_words} words):")
        
        examples = shape_examples[shape]
        for i, ex in enumerate(examples[:3], 1):
            print(f"  [{i}] Q: {ex['question'][:70]}...")
            print(f"      A: {ex['answer']}")
            print(f"      Doc: {ex['snippet'][:100]}...")
    
    print()
    
    # ========== 3. QUESTION-TYPE BUCKET ASSIGNMENTS ==========
    print("=" * 80)
    print("3. QUESTION-TYPE BUCKET ASSIGNMENTS")
    print("=" * 80)
    print()
    
    bucket_stats = defaultdict(int)
    bucket_examples = defaultdict(list)
    bucket_word_counts = defaultdict(list)
    
    for row in dev_rows:
        if not row.answerable:
            continue
        
        bucket = question_to_bucket(row.question)
        bucket_stats[bucket] += 1
        
        word_count = len(row.answer.split())
        bucket_word_counts[bucket].append(word_count)
        
        if len(bucket_examples[bucket]) < 3:
            bucket_examples[bucket].append({
                "question": row.question,
                "answer": row.answer,
            })
    
    for bucket in sorted(bucket_stats.keys()):
        count = bucket_stats[bucket]
        pct = 100 * count / len(dev_rows) if dev_rows else 0
        words = bucket_word_counts[bucket]
        median_words = sorted(words)[len(words) // 2] if words else 0
        
        print(f"\n{bucket} ({count} questions, {pct:.1f}%, median answer {median_words} words):")
        
        examples = bucket_examples[bucket]
        for i, ex in enumerate(examples[:3], 1):
            print(f"  [{i}] Q: {ex['question'][:70]}...")
            print(f"      A: {ex['answer']}")
    
    print()
    
    # ========== 4. PER-BUCKET VERBATIM-EXTRACTABILITY ==========
    print("=" * 80)
    print("4. PER-BUCKET VERBATIM-EXTRACTABILITY (exact substring match)")
    print("=" * 80)
    print()
    
    bucket_verbatim = defaultdict(lambda: {"exact": 0, "total": 0})
    
    for row in dev_rows:
        if not row.answerable or not row.source_docs:
            continue
        
        bucket = question_to_bucket(row.question)
        doc_text = load_doc(row.source_docs[0])
        if not doc_text:
            continue
        
        result = verbatim_check(row.answer, doc_text)
        bucket_verbatim[bucket]["total"] += 1
        if result["type"] == "exact":
            bucket_verbatim[bucket]["exact"] += 1
    
    for bucket in sorted(bucket_verbatim.keys()):
        stats = bucket_verbatim[bucket]
        if stats["total"] == 0:
            continue
        pct = 100 * stats["exact"] / stats["total"]
        print(f"{bucket:20s}: {stats['exact']:3d}/{stats['total']:3d} = {pct:5.1f}% exact match")
    
    print()
    
    # ========== 5. L2-SPECIFIC ANALYSIS ==========
    print("=" * 80)
    print("5. L2-SPECIFIC BREAKDOWN (arithmetic_subtract, arithmetic_divide, etc.)")
    print("=" * 80)
    print()
    
    l2_rows = [r for r in dev_rows if r.difficulty == "L2"]
    l2_type_stats = defaultdict(int)
    l2_type_examples = defaultdict(list)
    
    for row in l2_rows:
        if not row.answerable:
            continue
        
        l2_type = classify_l2_type(row.question, row.answer)
        l2_type_stats[l2_type] += 1
        
        if len(l2_type_examples[l2_type]) < 3:
            doc_text = load_doc(row.source_docs[0]) if row.source_docs else ""
            l2_type_examples[l2_type].append({
                "question": row.question,
                "answer": row.answer,
                "snippet": extract_snippet(doc_text, row.answer) if doc_text else "[no doc]",
            })
    
    print(f"Total L2 questions in dev: {len(l2_rows)}\n")
    
    for l2type in sorted(l2_type_stats.keys()):
        count = l2_type_stats[l2type]
        pct = 100 * count / len(l2_rows) if l2_rows else 0
        
        print(f"\n{l2type} ({count} questions, {pct:.1f}%):")
        
        examples = l2_type_examples[l2type]
        for i, ex in enumerate(examples[:3], 1):
            print(f"  [{i}] Q: {ex['question'][:80]}...")
            print(f"      A: {ex['answer']}")
            print(f"      Doc: {ex['snippet'][:100]}...")
    
    print()
    print("=" * 80)
    print("END OF REPORT")
    print("=" * 80)


if __name__ == "__main__":
    main()
