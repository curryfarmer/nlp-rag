"""Pure-regex retrieval-only NLP manager.

Phase 1 of the regex-cheese plan: rare-token-weighted inverted index with
phrase-match booster. No neural models, no chunking, no embeddings — stdlib
only. `qa_batch` returns the top-3 candidate document IDs and an empty
answer string.

Contract preserved for nlp_server.py:
- `load_corpus(documents: list[dict[str, str]])`
- `qa_batch(questions: list[str]) -> list[dict[str, list[str] | str]]`
- `loaded: bool`
"""
import math
import re
from collections import defaultdict


_STOPWORDS = frozenset((
    "the a an of and or to in for on with by is are was were be as at from "
    "that this it its their there they them which who whom whose what when "
    "where why how been being have has had do does did but if not no nor so "
    "than then into about over under between within against during through "
    "before after above below up down out off again further once each any "
    "all some most more many much few several other another such own same "
    "very can could should would may might must will shall did done"
).split())

_RE_CODE = re.compile(r"\b[A-Z][A-Za-z0-9]*(?:-[A-Z0-9]+)+\b")
_RE_DATE = re.compile(r"\bQ[1-4]\s+\d{2,4}\s+PCE\b|\b\d{2,4}-\d{2}-\d{2}\b|\b\d{2,4}\s+PCE\b")
_RE_CAPS_WORD = re.compile(r"\b[A-Z][A-Za-z'-]{1,}\b")
_RE_WORD = re.compile(r"\b[a-z][a-z'-]{2,}\b")
_RE_NUM = re.compile(r"\b\d{2,}\b")
# Run of capitalised words (entity phrase), allows internal hyphens.
_RE_CAP_PHRASE = re.compile(r"\b(?:[A-Z][A-Za-z0-9'-]+)(?:\s+(?:[A-Z][A-Za-z0-9'-]+|of|the|and|de|du|von|van))*\b")


def _tokens(text: str) -> list[str]:
    out: list[str] = []
    out.extend(_RE_CODE.findall(text))
    out.extend(_RE_DATE.findall(text))
    out.extend(t for t in _RE_CAPS_WORD.findall(text) if len(t) >= 3)
    out.extend(_RE_NUM.findall(text))
    for w in _RE_WORD.findall(text.lower()):
        if w not in _STOPWORDS:
            out.append(w)
    return out


def _question_phrases(question: str) -> list[str]:
    """Capitalised n-gram phrases (length >=2 words OR length >=4 chars single word)
    to search verbatim in doc text. Also includes raw codes and date stamps."""
    phrases: set[str] = set()
    for m in _RE_CAP_PHRASE.findall(question):
        m = m.strip()
        words = m.split()
        if len(words) >= 2:
            phrases.add(m)
        elif len(m) >= 4:
            phrases.add(m)
    phrases.update(_RE_CODE.findall(question))
    phrases.update(_RE_DATE.findall(question))
    phrases = {p for p in phrases if not all(w.lower() in _STOPWORDS for w in p.split())}
    return sorted(phrases, key=lambda p: (-len(p), p))


def _content_bigrams(question: str) -> list[str]:
    """Adjacent pairs of content (non-stopword) words from the question, lowercased."""
    words = [w.lower() for w in re.findall(r"\b[A-Za-z][A-Za-z'-]+\b", question)]
    content = [w for w in words if w not in _STOPWORDS and len(w) >= 3]
    return [f"{a} {b}" for a, b in zip(content, content[1:])]


class NLPManager:
    loaded = False

    def __init__(self) -> None:
        self.postings: dict[str, set[str]] = defaultdict(set)
        self.df: dict[str, int] = {}
        self.doc_text: dict[str, str] = {}
        self.doc_text_lower: dict[str, str] = {}
        self.doc_len: dict[str, int] = {}
        self.doc_ids: list[str] = []
        self.N: int = 0
        self.avg_len: float = 1.0

    def load_corpus(self, documents: list[dict[str, str]]) -> None:
        self.postings.clear()
        self.df.clear()
        self.doc_text.clear()
        self.doc_text_lower.clear()
        self.doc_len.clear()
        self.doc_ids = []
        for doc in documents:
            doc_id = doc["id"]
            text = doc["document"]
            self.doc_ids.append(doc_id)
            self.doc_text[doc_id] = text
            self.doc_text_lower[doc_id] = text.lower()
            toks = _tokens(text)
            self.doc_len[doc_id] = max(1, len(toks))
            for tok in set(toks):
                self.postings[tok].add(doc_id)
        self.df = {tok: len(ids) for tok, ids in self.postings.items()}
        self.N = len(self.doc_ids)
        self.avg_len = (sum(self.doc_len.values()) / self.N) if self.N else 1.0
        self.loaded = True

    def _retrieve(self, question: str, k: int = 3) -> list[str]:
        if not self.loaded or self.N == 0:
            return []
        q_tokens = _tokens(question)
        scores: dict[str, float] = defaultdict(float)
        unique_hits: list[str] = []

        # 1. Token-level rarity-weighted scoring + per-doc distinct-token coverage bonus.
        coverage: dict[str, int] = defaultdict(int)
        seen_q_tokens: set[str] = set()
        for tok in q_tokens:
            if tok in seen_q_tokens:
                continue
            seen_q_tokens.add(tok)
            df = self.df.get(tok)
            if not df:
                continue
            idf = math.log((self.N + 1) / (df + 1)) + 1.0
            for doc_id in self.postings[tok]:
                scores[doc_id] += idf
                coverage[doc_id] += 1
            if df == 1:
                unique_hits.extend(self.postings[tok])
        # Add coverage bonus so docs matching many question terms beat narrow-but-redundant matches.
        for doc_id, c in coverage.items():
            scores[doc_id] += 0.5 * c

        # 2. Verbatim multi-word phrase booster — only fires for very rare phrases.
        for phrase in _question_phrases(question):
            words = phrase.split()
            if len(words) < 2:
                continue
            needle = phrase.lower()
            hits = [d for d, t in self.doc_text_lower.items() if needle in t]
            if not hits or len(hits) > 3:
                continue
            phrase_idf = math.log((self.N + 1) / (len(hits) + 1)) + 1.0
            bonus = phrase_idf * len(words)
            for doc_id in hits:
                scores[doc_id] += bonus
            if len(hits) == 1:
                unique_hits.extend(hits)

        # 3. Content-bigram booster — adjacent content words in question, verbatim in doc.
        for bg in _content_bigrams(question):
            hits = [d for d, t in self.doc_text_lower.items() if bg in t]
            if not hits or len(hits) > 5:
                continue
            bg_idf = math.log((self.N + 1) / (len(hits) + 1)) + 1.0
            for doc_id in hits:
                scores[doc_id] += bg_idf * 0.8

        if not scores:
            return self.doc_ids[:k]

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        ordered = [d for d, _ in ranked]
        if unique_hits:
            seen: set[str] = set()
            forced: list[str] = []
            for d in unique_hits:
                if d not in seen:
                    seen.add(d)
                    forced.append(d)
            tail = [d for d in ordered if d not in seen]
            ordered = forced + tail
        return ordered[:k]

    def qa(self, question: str) -> dict[str, list[str] | str]:
        return self.qa_batch([question])[0]

    def qa_batch(self, questions: list[str]) -> list[dict[str, list[str] | str]]:
        import os
        use_llm = os.getenv("NLP_USE_LLM", "0") == "1"
        llm_top_k = int(os.getenv("NLP_LLM_TOP_K", "3"))
        if use_llm:
            from nlp_answer import extract_answer_llm
        elif os.getenv("NLP_USE_RERANK", "0") == "1":
            from nlp_answer import extract_answer_reranked as _extract
        else:
            from nlp_answer import extract_answer as _extract
        results = []
        for q in questions:
            docs = self._retrieve(q, k=3)
            if not docs:
                results.append({"documents": docs, "answer": ""})
                continue
            if use_llm:
                ans = extract_answer_llm(q, [self.doc_text[d] for d in docs[:llm_top_k]])
            else:
                ans = _extract(q, self.doc_text[docs[0]])
            results.append({"documents": docs, "answer": ans})
        return results
