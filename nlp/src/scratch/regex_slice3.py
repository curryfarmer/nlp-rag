#!/usr/bin/env python3
"""
REGEX EXTRACTION SYSTEM FOR DOC-0201–DOC-0296 SLICE
Pure-regex answer span extraction with structural pattern awareness.
"""

import json
import re
from pathlib import Path

def load_data():
    with open('nlp/src/eval_split.json') as f:
        split = json.load(f)
    with open('nlp/src/nlp.jsonl') as f:
        lines = f.readlines()
    
    dev_idx = set(split['dev'])
    slice_rows = []
    for idx, line in enumerate(lines):
        if idx in dev_idx:
            row = json.loads(line)
            doc_id = int(row['source_docs'][0].replace('DOC-', ''))
            if 201 <= doc_id <= 296:
                slice_rows.append((idx, row))
    return slice_rows

def load_doc(doc_id):
    path = f'nlp/src/documents/DOC-{doc_id:04d}.txt'
    try:
        with open(path) as f:
            return f.read()
    except:
        return None

# ===== REFINED REGEX PATTERNS =====

PATTERNS = {
    'money': [
        # Captures: "847 billion Phi Credits", "2.1 million Credits", "12,345,678 Phi Credits"
        (r'\d+(?:[.,]\d{3})*\s+(?:million|billion|thousand)?\s*(?:Phi\s)?Credits', 'Amount notation'),
        (r'\d+\.?\d*\s+(?:million|billion|thousand)\s+(?:Phi\s)?Credits', 'Scientific notation'),
    ],
    'percent': [
        # Captures: "5.3%", "11.2%", "3.0 percentage points"
        (r'\d+\.?\d*\s*%', 'Percent with %'),
        (r'\d+\.?\d*\s+percentage\s+points?', 'Percentage points'),
    ],
    'count': [
        # Captures: "18 hours per day", "21 years", "2.1 million metric tonnes"
        (r'\d+\.?\d*\s+(?:million\s+)?(?:metric\s+)?tonnes?', 'Mass quantity'),
        (r'\d+\s+(?:hours?|days?|weeks?|months?|years?|vessels?|cranes?|nodes?)', 'Unit count'),
    ],
    'codename': [
        # Captures: "OLINK-7", "SR-7741", "Class III"
        (r'[A-Z]{2,}-\d+', 'Alphanumeric codes'),
        (r'Class\s+[IVX]+', 'Classification'),
    ],
    'entity': [
        # Captures: "Genesis Labs", "Cyanite Industries", "ONE Network Enterprises"
        (r'(?:[A-Z][a-z]+\s+)+(?:Industries|Enterprises|Labs|Council|Group|Media|Research)', 'Org name'),
    ],
    'date': [
        # Captures: "less than one year", "approximately 37 years"
        (r'(?:less|more)\s+than\s+(?:one|a)\s+year', 'Relative time'),
        (r'(?:approximately\s+)?\d+\s+years?', 'Year count'),
    ],
    'fraction': [
        # Captures: "One-third (8 of 24 vessels)", "Two-thirds"
        (r'(?:One|Two|Three|Four|Five|Six|Seven|Eight|Nine)-(?:thirds?|quarters?|halves?)', 'Fraction word'),
        (r'\d+\s+of\s+\d+\s+(?:vessels?|units?|cranes?)', 'Fraction numeric'),
    ],
}

def test_pattern(pattern, text, gold_answer):
    """Test if pattern finds gold answer in text."""
    matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
    for m in matches:
        if isinstance(m, tuple):
            m = m[0]
        if gold_answer.lower() in m.lower() or m.lower() in gold_answer.lower():
            return True, matches[:5]
    return False, matches[:5] if matches else []

def main():
    print("=" * 90)
    print("REGEX EXTRACTION SYSTEM: DOC-0201 TO DOC-0296")
    print("=" * 90)
    
    slice_rows = load_data()
    print(f"\nLoaded {len(slice_rows)} dev Q&A pairs.\n")
    
    # Categorize by shape
    def cat(ans):
        if re.search(r'\d+.*(?:million|billion|thousand)?.*Credits', ans, re.I):
            return 'money'
        if re.search(r'\d+\.?\d*%', ans):
            return 'percent'
        if re.search(r'[A-Z]{2,}-\d|Class\s+[IVX]', ans):
            return 'codename'
        if len(ans) > 60:
            return 'sentence'
        if re.search(r'\d+\.?\d*\s+(?:million.*)?(?:metric.*)?tonnes?', ans, re.I):
            return 'count'
        if re.search(r'One-third|Two-thirds|Half|\d+\s+of\s+\d+', ans, re.I):
            return 'fraction'
        if re.search(r'(?:[A-Z][a-z]+\s+)+(?:Industries|Enterprises|Labs)', ans):
            return 'entity'
        if re.search(r'less than|more than|\d+\s+years?', ans, re.I):
            return 'date'
        return 'other'
    
    by_cat = {}
    for idx, row in slice_rows:
        c = cat(row['answer'])
        by_cat.setdefault(c, []).append((idx, row))
    
    print("ANSWER SHAPE INVENTORY:\n")
    for shape in ['money', 'percent', 'codename', 'count', 'entity', 'date', 'fraction', 'sentence', 'other']:
        if shape not in by_cat:
            continue
        rows = by_cat[shape]
        print(f"{shape.upper():12} {len(rows):3d} answers")
        for _, row in rows[:3]:
            ans = row['answer'][:70]
            print(f"  • {ans}")
    
    # Test patterns
    print("\n" + "=" * 90)
    print("REGEX PATTERN TESTING\n")
    
    for shape, pattern_list in PATTERNS.items():
        if shape not in by_cat:
            continue
        
        rows = by_cat[shape][:8]
        print(f"{shape.upper()} ({len(by_cat[shape])} total):")
        
        for pat_idx, (pattern, desc) in enumerate(pattern_list, 1):
            hits = 0
            for _, row in rows:
                doc_id = int(row['source_docs'][0].replace('DOC-', ''))
                doc = load_doc(doc_id)
                if doc and test_pattern(pattern, doc, row['answer'])[0]:
                    hits += 1
            
            print(f"  [{pat_idx}] {pattern[:75]}")
            print(f"      {desc} | Matches: {hits}/{len(rows)}")
        print()

if __name__ == '__main__':
    main()
