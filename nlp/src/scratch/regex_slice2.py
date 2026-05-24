"""
PHASE 2 REGEX EXTRACTOR - Slice 2 (DOC-0101 to DOC-0200)
Pure regex patterns for high-precision answer span extraction from cyberpunk corpus.
Testing against dev-set gold answers.
"""

import json
import re
import os
from collections import defaultdict

# ============================================================================
# 1. LOAD SLICE DATA
# ============================================================================

with open('/Users/jp/Desktop/Coding projects/Brainhack 2026/nlp-rag-stuff/nlp/src/eval_split.json') as f:
    split_data = json.load(f)
dev_indices = set(split_data['dev'])

target_docs = [f"DOC-{i:04d}" for i in range(101, 201)]
target_docs_set = set(target_docs)

slice_data = []
with open('/Users/jp/Desktop/Coding projects/Brainhack 2026/nlp-rag-stuff/nlp/src/nlp.jsonl') as f:
    for line in f:
        row = json.loads(line)
        if row['key'] in dev_indices and any(doc in target_docs_set for doc in row['source_docs']):
            slice_data.append(row)

print(f"Loaded {len(slice_data)} dev QA pairs for DOC-0101 to DOC-0200\n")

# ============================================================================
# 2. PROPOSED TIGHT REGEXES (HIGH-PRECISION)
# ============================================================================

REGEXES = {
    # Money: dollar amounts (Credits or Phi Credits)
    'money': r'(?:^|\s)(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*(?:Phi\s*)?Credits',
    
    # Percentage: percent signs
    'percent': r'~?\d+(?:\.\d+)?%',
    
    # Large numbers: with million/billion/trillion suffix
    'large_number': r'(?:approximately|roughly|~)?\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:million|billion|trillion)',
    
    # Date PCE: Q4 77 PCE, 77-02-10, 76 PCE etc
    'date_pce': r'(?:Q[1-4]\s*)?(\d{2})\s+(?:PCE|(?:\d{2}-\d{2}-\d{2}))',
    
    # Count: ~342000, approximately 71 million (for Fullwalker counts)
    'count': r'(?:approximately|roughly|~)?\s*(\d+(?:,\d{3})*)\s+(?:of|stations|attendees|nanobots|users|crew)?',
    
    # Ratio: 3 of 47, 312 of 547 (for station counts)
    'ratio': r'(\d+)\s+of\s+(\d+)\s+\w+\s*\(~?(\d+(?:\.\d+)?)%\)',
    
    # Codename: SEASTITCH, ASHCASTLE WATCHES (all caps proper nouns)
    'codename': r'\b([A-Z]{2,}(?:\s+[A-Z]{2,})*)\b',
}

# ============================================================================
# 3. CATEGORIZE ANSWERS
# ============================================================================

def get_answer_shape(ans):
    """Classify answer by shape for analysis."""
    if re.search(r'(?:Phi\s*)?Credits', ans, re.I):
        return 'money'
    if re.search(r'\d+(?:\.\d+)?%', ans):
        return 'percent'
    if re.search(r'\d+\s*(?:million|billion|trillion)', ans, re.I):
        return 'large_number'
    if re.search(r'(?:PCE|CE|\d{2}-\d{2}-\d{2})', ans):
        return 'date_pce'
    if re.search(r'\d+\s+of\s+\d+', ans):
        return 'ratio'
    if re.match(r'^[A-Z][A-Z0-9\s]+$', ans.strip()) and len(ans) < 50:
        return 'codename'
    if len(ans.split()) <= 2 and re.match(r'^\d+$', ans.strip()):
        return 'count'
    return 'sentence_fragment'

# Categorize all answers
shapes = defaultdict(list)
for qa in slice_data:
    shape = get_answer_shape(qa['answer'])
    shapes[shape].append(qa)

print("=== ANSWER SHAPE INVENTORY ===")
for shape in sorted(shapes.keys(), key=lambda x: -len(shapes[x])):
    print(f"{shape}: {len(shapes[shape])} answers")
    for qa in shapes[shape][:3]:
        print(f"  - {qa['answer']}")
print()

# ============================================================================
# 4. TEST REGEXES ON SAMPLE DOCUMENTS
# ============================================================================

doc_dir = '/Users/jp/Desktop/Coding projects/Brainhack 2026/nlp-rag-stuff/nlp/src/documents'

print("\n=== REGEX TEST RESULTS ===\n")

for shape, regex in REGEXES.items():
    print(f"\n{shape.upper()}")
    print(f"Pattern: {regex}\n")
    
    # Collect matches from all docs in slice
    all_matches = defaultdict(int)
    example_matches = []
    
    for doc_id in sorted(set(doc for qa in slice_data for doc in qa['source_docs'])):
        doc_path = os.path.join(doc_dir, f"{doc_id}.txt")
        if os.path.exists(doc_path):
            with open(doc_path) as f:
                text = f.read()
                matches = re.findall(regex, text)
                if matches:
                    all_matches[doc_id] = len(matches)
                    if len(example_matches) < 5:
                        example_matches.extend(matches[:5 - len(example_matches)])
    
    print(f"Total matches across slice: {sum(all_matches.values())} in {len(all_matches)} docs")
    print("Sample matches:")
    for m in example_matches[:5]:
        print(f"  - {m}")
    
    # Test precision: how many gold answers match this regex?
    precision_count = 0
    for qa in slice_data:
        if re.search(regex, qa['answer']):
            precision_count += 1
    
    print(f"Gold answer coverage: {precision_count}/{len(slice_data)} ({100*precision_count/len(slice_data):.1f}%)")
    print()

