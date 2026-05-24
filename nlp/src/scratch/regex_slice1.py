"""
REGEX EXTRACTION PATTERNS FOR DOC-0001..DOC-0100 SLICE
Analyzes 223 dev QA pairs, tests candidate regex patterns.
"""

import json
import re
import os
from collections import defaultdict, Counter

# === LOAD DATA ===
base_dir = '/Users/jp/Desktop/Coding projects/Brainhack 2026/nlp-rag-stuff/nlp/src'

with open(f'{base_dir}/eval_split.json') as f:
    splits = json.load(f)

dev_indices = set(splits['dev'])

all_rows = []
with open(f'{base_dir}/nlp.jsonl') as f:
    for i, line in enumerate(f):
        all_rows.append(json.loads(line))

# Filter to slice
dev_in_slice = {}
for row in all_rows:
    if row['key'] in dev_indices:
        for doc in row.get('source_docs', []):
            doc_num = int(doc.replace('DOC-', ''))
            if 1 <= doc_num <= 100:
                dev_in_slice[row['key']] = row
                break

# Load docs
doc_cache = {}
doc_dir = f'{base_dir}/documents'
for i in range(1, 101):
    fname = f'{doc_dir}/DOC-{i:04d}.txt'
    if os.path.exists(fname):
        with open(fname) as f:
            doc_cache[f'DOC-{i:04d}'] = f.read()

print("="*100)
print("REGEX PATTERN TEST SUITE FOR SLICE DOC-0001..DOC-0100")
print("="*100)
print(f"Total dev QA pairs in slice: {len(dev_in_slice)}")
print(f"Total docs available: {len(doc_cache)}\n")

# === CANDIDATE PATTERNS ===
patterns = {
    'MONEY_CREDITS': {
        'regex': r'(\d+(?:,\d{3})*)\s+(?:(?:million|billion|trillion)\s+)?(?:Phi\s+)?Credits',
        'desc': 'Money amounts ending in Credits/Phi'
    },
    'PERCENT_SYMBOL': {
        'regex': r'(\d+(?:\.\d+)?)\s*%',
        'desc': 'Percentage with % symbol'
    },
    'PERCENT_WORD': {
        'regex': r'(\d+(?:\.\d+)?)\s*(?:percent|percentage)',
        'desc': 'Percentage written as word'
    },
    'DATE_PCE': {
        'regex': r'(\d{2,4}(?:-\d{2}(?:-\d{2})?)?)\s+PCE',
        'desc': 'Date in PCE format'
    },
    'DATE_CE': {
        'regex': r'(\d{4})\s+CE',
        'desc': 'Date in CE format'
    },
    'COUNT_WITH_UNITS': {
        'regex': r'(\d+(?:,\d{3})*)\s+(?:million\s+)?(?:metric\s+)?(tons|nanobots|reactors|blocks|workers|staff|hours|days|years|jobs|credits)',
        'desc': 'Numeric count with units'
    },
    'QUOTED_TEXT': {
        'regex': r"['\"]([^'\"]{8,})['\"]",
        'desc': 'Text in quotes (8+ chars)'
    },
    'YESNO': {
        'regex': r'\b(Yes|No)\b',
        'desc': 'Simple Yes/No answer'
    },
}

# === TEST ON SAMPLE ===
import random
random.seed(42)
sample_keys = random.sample(list(dev_in_slice.keys()), min(40, len(dev_in_slice)))

results = defaultdict(lambda: {'tested': 0, 'extracted': 0, 'correct': 0})

for key in sample_keys:
    row = dev_in_slice[key]
    doc_id = row['source_docs'][0]
    doc_text = doc_cache.get(doc_id, "")
    answer = row['answer'].strip()
    
    if not doc_text or answer not in doc_text:
        continue
    
    for pname, pinfo in patterns.items():
        pregex = pinfo['regex']
        try:
            matches = list(re.finditer(pregex, doc_text))
        except:
            continue
        
        results[pname]['tested'] += 1
        
        if not matches:
            continue
        
        results[pname]['extracted'] += 1
        
        for m in matches:
            if answer in m.group(0):
                results[pname]['correct'] += 1
                break

print("\nPATTERN PERFORMANCE ON 40 SAMPLES")
print("="*100)
print(f"{'Pattern':<20} {'Tested':>6} {'Extracted':>10} {'Correct':>8} {'Precision':>10}")
print("-"*100)

for pname in sorted(patterns.keys()):
    r = results[pname]
    if r['tested'] == 0:
        continue
    prec = (r['correct'] / r['extracted'] * 100) if r['extracted'] > 0 else 0
    print(f"{pname:<20} {r['tested']:>6} {r['extracted']:>10} {r['correct']:>8} {prec:>9.1f}%")

# === ANSWER SHAPE INVENTORY ===
print("\n" + "="*100)
print("ANSWER SHAPE INVENTORY (223 QA pairs in slice)")
print("="*100)

def classify_answer(ans):
    if re.search(r'\d+(?:\.\d+)?\s*(?:million|billion|trillion)?\s*(?:Credits|Phi)', ans, re.I):
        return 'MONEY'
    if re.search(r'\d+(?:\.\d+)?\s*%|percent', ans, re.I):
        return 'PERCENT'
    if re.search(r'\d{2,4}\s*PCE|\d{4}\s*CE', ans):
        return 'DATE'
    if re.match(r'^(?:\d+|[Aa]pproximately\s+\d+|[A-Z]ne)', ans):
        return 'COUNT'
    if re.match(r'^[A-Z][A-Z0-9\-]{3,}$', ans):
        return 'CODENAME'
    if ans.lower() in ['yes', 'no']:
        return 'YESNO'
    if len(ans) > 60:
        return 'LONG_TEXT'
    return 'SHORT_TEXT'

shapes = Counter()
shape_examples = defaultdict(list)

for row in dev_in_slice.values():
    ans = row['answer'].strip()
    shape = classify_answer(ans)
    shapes[shape] += 1
    if len(shape_examples[shape]) < 3:
        shape_examples[shape].append(ans)

print(f"\n{'Shape':<15} {'Count':>6} {'Examples'}")
print("-"*100)
for shape, count in shapes.most_common():
    examples = ' | '.join(e[:35] for e in shape_examples[shape])
    print(f"{shape:<15} {count:>6}   {examples}")

print("\n" + "="*100)
print("END OF ANALYSIS")
print("="*100)
