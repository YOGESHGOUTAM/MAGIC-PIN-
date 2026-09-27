import sys
import json
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

pairs = json.load(open('dataset/expanded/test_pairs.json', encoding='utf-8'))['pairs']
triggers = {}
t_dir = Path('dataset/expanded/triggers')
for f in t_dir.glob('*.json'):
    t = json.load(open(f, encoding='utf-8'))
    triggers[t['id']] = t

merchants = {}
m_dir = Path('dataset/expanded/merchants')
for f in m_dir.glob('*.json'):
    m = json.load(open(f, encoding='utf-8'))
    merchants[m['merchant_id']] = m

for p in pairs:
    tid = p['trigger_id']
    t = triggers[tid]
    m = merchants[p['merchant_id']]
    kind = t.get('kind')
    scope = t.get('scope')
    keys = list(t.get('payload', {}).keys())
    print(f"{p['test_id']}: mid={m['merchant_id'][:20]} | cat={m.get('category_slug')} | kind={kind} | scope={scope} | payload={t.get('payload')}")
