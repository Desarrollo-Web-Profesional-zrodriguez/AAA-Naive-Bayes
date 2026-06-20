import json

nb = json.load(open('visualizaciones_naivebayes.ipynb', encoding='utf-8'))
for i, c in enumerate(nb['cells']):
    src = c['source']
    preview = (src[0] if src else '')[:100].encode('ascii', errors='replace').decode()
    print(f"[{i}] {c['cell_type']} | id={c.get('id','?')} | {preview}")
