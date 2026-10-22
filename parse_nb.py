import json

with open('alzheimer-detection.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']
print(f'Total cells: {len(cells)}')
for i, c in enumerate(cells):
    src = ''.join(c['source'])
    print(f'\n--- Cell {i} ({c["cell_type"]}) ---')
    print(src[:500])
