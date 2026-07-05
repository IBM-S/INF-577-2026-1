from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
INSTANCES_DIR = REPO_ROOT / 'data' / 'instances'
OUT_FILE = REPO_ROOT / 'ampl' / 'instance_lists.sh'


def order_block(var_name, files):
    block = [f'declare -a {var_name}=(']
    block += [f'    "{f.name}"' for f in files]
    block.append(')')
    return block


cdmx_files = sorted((INSTANCES_DIR / 'cdmx').glob('*.dat'), key=lambda f: f.name)
clustering_files = sorted((INSTANCES_DIR / 'clustering').glob('*.dat'), key=lambda f: f.name)
phub_files = sorted((INSTANCES_DIR / 'phub').glob('*.dat'),
                     key=lambda f: int(f.stem.split('_')[1]))

lines = order_block('CDMX_ORDER', cdmx_files) + ['']
lines += order_block('CLUSTERING_ORDER', clustering_files) + ['']
lines += order_block('PHUB_ORDER', phub_files)

with open(OUT_FILE, 'w', encoding='utf-8') as out:
    out.write('\n'.join(lines) + '\n')

print(f'Escrito: {OUT_FILE}')
print(f'  CDMX_ORDER:       {len(cdmx_files)} entradas')
print(f'  CLUSTERING_ORDER: {len(clustering_files)} entradas')
print(f'  PHUB_ORDER:       {len(phub_files)} entradas')
