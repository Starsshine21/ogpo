"""Refresh the parent collection/evaluation bundle integrity manifest."""
import hashlib
from pathlib import Path

root = Path(__file__).resolve().parents[2]
entries = []
for path in sorted(root.rglob('*')):
    if not path.is_file() or path.name == 'SHA256SUMS':
        continue
    if '__pycache__' in path.parts or '.pytest_cache' in path.parts:
        continue
    entries.append(hashlib.sha256(path.read_bytes()).hexdigest() + '  ' + str(path.relative_to(root)))
(root / 'SHA256SUMS').write_text('\n'.join(entries) + '\n')
