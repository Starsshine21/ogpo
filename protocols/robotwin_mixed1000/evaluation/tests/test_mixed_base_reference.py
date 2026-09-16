import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def test_all_thousand_reference_records_match_protocol():
    spec = importlib.util.spec_from_file_location('mixed_bundle_tools', ROOT / 'scripts/bundle_tools.py')
    tools = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tools)
    references = json.loads((ROOT / 'BASE_REFERENCE.json').read_text())
    for task in json.loads((ROOT / 'protocol.json').read_text())['tasks']:
        manifest = json.loads((ROOT / task['manifest']).read_text())
        results = tools._result_map(ROOT / 'base_reference' / task['name'], manifest, task['name'], None)
        assert len(results) == 100
        assert sum(results.values()) == references['tasks'][task['name']]['successes']
