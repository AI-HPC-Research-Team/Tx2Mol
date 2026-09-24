"""Verify every supplied data file before training or generation."""
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ('targets', 'sciplex3', 'patient')
ROLES = ('test', 'known_ligands')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(b)
    return h.hexdigest()


def read_csv(path):
    with path.open('r', newline='', encoding='utf-8-sig') as stream:
        return list(csv.reader(stream))


def validate_expression(values, name, row_number):
    try:
        valid = len(values) == 978 and all(math.isfinite(float(value)) for value in values)
    except (ValueError, TypeError):
        valid = False
    if not valid:
        raise ValueError(f'Invalid expression values in {name}, data row {row_number}')


def validate_scenarios(manifest, genes):
    entries = manifest.get('scenario_files')
    if not isinstance(entries, dict) or not entries:
        raise ValueError('scenario_files must describe all six populated scenario directories')
    actual_files = set()
    for scenario in SCENARIOS:
        for role in ROLES:
            directory = ROOT / 'data' / scenario / role
            files = sorted(path for path in directory.rglob('*.csv') if path.is_file())
            if not files:
                raise ValueError(f'Required input directory contains no CSV files: {directory}')
            actual_files.update(path.relative_to(ROOT).as_posix() for path in files)
    if set(entries) != actual_files:
        missing = sorted(actual_files - set(entries))
        absent = sorted(set(entries) - actual_files)
        raise ValueError(f'Scenario manifest coverage mismatch: unlisted={missing}; absent={absent}')
    tables = {}
    for name, entry in entries.items():
        relative = Path(name)
        scenario, role = relative.parts[1:3]
        if entry.get('scenario') != scenario or entry.get('role') != role:
            raise ValueError(f'Scenario/role metadata disagrees with file path: {name}')
        path = ROOT / relative
        if path.stat().st_size != entry['bytes'] or sha(path) != entry['sha256']:
            raise ValueError(f'Checksum or byte-count mismatch: {name}')
        rows = read_csv(path)
        if not rows or type(entry.get('header')) is not bool:
            raise ValueError(f'Empty CSV or invalid header specification: {name}')
        if any(len(row) != entry['columns'] for row in rows):
            raise ValueError(f'Column-count mismatch: {name}')
        header = rows[0] if entry['header'] else None
        data = rows[1:] if entry['header'] else rows
        if not data or len(data) != entry['rows']:
            raise ValueError(f'Row-count mismatch: {name}')
        offset = entry.get('expression_offset')
        if role == 'test' and offset is None:
            raise ValueError(f'Expression offset is required for test data: {name}')
        if offset is not None:
            if type(offset) is not int or offset < 0 or header is None:
                raise ValueError(f'Invalid expression offset or missing gene header: {name}')
            if len(header) != offset + 978 or header[offset:] != genes:
                raise ValueError(f'Gene order mismatch: {name}')
            for number, row in enumerate(data, start=1):
                validate_expression(row[offset:], name, number)
        tables[name] = (header, data)
        print(name, len(data), 'rows: OK')
    for name, entry in entries.items():
        paired_test = entry.get('paired_test')
        if paired_test is not None:
            paired_entry = entries.get(paired_test)
            if not paired_entry or paired_entry.get('scenario') != entry['scenario'] or paired_entry.get('role') != 'test':
                raise ValueError(f'Invalid paired test file for {name}: {paired_test}')
        if entry['scenario'] != 'sciplex3' or entry['role'] != 'known_ligands':
            continue
        if paired_test is None:
            raise ValueError(f'SciPlex3 known ligands require paired_test: {name}')
        header, data = tables[name]
        source_header, source_rows = tables[paired_test]
        if entries[paired_test].get('expression_offset') != 11:
            raise ValueError(f'SciPlex3 tests require exactly 11 metadata columns: {paired_test}')
        if header != ['test_row'] + source_header[:11] + ['smiles_parse_status']:
            raise ValueError(f'SciPlex3 known-ligand metadata header mismatch: {name}')
        if source_header[2].strip().lower() != 'perturbation' or source_header[4].strip().lower() != 'smiles':
            raise ValueError(f'SciPlex3 source lacks expected perturbation/SMILES columns: {paired_test}')
        expected = [index for index, row in enumerate(source_rows)
                    if row[2].strip().lower() != 'control']
        observed = []
        for row in data:
            try:
                index = int(row[0])
            except ValueError as error:
                raise ValueError(f'Invalid zero-based test_row in {name}: {row[0]!r}') from error
            if index < 0 or index >= len(source_rows) or row[1:12] != source_rows[index][:11]:
                raise ValueError(f'SciPlex3 source metadata/SMILES mismatch in {name}, test_row={index}')
            observed.append(index)
        if observed != expected:
            raise ValueError(f'SciPlex3 known ligands must retain every non-control row in original order: {name}')
        print(name, len(observed), 'non-control rows retain exact source metadata and SMILES: OK')


def main():
    manifest = json.loads((ROOT / 'assets/data_manifest.json').read_text(encoding='utf-8'))
    genes = json.loads((ROOT / 'data/gene_order.json').read_text(encoding='utf-8'))
    if len(genes) != 978 or len(set(genes)) != 978:
        raise ValueError('Expected 978 unique gene labels')
    for name, entry in manifest['dataset_files'].items():
        path = ROOT / name
        if sha(path) != entry['sha256']: raise ValueError(f'Checksum mismatch: {name}')
        count = 0
        with gzip.open(path, 'rt', newline='') as f:
            for row in csv.reader(f):
                if len(row) != 981 or not all(math.isfinite(float(v)) for v in row[3:]):
                    raise ValueError(f'Invalid row {count + 1} in {name}')
                count += 1
        if count != entry['rows']: raise ValueError(f'Row count mismatch: {name}')
        print(name, count, 'rows: OK')
    for target, entry in manifest['target_files'].items():
        path = ROOT / 'data/targets/test' / f'{target}.csv'
        ligand = ROOT / 'data/targets/known_ligands' / f'source_{target}.csv'
        if sha(path) != entry['sha256'] or sha(ligand) != entry['ligands_sha256']:
            raise ValueError(f'Checksum mismatch: {target}')
        with path.open() as f:
            reader = csv.reader(f)
            if next(reader) != genes: raise ValueError(f'Gene order mismatch: {target}')
            for row in reader:
                if len(row) != 978 or not all(math.isfinite(float(v)) for v in row):
                    raise ValueError(f'Invalid target profile: {target}')
        print(target, 'target and ligands: OK')
    validate_scenarios(manifest, genes)
    print('Data validated. Training matrices retain their supplied positional gene order.')


if __name__ == '__main__': main()
