"""Install checksum-pinned release weights from local assets or a GitHub Release."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def check_files(asset):
    return all((ROOT / name).is_file() and sha256(ROOT / name) == info['sha256']
               for name, info in asset['files'].items())


def install(asset, args):
    if check_files(asset):
        print('Already verified:', asset['asset'])
        return
    if args.assets_dir:
        archive = Path(args.assets_dir).expanduser().resolve() / asset['asset']
        if not archive.is_file():
            raise FileNotFoundError(archive)
    else:
        if not args.github_repo or len(args.github_repo.split('/')) != 2:
            raise ValueError('Use --assets_dir DIRECTORY or --github_repo OWNER/REPOSITORY')
        directory = ROOT / 'downloads'
        directory.mkdir(exist_ok=True)
        archive = directory / asset['asset']
        if not archive.exists() or sha256(archive) != asset['sha256']:
            url = f'https://github.com/{args.github_repo}/releases/download/{args.tag}/{asset["asset"]}'
            print('Downloading', url, flush=True)
            temporary = archive.with_suffix(archive.suffix + '.partial')
            request = urllib.request.Request(url, headers={'User-Agent': 'Tx2Mol-reproduce'})
            with urllib.request.urlopen(request, timeout=60) as source, temporary.open('wb') as target:
                shutil.copyfileobj(source, target, 8 * 1024 * 1024)
            temporary.replace(archive)
    if sha256(archive) != asset['sha256']:
        raise ValueError(f'Checksum mismatch: {archive}')
    allowed = set(asset['files'])
    extracted = set()
    with tarfile.open(archive, 'r:gz') as source:
        for member in source:
            if member.name not in allowed or not member.isfile():
                raise ValueError(f'Unexpected archive member: {member.name}')
            target = (ROOT / member.name).resolve()
            if ROOT not in target.parents:
                raise ValueError('Archive path leaves repository')
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.extractfile(member) as stream, target.open('wb') as out:
                shutil.copyfileobj(stream, out, 8 * 1024 * 1024)
            if sha256(target) != asset['files'][member.name]['sha256']:
                raise ValueError(f'Extracted file checksum mismatch: {member.name}')
            extracted.add(member.name)
    if extracted != allowed:
        raise ValueError('Release archive is incomplete')
    print('Installed and verified:', asset['asset'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets_dir', '--assets-dir')
    parser.add_argument('--github_repo', '--github-repo')
    parser.add_argument('--tag', default='v1.0')
    parser.add_argument('--base', action='store_true', help='Base model for fresh post-training')
    parser.add_argument('--reference', action='store_true', help='Archived epoch-9 model and its matching GeneVAE')
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()
    if not (args.base or args.reference or args.all):
        parser.error('Select --base, --reference, or --all')
    info = json.loads((ROOT / 'assets/release_manifest.json').read_text())
    for key in ('base', 'reference'):
        if args.all or getattr(args, key):
            install(info[key], args)


if __name__ == '__main__':
    main()
