"""Inspect Extended Yale B and write a reproducible, stratified 70/15/15 split manifest.

Preprocessing step 1: check files, labels and dimensions; drop unreadable/duplicate/ambient images.
Every person appears in train, val and test (identity classification). Images are not copied:
the manifest points at the raw files. Reruns require a fresh output directory.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
from PIL import Image
from sklearn.model_selection import train_test_split

NAME = re.compile(r'yaleB(\d+)_P(\d+)A([+-]\d+)E([+-]\d+)', re.I)


def light_angle(az, el):
    """Degrees between the light source and the camera axis."""
    return math.degrees(math.acos(math.cos(math.radians(az)) * math.cos(math.radians(el))))


def scan(root):
    recs, seen, sizes = [], set(), Counter()
    skipped = Counter()
    for p in sorted(Path(root).rglob('*')):
        if p.suffix.lower() not in {'.pgm', '.png', '.jpg', '.jpeg', '.bmp'}:
            continue
        m = NAME.search(p.name)
        if not m:  # e.g. *_Ambient.pgm has no light direction
            skipped['unparsed'] += 1
            continue
        try:
            with Image.open(p) as im:
                sizes[im.size] += 1
                im.verify()
        except Exception:
            skipped['unreadable'] += 1
            continue
        digest = hashlib.md5(p.read_bytes()).hexdigest()
        if digest in seen:
            skipped['duplicate'] += 1
            continue
        seen.add(digest)
        s, pose, az, el = m.groups()
        recs.append(dict(path=str(p.resolve()), subject=f'yaleB{s}', pose=int(pose),
                         angle=light_angle(int(az), int(el))))
    if not recs:
        raise ValueError(f'No Yale B images found under {root}')
    n_subj = len({r['subject'] for r in recs})
    print(f'{len(recs)} images, {n_subj} subjects, sizes {dict(sizes)}, skipped {dict(skipped)}')
    return recs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_dir', default=None, help='raw Yale B folder (default: download from Kaggle)')
    parser.add_argument('--output_dir', default='data/yaleb')
    parser.add_argument('--seed', type=int, default=4110)
    args = parser.parse_args()
    destination = Path(args.output_dir)
    if destination.exists() and any(p.name != '.gitkeep' for p in destination.iterdir()):
        raise FileExistsError('Use a fresh output directory to prevent stale split membership.')
    data_dir = args.data_dir
    if data_dir is None:  # kagglehub caches the download for later runs
        import kagglehub
        data_dir = kagglehub.dataset_download('jensdhondt/extendedyaleb-cropped-full')
        print('Path to dataset files:', data_dir)
    recs = scan(data_dir)
    y = [r['subject'] for r in recs]
    train, rest = train_test_split(range(len(recs)), train_size=0.70, stratify=y, random_state=args.seed)
    val, test = train_test_split(rest, train_size=0.50, stratify=[y[i] for i in rest], random_state=args.seed)
    splits = {name: [recs[i] for i in idx] for name, idx in [('train', train), ('val', val), ('test', test)]}
    payload = json.dumps(splits, sort_keys=True).encode()
    manifest = dict(source=str(Path(data_dir).resolve()), seed=args.seed, classes=sorted(set(y)),
                    splits=splits, split_sha256=hashlib.sha256(payload).hexdigest())
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'split_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print({split: len(items) for split, items in splits.items()})


if __name__ == '__main__':
    main()
