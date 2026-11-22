"""Generate tiny synthetic Yale-style images (3 fake people) to verify the code without a download.

Then: python build_dataset.py --data_dir data/toy_raw --output_dir data/toy
These are NOT faces or a valid course-project dataset.
"""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output_dir', default='data/toy_raw')
    parser.add_argument('--seed', type=int, default=4110)
    args = parser.parse_args()
    root = Path(args.output_dir)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError('Use a new toy-data directory.')
    rng = np.random.default_rng(args.seed)
    for s in range(3):  # each fake person = a bright square in a different place
        folder = root / f'yaleB{s + 11}'
        folder.mkdir(parents=True, exist_ok=True)
        for j in range(24):
            pixels = rng.integers(0, 40, (168, 192), dtype=np.uint8)
            pixels[40 * s + 20:40 * s + 70, 60:130] = 220
            name = f"yaleB{s + 11}_P0{j % 9}A{'+' if j % 2 else '-'}{j * 5:03d}E+{j % 4 * 20:02d}.pgm"
            Image.fromarray(pixels).save(folder / name)
    print(f'Toy images written to {root}; do not report these as project results.')


if __name__ == '__main__':
    main()
