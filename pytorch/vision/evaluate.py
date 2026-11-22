"""Evaluate a selected checkpoint on clean images and under fixed lighting/occlusion conditions.

Use validation for selection; reserve test for final evaluation. Writes metrics_<split>_<restore>.json
(clean images) plus robustness_<split>.json, accuracy_vs_illumination_<split>.png, confusion_clean_<split>.png.
"""
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
import torch
import utils
from model.net import Net, loss_fn
from model.data_loader import fetch_dataloader, make_transform, CONDITIONS

SUBSET_BOUNDS = [12, 25, 50, 77]  # Lee/Ho/Kriegman 2005 lighting subsets 1-5 (degrees off camera axis)


def subset(angle):
    return 1 + sum(angle > b for b in SUBSET_BOUNDS)


@torch.no_grad()
def predictions(model, loader, device):
    model.eval()
    return np.concatenate([model(x.to(device)).argmax(1).cpu().numpy() for x, _ in loader])


def report(preds, labels, records, classes, out, split):
    """Accuracy overall / per lighting subset / per pose / per person, plus plots."""
    out.mkdir(parents=True, exist_ok=True)
    y = np.asarray(labels)
    sub = np.array([subset(r['angle']) for r in records])
    pose = np.array([r['pose'] for r in records])
    res = {}
    plt.figure(figsize=(7, 4))
    for c, p in preds.items():
        ok = np.asarray(p) == y
        res[c] = {
            'accuracy': ok.mean(),
            'by_lighting_subset': {int(s): ok[sub == s].mean() for s in np.unique(sub)},
            'by_pose': {int(s): ok[pose == s].mean() for s in np.unique(pose)},
            'by_subject': {classes[k]: ok[y == k].mean() for k in np.unique(y)},
        }
        xs = sorted(res[c]['by_lighting_subset'])
        plt.plot(xs, [res[c]['by_lighting_subset'][s] for s in xs], marker='o', label=c)
        print(f'  {c:14s} acc {ok.mean():.4f}')
    plt.xlabel('Lighting subset (1 = frontal light ... 5 = extreme)')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.02)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / f'accuracy_vs_illumination_{split}.png', dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 10))
    ConfusionMatrixDisplay.from_predictions(y, np.asarray(preds['clean']), labels=range(len(classes)),
                                            display_labels=classes, ax=ax, colorbar=False,
                                            include_values=False, xticks_rotation=90)
    fig.tight_layout()
    fig.savefig(out / f'confusion_clean_{split}.png', dpi=150)
    plt.close(fig)
    utils.save_dict_to_json(res, out / f'robustness_{split}.json')
    print(f'wrote robustness_{split}.json, accuracy_vs_illumination_{split}.png, '
          f'confusion_clean_{split}.png in {out}')
    return res


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_dir', default='data/yaleb')
    parser.add_argument('--model_dir', default='experiments/base_model')
    parser.add_argument('--restore_file', default='best', choices=['best', 'last'])
    parser.add_argument('--split', default='test', choices=['val', 'test'])
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    checkpoint = Path(args.model_dir) / (args.restore_file + '.pth.tar')
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    params = utils.Params(Path(args.model_dir) / 'params.json')
    params.__dict__.update(state['params'])  # Architecture must match the trained checkpoint.
    params.device = str(utils.get_device(args.device))
    params.pretrained = False  # weights come from the checkpoint
    utils.seed_everything(params.seed)
    loaders = fetch_dataloader([args.split], args.data_dir, params)
    batches = loaders[args.split]
    model = Net(params).to(params.device)
    model.load_state_dict(state['state_dict'])
    metrics = utils.run_epoch(model, loss_fn, batches, params.device)
    metrics.update(epoch=state['epoch'], split=args.split)
    destination = Path(args.model_dir) / f'metrics_{args.split}_{args.restore_file}.json'
    utils.save_dict_to_json(metrics, destination)
    print(metrics)
    dataset = batches.dataset
    preds = {}
    for name in CONDITIONS:
        dataset.transform = make_transform(params, condition=name)
        preds[name] = predictions(model, batches, params.device)
    report(preds, dataset.labels, dataset.records, dataset.classes, Path(args.model_dir), args.split)


if __name__ == '__main__':
    main()
