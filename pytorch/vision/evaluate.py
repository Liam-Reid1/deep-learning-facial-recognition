"""Evaluate a selected checkpoint on clean images and under fixed lighting/occlusion conditions.

Use validation for selection; reserve test for final evaluation.
Writes metrics_<split>_<restore>.json (clean images) plus
robustness_<split>.json, accuracy_vs_illumination_<split>.png,
confusion_clean_<split>.png, and learning_curve.png.
"""
import argparse
import ast
import json
from pathlib import Path
import re
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


def _find_log_file(model_dir):
    """Find matching .log file inside model_dir or sibling to it (e.g. runs/base -> runs/base.log)."""
    out = Path(model_dir).resolve()
    candidates = [
        out / f"{out.name}.log",
        out.parent / f"{out.name}.log",
    ] + sorted(out.glob("*.log"))
    for path in candidates:
        if path.exists() and path.is_file():
            return path
    return None


def _load_history_records(model_dir):
    """Load epoch metrics from JSON history or .log files (ignoring download/progress lines)."""
    out = Path(model_dir)

    # 1. Check JSON / JSONL history files if present
    for fname in ["history.jsonl", "history.json", "train_history.jsonl", "train_history.json"]:
        path = out / fname
        if path.exists() and path.stat().st_size > 0:
            text = path.read_text(errors="ignore").strip()
            try:
                data = json.loads(text)
                return data if isinstance(data, list) else [data]
            except json.JSONDecodeError:
                return [json.loads(line) for line in text.splitlines() if line.strip()]

    # 2. Parse .log file (e.g., runs/base.log or runs/aug.log)
    log_path = _find_log_file(model_dir)
    if log_path is None:
        return None

    # Format A: epoch 1/15  loss 0.3137  val_acc 0.9723
    simple_pat = re.compile(
        r"epoch\s+(\d+)/\d+\s+loss\s+([0-9.eE+-]+)\s+val_acc\s+([0-9.eE+-]+)",
        re.IGNORECASE,
    )
    # Format B: Epoch 1/15 train={'loss': ..., 'accuracy': ...} val={'loss': ..., 'accuracy': ...}
    dict_pat = re.compile(r"Epoch\s+(\d+)/\d+\s+train=(\{.*?\})\s+val=(\{.*?\})")

    records_by_epoch = {}
    for line in log_path.read_text(errors="ignore").splitlines():
        m_simple = simple_pat.search(line)
        if m_simple:
            epoch = int(m_simple.group(1))
            records_by_epoch[epoch] = {
                "epoch": epoch,
                "train_loss": float(m_simple.group(2)),
                "val_accuracy": float(m_simple.group(3)),
            }
            continue

        m_dict = dict_pat.search(line)
        if m_dict:
            epoch = int(m_dict.group(1))
            train_dict = ast.literal_eval(m_dict.group(2))
            val_dict = ast.literal_eval(m_dict.group(3))
            records_by_epoch[epoch] = {
                "epoch": epoch,
                "train_loss": train_dict.get("loss"),
                "train_accuracy": train_dict.get("accuracy"),
                "val_loss": val_dict.get("loss"),
                "val_accuracy": val_dict.get("accuracy"),
            }

    if records_by_epoch:
        return [records_by_epoch[ep] for ep in sorted(records_by_epoch)]
    return None


def plot_learning_curve(model_dir, selected_epoch=None):
    """Plot training loss and validation accuracy curves."""
    records = _load_history_records(model_dir)
    if not records:
        print(f"No epoch records found for {model_dir}; skipping learning curve.")
        return

    epochs = [r["epoch"] for r in records]
    fig, (loss_ax, acc_ax) = plt.subplots(1, 2, figsize=(12, 4))

    # Left panel: Loss
    if any(r.get("train_loss") is not None for r in records):
        loss_ax.plot(epochs, [r["train_loss"] for r in records], marker="o", markersize=4, label="Train Loss")
    if any(r.get("val_loss") is not None for r in records):
        loss_ax.plot(epochs, [r["val_loss"] for r in records], marker="o", markersize=4, label="Val Loss")
    if selected_epoch is not None:
        loss_ax.axvline(
            selected_epoch,
            color="gray",
            linestyle="--",
            alpha=0.7,
            label=f"Checkpoint (ep {selected_epoch})",
        )
    loss_ax.set_title("Loss Curve")
    loss_ax.set_xlabel("Epoch")
    loss_ax.set_ylabel("Loss")
    loss_ax.legend()
    loss_ax.grid(True, alpha=0.3)

    # Right panel: Accuracy
    if any(r.get("train_accuracy") is not None for r in records):
        acc_ax.plot(epochs, [r["train_accuracy"] for r in records], marker="o", markersize=4, label="Train Acc")
    if any(r.get("val_accuracy") is not None for r in records):
        acc_ax.plot(epochs, [r["val_accuracy"] for r in records], marker="o", markersize=4, label="Val Acc")
    if selected_epoch is not None:
        acc_ax.axvline(
            selected_epoch,
            color="gray",
            linestyle="--",
            alpha=0.7,
            label=f"Checkpoint (ep {selected_epoch})",
        )
    acc_ax.set_title("Accuracy Curve")
    acc_ax.set_xlabel("Epoch")
    acc_ax.set_ylabel("Accuracy")
    acc_ax.set_ylim(0, 1.02)
    acc_ax.legend()
    acc_ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = Path(model_dir)
    fig.savefig(out / "learning_curve.png", dpi=150)
    plt.close(fig)
    print(f"wrote learning_curve.png in {out}")


def report(preds, labels, records, classes, out, split):
    """Accuracy overall / per lighting subset / per pose / per person, plus plots."""
    out.mkdir(parents=True, exist_ok=True)
    y = np.asarray(labels)
    sub = np.array([subset(r["angle"]) for r in records])
    pose = np.array([r["pose"] for r in records])
    res = {}

    plt.figure(figsize=(7, 4))
    for c, p in preds.items():
        ok = np.asarray(p) == y
        res[c] = {
            "accuracy": ok.mean(),
            "by_lighting_subset": {int(s): ok[sub == s].mean() for s in np.unique(sub)},
            "by_pose": {int(s): ok[pose == s].mean() for s in np.unique(pose)},
            "by_subject": {classes[k]: ok[y == k].mean() for k in np.unique(y)},
        }
        xs = sorted(res[c]["by_lighting_subset"])
        plt.plot(xs, [res[c]["by_lighting_subset"][s] for s in xs], marker="o", label=c)
        print(f"{c:14s} acc {ok.mean():.4f}")

    plt.xlabel("Lighting subset (1 = frontal light ... 5 = extreme)")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.02)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / f"accuracy_vs_illumination_{split}.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 10))
    ConfusionMatrixDisplay.from_predictions(
        y,
        np.asarray(preds["clean"]),
        labels=range(len(classes)),
        display_labels=classes,
        ax=ax,
        colorbar=False,
        include_values=False,
        xticks_rotation=90,
    )
    fig.tight_layout()
    fig.savefig(out / f"confusion_clean_{split}.png", dpi=150)
    plt.close(fig)

    utils.save_dict_to_json(res, out / f"robustness_{split}.json")
    print(
        f"wrote robustness_{split}.json, accuracy_vs_illumination_{split}.png, "
        f"confusion_clean_{split}.png in {out}"
    )
    return res


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", default="data/yaleb")
    parser.add_argument("--model_dir", default="experiments/base_model")
    parser.add_argument("--restore_file", default="best", choices=["best", "last"])
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    checkpoint = Path(args.model_dir) / (args.restore_file + ".pth.tar")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)

    params = utils.Params(Path(args.model_dir) / "params.json")
    params.__dict__.update(state["params"])  # Architecture must match the trained checkpoint.
    params.device = str(utils.get_device(args.device))
    params.pretrained = False  # weights come from the checkpoint
    utils.seed_everything(params.seed)
    loaders = fetch_dataloader([args.split], args.data_dir, params)
    batches = loaders[args.split]
    model = Net(params).to(params.device)
    model.load_state_dict(state["state_dict"])

    metrics = utils.run_epoch(model, loss_fn, batches, params.device)
    metrics.update(epoch=state["epoch"], split=args.split)
    destination = Path(args.model_dir) / f"metrics_{args.split}_{args.restore_file}.json"
    utils.save_dict_to_json(metrics, destination)
    print(metrics)
    dataset = batches.dataset
    preds = {}
    for name in CONDITIONS:
        dataset.transform = make_transform(params, condition=name)
        preds[name] = predictions(model, batches, params.device)
    report(preds, dataset.labels, dataset.records, dataset.classes, Path(args.model_dir), args.split)
    plot_learning_curve(args.model_dir, selected_epoch=state.get("epoch"))


if __name__ == "__main__":
    main()
