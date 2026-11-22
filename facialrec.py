"""FacialRec (ARTE 4110U, Group 13): ResNet-18 vs PCA (eigenfaces) on Extended Yale B,
evaluated under fixed illumination changes and synthetic occlusions.

  python facialrec.py pca     --out runs/pca            (add --data DIR to skip the Kaggle download)
  python facialrec.py train   --out runs/base
  python facialrec.py train   --out runs/aug --augment
  python facialrec.py predict --ckpt runs/aug/best.pt some_face.pgm

Research question = compare runs/base vs runs/aug results.json (clean vs challenging conditions).
"""
import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from torchvision import models
from torchvision import transforms as T

H, W = 168, 192  # this Kaggle copy is landscape: 192 wide x 168 tall
SEED = 0
BATCH = 64
NAME = re.compile(r"yaleB(\d+)_P(\d+)A([+-]\d+)E([+-]\d+)", re.I)
SUBSET_BOUNDS = [12, 25, 50, 77]  # Lee/Ho/Kriegman 2005 lighting subsets 1-5 (degrees off camera axis)
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)  # ImageNet stats for the pretrained ResNet
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ---------- data ----------
def light_angle(az, el):
    return math.degrees(math.acos(math.cos(math.radians(az)) * math.cos(math.radians(el))))


def subset(angle):
    return 1 + sum(angle > b for b in SUBSET_BOUNDS)


def scan(root):
    """Preprocessing step 1: inspect files, labels, dimensions; drop unreadable/duplicate entries."""
    recs, seen, sizes = [], set(), Counter()
    skipped = Counter()
    for p in sorted(Path(root).rglob("*")):
        if p.suffix.lower() not in {".pgm", ".png", ".jpg", ".jpeg", ".bmp"}:
            continue
        m = NAME.search(p.name)
        if not m:  # e.g. *_Ambient.pgm has no light direction
            skipped["unparsed"] += 1
            continue
        try:
            with Image.open(p) as im:
                sizes[im.size] += 1
                im.verify()
        except Exception:
            skipped["unreadable"] += 1
            continue
        digest = hashlib.md5(p.read_bytes()).hexdigest()
        if digest in seen:
            skipped["duplicate"] += 1
            continue
        seen.add(digest)
        s, pose, az, el = m.groups()
        recs.append(dict(path=str(p), subject=f"yaleB{s}", pose=int(pose),
                         angle=light_angle(int(az), int(el))))
    if not recs:
        raise SystemExit(f"No Yale B images found under {root}")
    n_subj = len({r['subject'] for r in recs})
    print(f"{len(recs)} images, {n_subj} subjects, sizes {dict(sizes)}, skipped {dict(skipped)}")
    return recs


def split(recs):
    """70/15/15, stratified so every person appears in every split (identity classification)."""
    idx = np.arange(len(recs))
    y = [r["subject"] for r in recs]
    tr, rest = train_test_split(idx, train_size=0.70, stratify=y, random_state=SEED)
    va, te = train_test_split(rest, train_size=0.50, stratify=[y[i] for i in rest], random_state=SEED)
    return torch.from_numpy(tr), torch.from_numpy(va), torch.from_numpy(te)


def load(recs):
    """Step 2: resize to H x W. Returns uint8 tensor (N, H, W) held in RAM (~0.5 GB for 16k images)."""
    return torch.from_numpy(np.stack([np.asarray(Image.open(r["path"]).convert("L").resize((W, H)))
                                      for r in recs]))


def prep(xb, dev):
    return xb.to(dev).float().div(255).unsqueeze(1)  # (B,1,H,W) in [0,1]


def to_net(x):
    """Steps 3-4: repeat grayscale to 3 channels, ImageNet normalisation."""
    return (x.repeat(1, 3, 1, 1) - MEAN.to(x.device)) / STD.to(x.device)


ERASE = T.RandomErasing(p=0.5, scale=(0.02, 0.25), ratio=(0.3, 3.3), value=0)


def augment(x):
    """Step 5: per-image random brightness + random black rectangle occlusion."""
    x = (x * torch.empty(len(x), 1, 1, 1, device=x.device).uniform_(0.2, 1.5)).clamp(0, 1)
    return torch.stack([ERASE(xi) for xi in x])


def occlude(x, r0, r1, c0, c1):
    x = x.clone()
    x[..., int(r0 * H):int(r1 * H), int(c0 * W):int(c1 * W)] = 0
    return x


# Step 6: fixed, reproducible robustness conditions (band positions approximate the cropped Yale faces).
CONDITIONS = {
    "clean": lambda x: x,
    "dark": lambda x: x * 0.3,
    "bright": lambda x: (x * 1.8).clamp(0, 1),
    "occ_eyes": lambda x: occlude(x, 0.30, 0.50, 0.0, 1.0),
    "occ_mouth": lambda x: occlude(x, 0.68, 0.88, 0.15, 0.85),
    "occ_left_half": lambda x: occlude(x, 0.0, 1.0, 0.0, 0.5),
}


# ---------- model ----------
def build_model(n_classes, pretrained=True):
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    m.fc = torch.nn.Linear(m.fc.in_features, n_classes)  # GAP -> FC to one output per person
    return m


@torch.no_grad()
def predict(model, X, dev, cond=CONDITIONS["clean"]):
    model.eval()
    return torch.cat([model(to_net(cond(prep(xb, dev)))).argmax(1).cpu() for xb in X.split(256)])


def fit(model, X, Y, tr, va, epochs, use_aug, dev, ckpt_path, classes):
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    best = -1.0
    for ep in range(epochs):
        model.train()
        total = 0.0
        for b in torch.randperm(len(tr)).split(BATCH):
            i = tr[b]
            x = prep(X[i], dev)
            if use_aug:
                x = augment(x)
            loss = F.cross_entropy(model(to_net(x)), Y[i].to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(i)
        acc = (predict(model, X[va], dev) == Y[va]).float().mean().item()
        print(f"epoch {ep + 1}/{epochs}  loss {total / len(tr):.4f}  val_acc {acc:.4f}")
        if acc > best:
            best = acc
            torch.save({"model": model.state_dict(), "classes": classes}, ckpt_path)
    return best


# ---------- evaluation ----------
def report(preds, Y, recs, classes, out):
    """Accuracy overall / per lighting subset / per pose / per person, plus plots."""
    out.mkdir(parents=True, exist_ok=True)
    y = Y.numpy()
    sub = np.array([subset(r["angle"]) for r in recs])
    pose = np.array([r["pose"] for r in recs])
    res = {}
    plt.figure(figsize=(7, 4))
    for c, p in preds.items():
        ok = p.numpy() == y
        res[c] = {
            "accuracy": ok.mean(),
            "by_lighting_subset": {int(s): ok[sub == s].mean() for s in np.unique(sub)},
            "by_pose": {int(s): ok[pose == s].mean() for s in np.unique(pose)},
            "by_subject": {classes[k]: ok[y == k].mean() for k in np.unique(y)},
        }
        xs = sorted(res[c]["by_lighting_subset"])
        plt.plot(xs, [res[c]["by_lighting_subset"][s] for s in xs], marker="o", label=c)
        print(f"  {c:14s} acc {ok.mean():.4f}")
    plt.xlabel("Lighting subset (1 = frontal light ... 5 = extreme)")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.02)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "accuracy_vs_illumination.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 10))
    ConfusionMatrixDisplay.from_predictions(y, preds["clean"].numpy(), labels=range(len(classes)),
                                            display_labels=classes, ax=ax, colorbar=False,
                                            include_values=False, xticks_rotation=90)
    fig.tight_layout()
    fig.savefig(out / "confusion_clean.png", dpi=150)
    plt.close(fig)
    (out / "results.json").write_text(json.dumps(res, indent=2))
    print(f"wrote {out}/results.json, accuracy_vs_illumination.png, confusion_clean.png")
    return res


def setup(data):
    if data is None:  # download once from Kaggle; kagglehub caches it for later runs
        import kagglehub
        data = kagglehub.dataset_download("jensdhondt/extendedyaleb-cropped-full")
        print("Path to dataset files:", data)
    recs = scan(data)
    classes = sorted({r["subject"] for r in recs})
    Y = torch.tensor([classes.index(r["subject"]) for r in recs])
    return recs, classes, Y, load(recs), *split(recs)


def cmd_train(a):
    recs, classes, Y, X, tr, va, te = setup(a.data)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model = build_model(len(classes)).to(dev)
    print(f"device {dev}, params {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M, augment={a.augment}")
    fit(model, X, Y, tr, va, a.epochs, a.augment, dev, out / "best.pt", classes)
    model.load_state_dict(torch.load(out / "best.pt", map_location=dev)["model"])
    print("test set:")
    report({c: predict(model, X[te], dev, f) for c, f in CONDITIONS.items()},
           Y[te], [recs[i] for i in te], classes, out)


def cmd_pca(a):
    recs, classes, Y, X, tr, va, te = setup(a.data)
    flat = lambda x: x.reshape(len(x), -1).numpy()
    model = PCA(n_components=a.k, svd_solver="randomized", random_state=SEED)
    knn = KNeighborsClassifier(n_neighbors=1).fit(model.fit_transform(flat(prep(X[tr], "cpu"))), Y[tr].numpy())
    print(f"PCA {a.k} components, explained variance {model.explained_variance_ratio_.sum():.3f}")
    print("test set:")
    preds = {c: torch.from_numpy(knn.predict(model.transform(flat(f(prep(X[te], "cpu"))))))
             for c, f in CONDITIONS.items()}
    report(preds, Y[te], [recs[i] for i in te], classes, Path(a.out))


@torch.no_grad()
def cmd_predict(a):
    """Interface: show the predicted identity as a label over the image."""
    ck = torch.load(a.ckpt, map_location="cpu")
    model = build_model(len(ck["classes"]), pretrained=False)
    model.load_state_dict(ck["model"])
    img = Image.open(a.image).convert("L").resize((W, H))
    prob = model.eval()(to_net(prep(torch.from_numpy(np.asarray(img))[None], "cpu"))).softmax(1)[0]
    conf, k = prob.max(0)
    label = f"{ck['classes'][k]} ({conf:.0%})"
    print(label)
    plt.imshow(img, cmap="gray")
    plt.text(4, 14, label, color="yellow", fontsize=12, backgroundcolor="black")
    plt.axis("off")
    plt.savefig(Path(a.image).with_suffix(".pred.png"), bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    t = sp.add_parser("train")
    t.add_argument("--data", help="dataset folder (default: download from Kaggle)")
    t.add_argument("--out", required=True)
    t.add_argument("--epochs", type=int, default=15)
    t.add_argument("--augment", action="store_true", help="targeted brightness + occlusion augmentation")
    t.set_defaults(fn=cmd_train)
    p = sp.add_parser("pca")
    p.add_argument("--data", help="dataset folder (default: download from Kaggle)")
    p.add_argument("--out", required=True)
    p.add_argument("--k", type=int, default=150, help="number of eigenfaces")
    p.set_defaults(fn=cmd_pca)
    q = sp.add_parser("predict")
    q.add_argument("--ckpt", required=True)
    q.add_argument("image")
    q.set_defaults(fn=cmd_predict)
    args = ap.parse_args()
    args.fn(args)
