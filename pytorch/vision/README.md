# FacialRec: Face Recognition under Lighting Changes and Occlusion (Group 13)

A pretrained ResNet-18 with a 28-way output (one per person) vs a PCA/eigenfaces baseline on
Extended Yale B (Cropped, Full). Research question: does targeted augmentation (random brightness
+ rectangle occlusion) improve recognition under hard conditions without hurting clean accuracy?
Run all commands from `pytorch/vision` after installing the root requirements.

## Try the code without a download

```bash
python make_toy_dataset.py --output_dir data/toy_raw
python build_dataset.py --data_dir data/toy_raw --output_dir data/toy
python train.py --data_dir data/toy --model_dir experiments/base_model --device cpu
python evaluate.py --data_dir data/toy --model_dir experiments/base_model --split val --device cpu
python predict.py --image data/toy_raw/yaleB11/yaleB11_P00A-000E+00.pgm --model_dir experiments/base_model --device cpu
```

The toy images are synthetic squares (3 fake people) for checking the pipeline only. Delete
`experiments/base_model` outputs (or use `--overwrite`) before training on the real data.

## Prepare Extended Yale B

```bash
python build_dataset.py --output_dir data/yaleb            # downloads via kagglehub (cached)
python build_dataset.py --data_dir RAW_DIR --output_dir data/yaleb   # or use a local copy
```

This checks every file (labels from filenames such as `yaleB11_P00A+000E+00.pgm`, sizes,
unreadable files, duplicates; the 252 `_Ambient` images have no light direction and are skipped)
and writes `data/yaleb/split_manifest.json`: a reproducible 70/15/15 split stratified by person,
so every identity appears in train, val and test. Images are not copied.

## Run experiments

```bash
python pca_baseline.py                                     # baseline -> experiments/pca
python train.py --model_dir experiments/base_model         # ResNet-18, no augmentation
python train.py --model_dir experiments/augmented          # ResNet-18 + targeted augmentation
python synthesize_results.py --parent_dir experiments      # validation summary
python evaluate.py --model_dir experiments/base_model --split test
python evaluate.py --model_dir experiments/augmented --split test
python predict.py --image some_face.pgm --model_dir experiments/augmented
```

`evaluate.py` and `pca_baseline.py` score six fixed conditions (`clean`, `dark`, `bright`,
`occ_eyes`, `occ_mouth`, `occ_left_half`) and write `robustness_<split>.json` (accuracy overall,
per lighting subset 1-5, per pose, per person), `accuracy_vs_illumination_<split>.png` and
`confusion_clean_<split>.png`. Compare `experiments/base_model` with `experiments/augmented`
to answer the research question.

`python search_hyperparams.py --parent_dir experiments/learning_rate` sweeps the learning rate.

## Where things live

- `model/data_loader.py`: manifest loading, preprocessing, augmentation, robustness conditions.
- `model/net.py`: ResNet-18 with global average pooling and a 28-way FC layer.
- `train.py` / `evaluate.py` / `predict.py`: training with validation checkpointing, test evaluation,
  single-image prediction (saves `<image>.pred.png` with the identity drawn on it).
- `pca_baseline.py`: eigenfaces + 1-nearest-neighbour.
- `experiments/*/params.json`: one configuration per run.
