"""Eigenfaces baseline (PCA + 1-nearest-neighbour) on the same split, scored under the same conditions."""
import argparse
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import utils
from evaluate import report
from model.data_loader import load_manifest, YaleBDataset, CONDITIONS


def flat(images, condition):
    return condition(images.float().div(255)).reshape(len(images), -1).numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_dir', default='data/yaleb')
    parser.add_argument('--model_dir', default='experiments/pca')
    parser.add_argument('--split', default='test', choices=['val', 'test'])
    args = parser.parse_args()
    params = utils.Params(Path(args.model_dir) / 'params.json')
    manifest = load_manifest(args.data_dir)
    train = YaleBDataset(manifest['splits']['train'], manifest['classes'], params, None)
    target = YaleBDataset(manifest['splits'][args.split], manifest['classes'], params, None)
    pca = PCA(n_components=params.n_components, svd_solver='randomized', random_state=params.seed)
    knn = KNeighborsClassifier(n_neighbors=1).fit(pca.fit_transform(flat(train.images, CONDITIONS['clean'])),
                                                  train.labels)
    print(f'PCA {params.n_components} components, explained variance {pca.explained_variance_ratio_.sum():.3f}')
    preds = {name: knn.predict(pca.transform(flat(target.images, fn))) for name, fn in CONDITIONS.items()}
    report(preds, target.labels, target.records, target.classes, Path(args.model_dir), args.split)


if __name__ == '__main__':
    main()
