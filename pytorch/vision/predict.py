"""Predict one face with the saved checkpoint and save it with the identity drawn over the image."""
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import utils
from model.net import Net
from model.data_loader import load_image, make_transform


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', required=True)
    parser.add_argument('--model_dir', default='experiments/augmented')
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    device = utils.get_device(args.device)
    state = torch.load(Path(args.model_dir)/'best.pth.tar', map_location='cpu', weights_only=True)
    params = utils.Params(Path(args.model_dir)/'params.json')
    params.__dict__.update(state['params'])
    params.pretrained = False  # weights come from the checkpoint
    model = Net(params).to(device)
    model.load_state_dict(state['state_dict'])
    model.eval()
    image = load_image(args.image, params)
    tensor = make_transform(params)(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probabilities = model(tensor).softmax(dim=-1)[0].cpu()
    index = int(probabilities.argmax())
    identity = params.classes[index]
    output = Path(args.image).with_suffix('.pred.png')
    print(json.dumps({'class_id': index, 'identity': identity, 'probability': float(probabilities[index]),
                      'labelled_image': str(output)}))
    # Interface (proposal): predicted identity shown as a label over the image.
    plt.imshow(image.numpy(), cmap='gray')
    plt.text(4, 14, f'{identity} ({float(probabilities[index]):.0%})', color='yellow',
             fontsize=12, backgroundcolor='black')
    plt.axis('off')
    plt.savefig(output, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    main()
