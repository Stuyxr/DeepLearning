import argparse
from mlp import MLP
from mlp import ImageFolder
import torch
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', help='test files directory', type=str, default='./mnist/test')
    parser.add_argument('--model_dir', help='model file directory', type=str, default='./checkpoints/models/mlp_200.pth')
    args = parser.parse_args()
    dataset = ImageFolder(args.input_dir)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=10000,
        shuffle=True,
        pin_memory=True
    )
    mlp = MLP()
    print("Loading the generator weights from: ", args.model_dir)
    mlp.load_state_dict(torch.load(args.model_dir, map_location=torch.device('cpu')))
    for step, (x, label) in enumerate(dataloader):
        predict = mlp(x)
        predict = torch.argmax(predict, dim=1)
        right = int((predict == label).sum(0))
        total = len(predict)
        acc = right / total
        print('accuracy: {}/{} = {}%'.format(right, total, acc * 100))

if __name__ == '__main__':
    main()
