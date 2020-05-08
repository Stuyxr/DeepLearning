import argparse
from cnn import AlexNet
import torch
from calculate_accuracy import calculate_accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', help='test files directory', type=str, default='./data/test')
    parser.add_argument('--model_dir', help='model file directory', type=str, default='./checkpoints/models/cnn_250.pth')
    args = parser.parse_args()
    alexnet = AlexNet().cuda()
    alexnet.eval()
    print("Loading the generator weights from: ", args.model_dir)
    alexnet.load_state_dict(torch.load(args.model_dir))
    accuracy = calculate_accuracy(path=args.input_dir, net=alexnet, gpu=True, num_workers=8)
    print('accuracy = {}'.format(accuracy))

if __name__ == '__main__':
    main()
