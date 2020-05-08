import argparse
from resnet import ResNet
import torch
from calculate_accuracy import calculate_accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', help='test files directory', type=str, default='./data/test')
    parser.add_argument('--model_dir', help='model file directory', type=str, default='./checkpoints/models/cnn_58.pth')
    parser.add_argument('--use_se', help='whether to use se block', action='store_const', const=True, default=False)
    args = parser.parse_args()
    net = ResNet(use_se=args.use_se).cuda()
    net.eval()
    print("Loading the generator weights from: ", args.model_dir)
    net.load_state_dict(torch.load(args.model_dir))
    accuracy = calculate_accuracy(path=args.input_dir, net=net, gpu=True, num_workers=8)
    print('accuracy = {}'.format(accuracy))

if __name__ == '__main__':
    main()
