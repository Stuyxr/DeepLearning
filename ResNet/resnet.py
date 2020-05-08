import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision
from torchvision import datasets
from tensorboardX import SummaryWriter
import os
from calculate_accuracy import calculate_accuracy


class ImageFolder(datasets.ImageFolder):
    def __init__(self, *args, **kwargs):
        super(ImageFolder, self).__init__(*args, **kwargs)
        transform = []
        transform.append(torchvision.transforms.RandomHorizontalFlip())
        transform.append(torchvision.transforms.Resize(size=32))
        transform.append(torchvision.transforms.ToTensor())
        transform.append(torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
        self.transform = torchvision.transforms.Compose(transform)
        

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, use_se=False):
        """
        stride=1时，尺寸不变
        stride=2时，尺寸变为一半（虚线）
        """
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, 2, 0),
                nn.BatchNorm2d(out_channels)
            )
        self.use_se = use_se
        if self.use_se:
            self.global_pool = nn.AdaptiveAvgPool2d(1)
            self.conv_down = nn.Conv2d(out_channels, out_channels // 16, kernel_size=1, bias=False)
            self.conv_up = nn.Conv2d(out_channels // 16, out_channels, kernel_size=1, bias=False)
            self.sig = nn.Sigmoid()
    
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        if self.use_se:
            out1 = self.global_pool(out)
            out1 = self.conv_down(out1)
            out1 = self.relu(out1)
            out1 = self.conv_up(out1)
            out1 = self.sig(out1)
            out = out1 * out
        out += identity
        out = self.relu(out)
        return out

    
class ResNet(nn.Module):
    def __init__(self, blocks_num=[3, 4, 6, 3], use_se=False):
        super(ResNet, self).__init__()
        # batch, 3, 32 ,32
        self.first = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 1, 0)
        )
        self.use_se = use_se
        self.layer1 = self.make_layer(64, 64, blocks_num[0], 1)
        self.layer2 = self.make_layer(64, 128, blocks_num[1], 2)
        self.layer3 = self.make_layer(128, 256, blocks_num[2], 2)
        self.layer4 = self.make_layer(256, 512, blocks_num[3], 2)
        self.average_pool = nn.AvgPool2d(4)
        self.fc = nn.Linear(512, 10)
    
    def make_layer(self, in_channels, out_channels, block_num, stride):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride, self.use_se))
        for _ in range(block_num - 1):
            layers.append(BasicBlock(out_channels, out_channels, 1, self.use_se))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.first(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.average_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x



class MyNet():
    def __init__(self, checkpoint_dir, device, train_dataset, show_test_accuracy, test_dataset_dir, num_workers, batch_size, epochs, use_se):
        """
        checkpoint_dir: 断点保存路径
        device: cpu or gpu
        train_dataset: 训练集
        show_test_accuracy: 训练过程中是否要跑测试集
        test_dataset_dir: 测试集路径
        num_workers: dataloader参数，cpu个数
        batch_size: dataloader参数，批大小
        epochs: 训练时的epochs
        """
        self.net = ResNet(use_se=use_se).to(device)
        self.net.train()
        if not os.path.exists('./writer'):
            os.mkdir('./writer')
        self.writer = SummaryWriter('./writer/')
        self.dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            pin_memory=True
        )
        self.test_dataset_dir = test_dataset_dir
        self.show_test_accuracy = show_test_accuracy
        self.epochs = epochs
        self.device = device
        self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=1e-4)
        self.checkpoint_dir = checkpoint_dir
        path = os.path.join(self.checkpoint_dir, 'log.txt')
        with open(path, 'w') as f:
            f.write('')
       
    def save_model(self, epoch):
        """
        保存模型
        """
        save_dir = os.path.join(self.checkpoint_dir, 'models')
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        cnn_dir = os.path.join(save_dir, 'resnet_{}.pth'.format(epoch))
        opt_dir = os.path.join(save_dir, 'opt_{}.pth'.format(epoch))
        torch.save(self.net.state_dict(), cnn_dir)
        torch.save(self.optimizer.state_dict(), opt_dir)
        self.logging(content='Save models to {}, epochs: {}.\n'.format(save_dir, epoch))

    def logging(self, content):
        """
        记录log
        """
        print(content, end='')
        path = os.path.join(self.checkpoint_dir, 'log.txt')
        with open(path, 'a') as f:
            f.write(content)

    def train(self):    
        """
        训练
        """
        criterion = nn.CrossEntropyLoss()
        for epoch in range(self.epochs):
            total_loss = 0
            acc_train = 0
            self.net.train()
            for _, (x, label) in enumerate(self.dataloader):
                x = Variable(x).to(self.device)
                real_label = Variable(label).to(self.device)
                output = self.net(x)
                loss = criterion(output, real_label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                predict_label = torch.argmax(output, dim=1)
                right = int((predict_label.cpu() == label).sum(0)) / len(label)
                acc_train += right

            total_loss /= len(self.dataloader)
            acc_train /= len(self.dataloader)

            if self.show_test_accuracy:
                self.net.eval()
                acc_test = calculate_accuracy(path=self.test_dataset_dir, net=self.net)
                self.logging(content='epoch: {}, loss = {}, train_acc = {}, test_acc = {}\n'.format(epoch, total_loss, acc_train, acc_test))
            else:
                self.logging(content='epoch: {}, loss = {}, train_acc = {}\n'.format(epoch, total_loss, acc_train))
            
            if (epoch % 5 == 0 and epoch > 0) or acc_test > 0.8:
                self.save_model(epoch)

            if self.show_test_accuracy:
                self.writer.add_scalar('test_accuracy', acc_test, epoch)            
            self.writer.add_scalar('loss', total_loss, epoch)
            self.writer.add_scalar('train_accuracy', acc_train, epoch)


        
        self.writer.close()

"""
python resnet.py --gpu --show_test_accuracy --batch_size 128 --num_workers 8
"""
def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dataset_dir', help='the directory of the train images', type=str, default='./data/train')
    parser.add_argument('--test_dataset_dir', help='the directory of the test images', type=str, default='./data/test')
    parser.add_argument('--show_test_accuracy', help='whether to show the test accuracy while training', action='store_const', const=True, default=False)
    parser.add_argument('--batch_size', help='batch size of data loader', type=int, default=32)
    parser.add_argument('--num_workers', help='number workers of data loader', type=int, default=0)
    parser.add_argument('--epochs', help='model train epochs', type=int, default=10000)
    parser.add_argument('--cpu', help='whether to use gpu to train', action='store_const', const=True, default=False)
    parser.add_argument('--resume', help='resume from a checkpoint', action='store_const', const=True, default=False)
    parser.add_argument('--checkpoint_dir', help='directory of checkpoint', type=str, default='./checkpoints')
    parser.add_argument('--optimizer_dir', help='resume optimizer from file', type=str, default='./checkpoints/models/opt_150.pth')
    parser.add_argument('--resnet_dir', help='resume resnet from file', type=str, default='./checkpoints/models/cnn_150.pth')
    parser.add_argument('--use_se', help='whether to use se block', action='store_const', const=True, default=False)
    return parser


def main():
    parser = get_arg_parser()
    args = parser.parse_args()
    train_dataset = ImageFolder(args.train_dataset_dir)
    device = torch.device('cpu') if args.cpu else torch.device(0)
    print('Training on ' + str(device) + '.')
    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)

    my_net = MyNet(
        checkpoint_dir=args.checkpoint_dir,
        device=device,
        train_dataset=train_dataset,
        test_dataset_dir=args.test_dataset_dir,
        show_test_accuracy=args.show_test_accuracy,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        epochs=args.epochs,
        use_se=args.use_se
    )
    # 是否从断点恢复训练 
    if args.resume:
        my_net.logging('loading resnet from {}.\n'.format(args.cnn_dir))
        my_net.logging('loading optimizer from {}.\n'.format(args.optimizer_dir))
        my_net.net.load_state_dict(torch.load(args.cnn_dir))
        my_net.optimizer.load_state_dict(torch.load(args.optimizer_dir))

    my_net.train()

if __name__ == '__main__':
    main()






