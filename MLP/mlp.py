import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision
from torchvision import datasets
import os
#import time

class ImageFolder(datasets.ImageFolder):
    def __init__(self, *args, **kwargs):
        super(ImageFolder, self).__init__(*args, **kwargs)
        transform = []
        transform.append(torchvision.transforms.Grayscale())
        transform.append(torchvision.transforms.ToTensor())
        transform.append(torchvision.transforms.Normalize(mean=[0.2], std=[0.1]))
        self.transform = torchvision.transforms.Compose(transform)
        

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        # (batch, 28*28)
        self.fc1 = nn.Sequential(
            nn.Linear(28 * 28, 2000),
            nn.LeakyReLU()
        )
        # (batch, 2000)
        self.fc2 = nn.Sequential(
            nn.Linear(2000, 1000),
            nn.LeakyReLU()
        )
        # (batch, 1000)
        self.fc3 = nn.Sequential(
            nn.Linear(1000, 100),
            nn.LeakyReLU()
        )
        # (batch, 100)
        self.fc4 = nn.Sequential(
            nn.Linear(100, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x

class MyNet():
    def __init__(self, checkpoint_dir, device, dataset, num_workers, batch_size, epochs):
        """
        定义训练时需要用到的一些参数
        """
        self.mlp = MLP().to(device)
        self.dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            pin_memory=True
        )
        self.epochs = epochs
        self.device = device
        self.optimizer = torch.optim.Adam(params=self.mlp.parameters(), lr=1e-4)
        self.checkpoint_dir = checkpoint_dir
        path = os.path.join(self.checkpoint_dir, 'log.txt')
        with open(path, 'w') as f:
            f.write('')
       
    def save_model(self, epoch):
        """
        将model保存到self.checkpoint_dir中
        """
        save_dir = os.path.join(self.checkpoint_dir, 'models')
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        mlp_dir = os.path.join(save_dir, 'mlp_{}.pth'.format(epoch))
        opt_dir = os.path.join(save_dir, 'opt_{}.pth'.format(epoch))
        torch.save(self.mlp.state_dict(), mlp_dir)
        torch.save(self.optimizer.state_dict(), opt_dir)
        self.logging(content='Save models to {}, epochs: {}.\n'.format(save_dir, epoch))

    def logging(self, content):
        """
        记录信息
        """
        print(content, end='')
        path = os.path.join(self.checkpoint_dir, 'log.txt')
        with open(path, 'a') as f:
            f.write(content)

    def train(self):
        """
        训练模型
        """
        criterion = nn.CrossEntropyLoss()
        for epoch in range(self.epochs):
            acc_loss = 0
            for _, (x, label) in enumerate(self.dataloader):
                x = Variable(x).to(self.device)
                real_label = Variable(label).to(self.device)
                output = self.mlp(x)
                loss = criterion(output, real_label) # 计算loss
                self.optimizer.zero_grad() # 清空上一步的残余更新参数值
                loss.backward() # 反向传播
                self.optimizer.step() # 更新参数
                acc_loss += loss
            acc_loss /= len(self.dataloader)
            self.logging(content='epoch: {}, loss = {}\n'.format(epoch, acc_loss))
            if epoch % 100 == 0:
                self.save_model(epoch)

def get_arg_parser():
    """
    定义parser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', help='the directory of the input images', type=str, default='./mnist/train')
    parser.add_argument('--batch_size', help='batch size of data loader', type=int, default=32)
    parser.add_argument('--num_workers', help='number workers of data loader', type=int, default=0)
    parser.add_argument('--epochs', help='model train epochs', type=int, default=10000)
    parser.add_argument('--gpu', help='whether to use gpu to train', action='store_const', const=True, default=False)
    # parser.add_argument('--resume', help='resume from a checkpoint', action='store_const', const=True, default=False)
    parser.add_argument('--checkpoint_dir', help='directory of checkpoint', type=str, default='./checkpoints')
    # parser.add_argument('--optimizer_dir', help='resume optimizer from file', type=str, default=None)
    # parser.add_argument('--mlp_dir', help='resume mlp from file', type=str, default=None)
    return parser


def main():
    parser = get_arg_parser()
    args = parser.parse_args()
    train_dataset = ImageFolder(args.input_dir)
    device = torch.device(0) if args.gpu else torch.device('cpu')
    print('Training on ' + str(device) + '.')
    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)
    my_net = MyNet(
        checkpoint_dir=args.checkpoint_dir,
        device=device,
        dataset=train_dataset,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        epochs=args.epochs
    )
    my_net.train()
#    if args.optimizer_dir:
#        my_net.optimizer.load_state_dict(args.optimizer_dir)
        
if __name__ == '__main__':
     main()
