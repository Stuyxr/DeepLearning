import torch
from torch.autograd import Variable
from torchvision import datasets
import torchvision
from torch.utils.data import DataLoader

class ImageFolder(datasets.ImageFolder):
    def __init__(self, *args, **kwargs):
        super(ImageFolder, self).__init__(*args, **kwargs)
        transform = []
        transform.append(torchvision.transforms.RandomHorizontalFlip())
        transform.append(torchvision.transforms.Resize(size=32))
        transform.append(torchvision.transforms.ToTensor())
        transform.append(torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
        self.transform = torchvision.transforms.Compose(transform)

def calculate_accuracy(path, net, gpu=True, num_workers=2):
    dataset = ImageFolder(path)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=100,
        num_workers=num_workers,
        shuffle=False
    )
    acc = 0
    for step, (x, label) in enumerate(dataloader):
        x = Variable(x).cuda() if gpu else Variable(x)
        predict = net(x).cpu()
        predict = torch.argmax(predict, dim=1)
        right = int((predict == label).sum(0))
        acc += 1.0 * right / len(predict)
    return acc / len(dataloader)
