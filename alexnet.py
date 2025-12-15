import argparse
import torch

import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torch.distributed as dist
import torch.nn as nn

from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser(description='PyTorch AlexNet Training')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                            'batch size of all GPUs on the current node when '
                            'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    args = parser.parse_args()
    dist.init_process_group(backend='nccl', init_method=args.dist_url, world_size=args.world_size, rank=args.rank)

    alexnet = models.alexnet(pretrained=True)
    torch.cuda.set_device(args.gpu)
    alexnet.cuda(args.gpu)
    alexnet = torch.nn.parallel.DistributedDataParallel(alexnet, device_ids=[args.gpu])

    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    optimizer = torch.optim.SGD(alexnet.parameters(), lr=1e-4, weight_decay=1e-2, momentum=0.9)

    transform = transforms.Compose([
        transforms.RandomResizedCrop((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    cifar10_dataset = torchvision.datasets.CIFAR10(root='./data',
                                        train=False,
                                        transform=transform,
                                        target_transform=None,
                                        download=True)
    train_sampler = torch.utils.data.distributed.DistributedSampler(cifar10_dataset)                                
    dataloader = DataLoader(dataset=cifar10_dataset, sampler=train_sampler, batch_size=32)

    for _ in range(100):
        for item in dataloader: 
            output = alexnet(item[0])
            target = item[1]
            loss = criterion(output, target)
            alexnet.zero_grad()
            loss.backward()
            optimizer.step()


if __name__ == "__main__":
    main()