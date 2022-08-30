import sys
sys.path.append("..")
import torch
import torch.utils.data
from torchvision import datasets, transforms
import numpy as np
import argparse
from lib.stackedDAE import StackedDAE
from lib.datasets import MNIST,FASHIONMNIST

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VAE MNIST Example')
    parser.add_argument('--lr', type=float, default=0.1, metavar='N',
                        help='learning rate for training (default: 0.001)')
    parser.add_argument('--lr1',type=float, default=0.08, metavar='N',
                        help='learning rate for training (default: 0.001)')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--pretrainepochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    args = parser.parse_args()

    train_loader = torch.utils.data.DataLoader(
        MNIST('./dataset/mnist', train=True, download=True),
        batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(
        MNIST('./dataset/mnist', train=False),
        batch_size=args.batch_size, shuffle=False, num_workers=0)
    '''train_loader = torch.utils.data.DataLoader(
        FASHIONMNIST('./dataset/fashionmnist', train=True, download=False),
        batch_size=args.batch_size, shuffle=True, num_workers=0)
    #test_loader = torch.utils.data.DataLoader(
        FASHIONMNIST('./dataset/fashionmnist', train=False,download=False),
        batch_size=args.batch_size, shuffle=False, num_workers=0)'''

    sdae = StackedDAE(input_dim=784, z_dim=10, binary=False,
        encodeLayer=[500,500,2000], decodeLayer=[2000,500,500], activation="relu", 
        dropout=0)
    print(sdae)
    import time
    start = time.time()
    sdae.pretrain(train_loader, test_loader, lr=args.lr, batch_size=args.batch_size,
        num_epochs=args.pretrainepochs, corrupt=0.2, loss_type="my_loss")
    sdae.fit(train_loader, test_loader, lr=args.lr1, num_epochs=args.epochs, corrupt=0.2, loss_type="my_loss")
    end = time.time()
    print("runtime: {}".format(end-start))
    #sdae.save_model("./model/mnist-csdae.pt")
