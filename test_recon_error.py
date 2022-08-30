import sys

sys.path.append("..")
import torch
import torch.utils.data
from torchvision import datasets, transforms
import numpy as np
import argparse
from lib.stackedDAE import StackedDAE
from lib.datasets import MNIST, FASHIONMNIST
from torch.autograd import Variable
from lib.ops import MSELoss, BCELoss, my_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VAE MNIST Example')
    parser.add_argument('--lr', type=float, default=0.1, metavar='N',
                        help='learning rate for training (default: 0.001)')
    parser.add_argument('--lr1', type=float, default=0.08, metavar='N',
                        help='learning rate for training (default: 0.001)')
    parser.add_argument('--batch-size', type=int, default=600, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--pretrainepochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of epochs to train (default: 10)')
    args = parser.parse_args()

    # according to the released code, mnist data is multiplied by 0.02
    # 255*0.02 = 5.1. transforms.ToTensor() coverts 255 -> 1.0
    # so add a customized Scale transform to multiple by 5.1
    train_loader = torch.utils.data.DataLoader(
        MNIST('./dataset/mnist', train=True, download=True),
        batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(
        MNIST('./dataset/mnist', train=False),
        batch_size=args.batch_size, shuffle=False, num_workers=0)

    sdae = StackedDAE(input_dim=784, z_dim=10, binary=False,
                      encodeLayer=[500, 500, 2000], decodeLayer=[2000, 500, 500], activation="relu",
                      dropout=0)
    sdae.load_state_dict(torch.load('./model/sdae.pt'))
    sdae.eval()
    criterion = MSELoss()
    total_loss = 0.0
    total_num = 0
    print(len(train_loader))
    for batch_idx, (inputs, _) in enumerate(train_loader):
        inputs = inputs.view(inputs.size(0), -1).float()
        inputs = Variable(inputs)
        z, outputs = sdae.forward(inputs)

        valid_recon_loss = criterion(inputs, outputs)
        total_loss += valid_recon_loss.data * len(inputs)
        total_num += inputs.size()[0]

    valid_loss = total_loss / total_num
    print("#Epoch 0: Valid Reconstruct Loss: %.4f" % (valid_loss*784/5.1))