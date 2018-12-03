import torch
from PIL import ImageFile
import torchvision.transforms as transforms
import torchvision.datasets as data
import torchvision.utils as vutils
from torch import nn
from torch import optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES = True

labeled_dataset = "data/UTKFace/labeled"
img_size = 128
batch_size = 32


def converttoage(i):
    if i == 0:
        return 0
    elif i == 1:
        return 1
    elif i == 2:
        return 10
    elif i == 3:
        return 11
    elif i == 4:
        return 12
    elif i == 5:
        return 13
    elif i == 6:
        return 14
    elif i == 7:
        return 15
    elif i == 8:
        return 16
    elif i == 9:
        return 17
    elif i == 10:
        return 18
    elif i == 11:
        return 19
    elif i == 12:
        return 2
    elif i == 13:
        return 3
    elif i == 14:
        return 4
    elif i == 15:
        return 5
    elif i == 16:
        return 6
    elif i == 17:
        return 7
    elif i == 18:
        return 8
    elif i == 19:
        return 9
    else:
        return -1


def get_data_loader():
    dataset = data.ImageFolder(root=labeled_dataset,
                               transform=transforms.Compose([
                                   transforms.Resize(img_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=True)

    return dataloader


# function to show an image
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# check torch documentation
def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find("Linear") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def one_hot_encode(label_tensor, batch_size, n_l, use_cuda=False):
    one_hot = - torch.ones(batch_size * n_l).view(batch_size, n_l)
    for i, j in enumerate(label_tensor):
        one_hot[i, j] = 1
    if use_cuda:
        return Variable(one_hot).cuda()
    else:
        return Variable(one_hot)


# used to compute total variance loss (Denoise)
def compute_loss(img_tensor, img_size=128):
    x = (img_tensor[:, :, 1:, :] - img_tensor[:, :, :img_size - 1, :]) ** 2
    y = (img_tensor[:, :, :, 1:] - img_tensor[:, :, :, :img_size - 1]) ** 2

    out = (x.mean(dim=2) + y.mean(dim=3)).mean()
    return out
