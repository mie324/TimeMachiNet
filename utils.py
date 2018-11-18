import torch
from PIL import ImageFile
import torchvision.transforms as transforms
import torchvision.datasets as data
import torchvision.utils as vutils
from torch import nn
from torch import optim
from torch.autograd import Variable


ImageFile.LOAD_TRUNCATED_IMAGES = True

labeled_dataset = "data/UTKFace/labeled"
img_size = 128
batch_size = 20

def get_data_loader():
    dataset = data.ImageFolder(root=labeled_dataset,
                               transform=transforms.Compose([
                                   transforms.Resize(img_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size= batch_size,
                                             shuffle=True)
    return dataloader

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find("Linear") !=-1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def one_hot_encode(label_tensor,batch_size,n_l,use_cuda=False):
    one_hot = - torch.ones(batch_size*n_l).view(batch_size,n_l)
    for i,j in enumerate(label_tensor):
        one_hot[i,j] = 1
    if use_cuda:
        return Variable(one_hot).cuda()
    else:
        return Variable(one_hot)

def compute_loss(img_tensor,img_size=128):
    x = (img_tensor[:,:,1:,:]-img_tensor[:,:,:img_size-1,:])**2
    y = (img_tensor[:,:,:,1:]-img_tensor[:,:,:,:img_size-1])**2

    out = (x.mean(dim=2)+y.mean(dim=3)).mean()
    return out
