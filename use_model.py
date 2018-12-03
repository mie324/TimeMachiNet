import torch
import argparse
import os
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as data
import torchvision.utils as vutils

pretrained_encoder = "./models/encoder_epoch_40.pt"
pretrained_generator = "./models/generator_epoch_40.pt"
input = "./static/test/input"
output_model = "./static/test/output/output.png"
output_orig = "./static/test/output/input_img.png"


def evaluate():
    current_dir = os.getcwd()
    output = os.path.join(current_dir, r'static/test/output')
    if not os.path.exists(output):
        os.mkdir(output)
    netG = torch.load(pretrained_generator, map_location="cpu")
    netE = torch.load(pretrained_encoder, map_location="cpu")

    val_l = -torch.ones(10 * 10).view(10, 10)
    for i, l in enumerate(val_l):
        l[i // 1] = 1
    val_l = Variable(val_l)
    img = data.ImageFolder(root=input,
                           transform=transforms.Compose([
                               transforms.Resize(128),
                               transforms.CenterCrop(128),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
    dataloader = torch.utils.data.DataLoader(img,
                                             batch_size=1,
                                             shuffle=True)
    for i, (img, label) in enumerate(dataloader):
        input_img = Variable(img)
        vutils.save_image(input_img.data,
                          output_orig,
                          normalize=True)
        val_z = netE(Variable(img.repeat(10, 1, 1, 1)))
        gender_var = Variable(torch.Tensor(1))
        val_gender_var = Variable(gender_var[:1].view(-1, 1).repeat(10, 1))
        val_gen = netG(val_z, val_l, val_gender_var)
        vutils.save_image(val_gen.data,
                          output_model,
                          normalize=True,
                          nrow=10)