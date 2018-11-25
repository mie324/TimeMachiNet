import torch
import argparse
import os
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as data
import torchvision.utils as vutils

pretrained_encoder = "./models/encoder_epoch_49"
pretrained_generator = "./models/generator_epoch_49"
input = "./test/input"
output_model = "./test/output/output.png"
output_orig = "./test/output/input_img.png"

def evaluate():
    netG = torch.load(pretrained_encoder, map_location="cpu")
    netE = torch.load(pretrained_generator, map_location="cpu")

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
        vutils.save_image(val_gen.data,
                          output_orig,
                          normalize=True)
        val_z = netE(Variable(img.repeat(10, 1, 1, 1)))
        gender_var = Variable(torch.Tensor(args.gender))
        val_gender_var = Variable(gender_var[:1].view(-1, 1).repeat(10, 1))
        val_gen = netG(val_z, val_l, val_gender_var)
        vutils.save_image(val_gen.data,
                          output_model,
                          normalize=True)

        
def test(args):
    output = args.output
    if not os.path.exists(output):
        os.mkdir(output)

    netG = torch.load(args.pretrained_generator, map_location='cpu')
    netE = torch.load(args.pretrained_encoder, map_location='cpu')

    val_l = -torch.ones(10 * 10).view(10, 10)
    for i, l in enumerate(val_l):
        l[i // 1] = 1
    val_l = Variable(val_l)
    img = data.ImageFolder(root=args.input,
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
        val_z = netE(Variable(img.repeat(10, 1, 1, 1)))
        gender_var = Variable(torch.Tensor(args.gender))
        val_gender_var = Variable(gender_var[:1].view(-1, 1).repeat(10, 1))
        val_gen = netG(val_z, val_l, val_gender_var)
        vutils.save_image(val_gen.data,
                          "{}/test.png".format(args.output),
                          normalize=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained-generator', type=str, default="models/generator_epoch_49.pt")
    parser.add_argument('--pretrained-encoder', type=str, default="models/encoder_epoch_49.pt")
    parser.add_argument('--input', type=str, default="test")
    parser.add_argument('--output', type=str, default='output')
    parser.add_argument('--gender', type=int, default=1)
    parser.add_argument('--age', type=int, default=20)

    args = parser.parse_args()

    test(args)
