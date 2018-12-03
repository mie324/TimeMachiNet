import torch
from torch import nn

img_size = 128
cuda_available = torch.cuda.is_available()

n_channel = 3
n_disc = 16
n_gen = 64
n_encode = 64
n_l = 10  # age vector
n_z = 50  # latent vector size
num_age = int(n_z / n_l)
num_gender = int(n_z / 2)


class Dzs(nn.Module):
    def __init__(self):
        super(Dzs, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(n_z, n_disc * 4),
            nn.ReLU(),
            nn.Linear(n_disc * 4, n_disc * 2),
            nn.ReLU(),
            nn.Linear(n_disc * 2, n_disc),
            nn.ReLU(),
            nn.Linear(n_disc, 1),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.model(z)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(n_channel, n_encode, 5, 2, 2),
            nn.ReLU(),
            nn.Conv2d(n_encode, 2 * n_encode, 5, 2, 2),
            nn.ReLU(),
            nn.Conv2d(2 * n_encode, 4 * n_encode, 5, 2, 2),
            nn.ReLU(),
            nn.Conv2d(4 * n_encode, 8 * n_encode, 5, 2, 2),
            nn.ReLU(),
        )
        self.fc = nn.Linear(8 * n_encode * 8 * 8, 50)

    def forward(self, x):
        conv = self.conv(x).view(-1, 8 * n_encode * 8 * 8)
        out = self.fc(conv)
        return out


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc = nn.Sequential(nn.Linear(n_z + n_l * num_age + num_gender,
                                          8 * 8 * n_gen * 16),
                                nn.ReLU())
        self.upconv = nn.Sequential(
            nn.ConvTranspose2d(16 * n_gen, 8 * n_gen, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(8 * n_gen, 4 * n_gen, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(4 * n_gen, 2 * n_gen, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(2 * n_gen, n_gen, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(n_gen, n_channel, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, z, age, gender):
        l = age.repeat(1, num_age).float()
        k = gender.view(-1, 1).repeat(1, num_gender).float()
        x = torch.cat([z, l, k], dim=1)
        fc = self.fc(x).view(-1, 16 * n_gen, 8, 8)
        out = self.upconv(fc)
        return out


class Dimg(nn.Module):
    def __init__(self):
        super(Dimg, self).__init__()
        self.conv_img = nn.Sequential(
            nn.Conv2d(n_channel, n_disc, 4, 2, 1),
        )
        self.conv_l = nn.Sequential(
            nn.ConvTranspose2d(n_l * num_age + num_gender, n_l * num_age + num_gender, 64, 1, 0),
            nn.ReLU()
        )
        self.total_conv = nn.Sequential(
            nn.Conv2d(n_disc + n_l * num_age + num_gender, n_disc * 2, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(n_disc * 2, n_disc * 4, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(n_disc * 4, n_disc * 8, 4, 2, 1),
            nn.ReLU()
        )
        self.fc_common = nn.Sequential(
            nn.Linear(8 * 8 * img_size, 1024),
            nn.ReLU()
        )
        self.fc_1 = nn.Sequential(
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )
        self.fc_2 = nn.Sequential(
            nn.Linear(1024, n_l),
            nn.Softmax()
        )

    def forward(self, img, age, gender):
        l = age.repeat(1, num_age, 1, 1, )
        k = gender.repeat(1, num_gender, 1, 1, )
        conv_img = self.conv_img(img)
        conv_l = self.conv_l(torch.cat([l, k], dim=1))
        concat = torch.cat((conv_img, conv_l), dim=1)
        total_conv = self.total_conv(concat).view(-1, 8 * 8 * img_size)
        com = self.fc_common(total_conv)
        a = self.fc_1(com)
        b = self.fc_2(com)
        return a, b
