from neural_network.models import *
from neural_network.utils import *
import pickle
from neural_network.labelDataset import *
import os
import argparse
import csv


def train(args):
    output = args.output
    if not os.path.exists(output):
        os.mkdir(output)

    loss_file = open('output/losses.csv', mode='w')
    columns = ['epoch', 'EG_L1_loss', 'G_tv_loss', 'G_img_loss', 'Ez_loss', 'D_loss', 'Dz_loss', 'D_img', 'D_z',
               'D_z_prior', 'D_reconst']
    writer = csv.DictWriter(loss_file, fieldnames=columns)
    writer.writeheader()

    cuda_available = torch.cuda.is_available()

    create_destination()
    move_dataset()
    image_loader = get_data_loader()

    if cuda_available:
        netD_z = Dzs().cuda()
        netE = Encoder().cuda()
        netG = Generator().cuda()
        netD_img = Dimg().cuda()

    else:
        netE = Encoder()
        netG = Generator()
        netD_img = Dimg()
        netD_z = Dzs()

    netE.apply(init_weights)
    netD_img.apply(init_weights)
    netD_z.apply(init_weights)
    netG.apply(init_weights)

    beta = (args.beta1, args.beta2)
    optimizerE = optim.Adam(netE.parameters(), lr=args.lr, betas=beta)
    optimizerD_z = optim.Adam(netD_z.parameters(), lr=args.lr, betas=beta)
    optimizerD_img = optim.Adam(netD_img.parameters(), lr=args.lr, betas=beta)
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=beta)

    if cuda_available:
        BCE = nn.BCELoss().cuda()
        L1 = nn.L1Loss().cuda()
    else:
        BCE = nn.BCELoss()
        L1 = nn.L1Loss()

    val_l = -torch.ones(80 * 10).view(80, 10)
    for i, l in enumerate(val_l):
        l[i // 8] = 1
    val_l = Variable(val_l)

    if cuda_available:
        val_l = val_l.cuda()

    for epoch in range(args.epochs):
        for i, (img, label) in enumerate(image_loader):
            for j in range(len(label)):
                label[j] = converttoage(label[j])
            img = Variable(img)
            age = label / 2
            gender = label % 2 * 2 - 1

            age_var = Variable(age).view(-1, 1)
            gender_var = Variable(gender.float())

            if epoch == 0 and i == 0:
                val_img = img[:8].repeat(10, 1, 1, 1)
                val_gender = gender[:8].view(-1, 1).repeat(10, 1)
                val_img_var = Variable(val_img)
                val_gender_var = Variable(val_gender)

                pickle.dump(val_img, open("fixed_noise.p", "wb"))

                if cuda_available:
                    val_img_var = val_img_var.cuda()
                    val_gender_var = val_gender_var.cuda()

            if cuda_available:
                img = img.cuda()
                age_var = age_var.cuda()
                gender_var = gender_var.cuda()

            # make one hot encoding version of label
            batch_size = img.size(0)
            age_one_hot = one_hot_encode(age, batch_size, n_l, cuda_available)

            # prior distribution z_star, real_label, fake_label
            z_star = Variable(torch.FloatTensor(batch_size * n_z).uniform_(-1, 1)).view(batch_size, n_z)
            real_label = Variable(torch.ones(batch_size).fill_(1)).view(-1, 1)
            fake_label = Variable(torch.ones(batch_size).fill_(0)).view(-1, 1)

            if cuda_available:
                z_star, real_label, fake_label = z_star.cuda(), real_label.cuda(), fake_label.cuda()

            # train Encoder and Generator with reconstruction loss
            netE.zero_grad()
            netG.zero_grad()

            # EG_loss 1. L1 reconstruction loss
            z = netE(img)
            reconst = netG(z, age_one_hot, gender_var)
            EG_L1_loss = L1(reconst, img)

            # EG_loss 2. GAN loss - image
            z = netE(img)
            reconstruction = netG(z, age_one_hot, gender_var)
            D_reconstruction, _ = netD_img(reconstruction, age_one_hot.view(batch_size, n_l, 1, 1),
                                           gender_var.view(batch_size, 1, 1, 1))
            G_img_loss = BCE(D_reconstruction, real_label)

            # EG_loss 3. GAN loss - z
            Dz_prior = netD_z(z_star)
            Dz = netD_z(z)
            Ez_loss = BCE(Dz, real_label)

            # EG_loss 4. loss - G
            reconstruction = netG(z.detach(), age_one_hot, gender_var)
            G_tv_loss = compute_loss(reconstruction)

            EG_loss = EG_L1_loss + 0.0001 * G_img_loss + 0.01 * Ez_loss + G_tv_loss
            EG_loss.backward()

            optimizerE.step()
            optimizerG.step()

            # train netD_z with prior distribution U(-1,1)
            netD_z.zero_grad()
            Dz_prior = netD_z(z_star)
            Dz = netD_z(z.detach())

            Dz_loss = BCE(Dz_prior, real_label) + BCE(Dz, fake_label)
            Dz_loss.backward()
            optimizerD_z.step()

            # train D_img with real images
            netD_img.zero_grad()
            D_img, D_clf = netD_img(img, age_one_hot.view(batch_size, n_l, 1, 1), gender_var.view(batch_size, 1, 1, 1))
            D_reconstruction, _ = netD_img(reconst.detach(), age_one_hot.view(batch_size, n_l, 1, 1),
                                           gender_var.view(batch_size, 1, 1, 1))

            D_loss = BCE(D_img, real_label) + BCE(D_reconstruction, fake_label)
            D_loss.backward()
            optimizerD_img.step()

            print("epoch:{}, step:{}".format(epoch + 1, i + 1))

        if (epoch % 10 == 0 or epoch == 149 or G_img_loss <= 5):
            torch.save(netE, "{}/encoder_epoch_{}.pt".format(output, epoch))
            torch.save(netG, "{}/generator_epoch_{}.pt".format(output, epoch))
            val_z = netE(val_img_var)
            val_gen = netG(val_z, val_l, val_gender_var)
            vutils.save_image(val_gen.data,
                              "{}/validation_epoch_{}.png".format(output, epoch),
                              normalize=True)
        writer.writerow({'epoch': epoch, 'EG_L1_loss': EG_L1_loss.data[0], 'G_tv_loss': G_tv_loss.data[0],
                         'G_img_loss': G_img_loss.data[0], 'Ez_loss': Ez_loss.data[0]
                            , 'D_loss': D_loss.data[0], 'Dz_loss': Dz_loss.data[0], 'D_img': D_img.mean().data[0],
                         'D_z': Dz.mean().data[0],
                         'D_z_prior': Dz_prior.mean().data[0], 'D_reconst': D_reconstruction.mean().data[0]})

        print("epoch:{}, step:{}".format(epoch + 1, i + 1))
        print("EG_L1_loss:{} | G_img_loss:{}".format(EG_L1_loss.data[0], G_img_loss.data[0]))
        print("G_tv_loss:{} | Ez_loss:{}".format(G_tv_loss.data[0], Ez_loss.data[0]))
        print("D_img:{} | D_reconst:{} | D_loss:{}".format(D_img.mean().data[0], D_reconstruction.mean().data[0],
                                                           D_loss.data[0]))
        print("D_z:{} | D_z_prior:{} | Dz_loss:{}".format(Dz.mean().data[0], Dz_prior.mean().data[0], Dz_loss.data[0]))
        print("-" * 70)
    loss_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--output', type=str, default='output')

    args = parser.parse_args()

    train(args)
