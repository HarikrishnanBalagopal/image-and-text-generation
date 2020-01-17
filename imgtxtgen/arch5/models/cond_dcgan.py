"""
Conditional DCGAN.
"""

import os
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from torch import nn
from torch import optim
from imgtxtgen.common.utils import mkdir_p, get_timestamp
from imgtxtgen.arch1.models.image_generator import ImageGenerator64

def _weights_init(layer):
    """
    Custom weights initialization called on netG and netD.
    """

    classname = layer.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(layer.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(layer.weight.data, 1.0, 0.02)
        nn.init.constant_(layer.bias.data, 0)

class ImageDiscriminator64(nn.Module):
    """
    The discriminator part of the DCGAN for 64 x 64 images.
    """

    def __init__(self, d_dis, d_embed, d_ch=3):
        super().__init__()
        self.d_ch = d_ch
        self.d_dis = d_dis
        self.d_embed = d_embed

        self.define_module()
        self.model.apply(_weights_init)

    def define_module(self):
        """
        Define each part of the discriminator.
        """

        d_dis = self.d_dis
        self.model = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(self.d_ch, d_dis, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(d_dis, d_dis * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d_dis * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(d_dis * 2, d_dis * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d_dis * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(d_dis * 4, d_dis * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d_dis * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(d_dis * 8, self.d_embed, 4, 1, 0, bias=False),
        )

    def forward(self, x):
        """
        Run the model on the input.
        """
        # pylint: disable=arguments-differ
        # The arguments will differ from the base class since nn.Module is an abstract class.

        return self.model(x)

class CondDCGAN(nn.Module):
    """
    Network that takes class label and noise to generate an image.
    Designed to produce 64 x 64 color images.
    """

    def __init__(self, d_noise, d_classes, d_embed, d_hidden, d_gen, d_dis):
        super().__init__()
        self.d_noise = d_noise
        self.d_classes = d_classes
        self.d_embed = d_embed
        self.d_hidden = d_hidden
        self.d_gen = d_gen
        self.d_dis = d_dis

        self.define_module()

    def define_module(self):
        """
        Define each part of CondDCGAN.
        """

        self.embed = nn.Embedding(self.d_classes, self.d_embed)
        self.fc_hidden = nn.Linear(self.d_noise + self.d_embed, self.d_hidden)
        self.gen = ImageGenerator64(self.d_hidden, self.d_gen)
        self.dis = ImageDiscriminator64(self.d_dis, self.d_hidden)
        self.fc_classes = nn.Linear(self.d_hidden, self.d_classes)
        self.fc_real = nn.Linear(self.d_hidden, 1)

    def forward(self, noise, labels):
        """
        Run the network on noise and labels.
        noise  : torch float tensor of size d_batch x d_noise
        labels : torch long tensor of size d_batch
        """
        # pylint: disable=arguments-differ
        # The arguments will differ from the base class since nn.Module is an abstract class.

        x = self.embed(labels)
        x = torch.cat((noise, x), dim=1)
        x = self.fc_hidden(x).unsqueeze(2).unsqueeze(3) # make latent into a 4d tensor to give to conv layers.
        images = self.gen(x)

        # convert d_batch x d_hidden x 1 x 1 to d_batch x d_hidden.
        # squeeze twice to avoid getting rid of batch dimension on single image batches.
        x = self.dis(images).squeeze(2).squeeze(2)
        class_logits = self.fc_classes(x)
        real_logits = self.fc_real(x).squeeze()
        return class_logits, real_logits

    def discriminate(self, images):
        """
        Run just the discriminator on given images.
        """

        # convert d_batch x d_hidden x 1 x 1 to d_batch x d_hidden.
        # squeeze twice to avoid getting rid of batch dimension on single image batches.
        x = self.dis(images).squeeze(2).squeeze(2)
        class_logits = self.fc_classes(x)
        real_logits = self.fc_real(x).squeeze()
        return class_logits, real_logits

    def sample(self, noise, labels):
        """
        Sample some images from the network given noise and labels.
        noise  : torch float tensor of size d_batch x d_noise
        labels : torch long tensor of size d_batch
        """

        x = self.embed(labels)
        x = torch.cat((noise, x), dim=1)
        x = self.fc_hidden(x).unsqueeze(2).unsqueeze(3) # make latent into a 4d tensor to give to conv layers.
        images = self.gen(x)
        return images

    def get_gen_parameters(self):
        """
        Get the generator parameters.
        """

        return list(self.embed.parameters()) + list(self.fc_hidden.parameters()) + list(self.gen.parameters())

    def get_dis_parameters(self):
        """
        Get the discriminator parameters.
        """

        return list(self.dis.parameters()) + list(self.fc_classes.parameters()) + list(self.fc_real.parameters())

def train_cond_dcgan(model, data_loader, device, d_batch, num_epochs, output_dir, print_every=100, save_results=True, config_name='cond_dcgan', learning_rate=0.0002):
    """
    Train the conditonal DCGAN on the given dataset.
    """
    # pylint: disable=too-many-locals
    # pylint: disable=too-many-arguments
    # This number of local variables is necessary for training.
    # The arguments are necesarry for training.

    beta1 = 0.5
    d_noise = model.d_noise
    s_noise = (d_batch, d_noise)
    output_dir = os.path.join(output_dir, f'{config_name}_{get_timestamp()}')
    output_images_dir = os.path.join(output_dir, 'images')
    output_weights_dir = os.path.join(output_dir, 'weights')

    real_targets = torch.full((d_batch,), 1, device=device)
    fake_targets = torch.full((d_batch,), 0, device=device)
    fixed_noise = torch.randn(s_noise, device=device)
    fixed_fake_labels = torch.randint(low=0, high=model.d_classes, size=(d_batch,), device=device)

    criterion_real = nn.BCEWithLogitsLoss()
    criterion_class = nn.CrossEntropyLoss()

    opt_g = optim.Adam(model.get_gen_parameters(), lr=learning_rate, betas=(beta1, 0.999))
    opt_d = optim.Adam(model.get_dis_parameters(), lr=learning_rate, betas=(beta1, 0.999))

    mkdir_p(output_images_dir)
    mkdir_p(output_weights_dir)

    dis_train_losses = []
    gen_train_losses = []
    gen_eval_losses = []

    model.train()

    for epoch in range(1, num_epochs+1):
        print('epoch:', epoch)
        running_loss_g = 0
        running_loss_d = 0

        for i, batch in enumerate(data_loader, start=1):
            # Get batch.
            real_imgs, _, _, real_labels = batch
            real_imgs, real_labels = real_imgs.to(device), real_labels.to(device)
            with torch.no_grad():
                noise = torch.randn(s_noise, device=device)
                fake_labels = torch.randint(low=0, high=model.d_classes, size=(d_batch,), device=device)
                fake_imgs = model.sample(noise, fake_labels)

            # Train discriminator
            real_class_logits, real_real_logits = model.discriminate(real_imgs)
            fake_class_logits, fake_real_logits = model.discriminate(fake_imgs)

            loss_d_real = criterion_real(real_real_logits, real_targets) + criterion_class(real_class_logits, real_labels)
            loss_d_fake = criterion_real(fake_real_logits, fake_targets) + criterion_class(fake_class_logits, fake_labels)
            loss_d = loss_d_real + loss_d_fake

            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()

            # Train generator
            noise = torch.randn(s_noise, device=device)
            fake_labels = torch.randint(low=0, high=model.d_classes, size=(d_batch,), device=device)
            fake_class_logits, fake_real_logits = model(noise, fake_labels)
            loss_g = criterion_real(fake_real_logits, real_targets) + criterion_class(fake_class_logits, fake_labels)

            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()

            running_loss_g += loss_g.item()
            running_loss_d += loss_d.item()

            if i % print_every == 0:
                running_loss_g /= print_every
                running_loss_d /= print_every
                print(f'i: {i}| loss_d_real: {loss_d_real}, loss_d_fake: {loss_d_fake} | loss_g: {loss_g} | running losses -> gen:{running_loss_g} | dis: {running_loss_d}')
                gen_train_losses.append(running_loss_g)
                dis_train_losses.append(running_loss_d)
                running_loss_g = 0
                running_loss_d = 0
                model.eval()
                with torch.no_grad():
                    fake_imgs = model.sample(fixed_noise, fixed_fake_labels)
                    fake_class_logits, fake_real_logits = model.discriminate(fake_imgs)
                    loss_g = criterion_real(fake_real_logits, real_targets) + criterion_class(fake_class_logits, fake_labels)
                model.train()
                print('evaluation loss_g:', loss_g.item())
                gen_eval_losses.append(loss_g.item())
                image_grid = torchvision.utils.make_grid(fake_imgs, padding=2, normalize=True).cpu()
                image_grid = np.transpose(image_grid, (1, 2, 0)) # convert channels first to channels last format.
                plt.imshow(image_grid)
                if save_results:
                    plt.savefig(os.path.join(output_images_dir, f'gen_epoch_{epoch}_iter_{i}.png'))
                    torch.save(model.state_dict(), os.path.join(output_weights_dir, f'gen_epoch_{epoch}_iter_{i}.pth'))
                    plt.close()
                    plt.plot(dis_train_losses, label='dis train')
                    plt.plot(gen_train_losses, label='gen train')
                    plt.plot(gen_eval_losses, label='gen eval')
                    plt.legend()
                    plt.savefig(os.path.join(output_images_dir, f'losses_epoch_{epoch}_iter_{i}.png'))
                    plt.close()
