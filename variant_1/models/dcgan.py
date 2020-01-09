"""
Deep Convolutional Generative Adversarial Network (DCGAN) is a generative model
that can be sampled to get images. The network uses 2d convolutional layers.
"""
# pylint: disable=wrong-import-order
# pylint: disable=ungrouped-imports
# The imports are ordered by length.

import os
import torch
import datetime
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from torch import nn
from torch import optim
from utils import mkdir_p
from torchvision import transforms
from datasets.cub2011 import CUB2011Dataset
from datasets.celeba import get_celeba_dataset
from .image_generator import ImageGenerator64, ImageGenerator256
from .image_discriminator import ImageDiscriminator64, ImageDiscriminator256

class DCGAN256(nn.Module):
    """
    The DCGAN network for 256 x 256 images.
    """

    def __init__(self, d_noise, d_gen, d_dis):
        super().__init__()
        self.d_noise = d_noise
        self.d_gen = d_gen
        self.d_dis = d_dis

        self.define_module()

    def define_module(self):
        """
        Define each part of the GAN.
        """

        self.img_gen = ImageGenerator256(d_noise=self.d_noise, d_gen=self.d_gen)
        self.img_dis = ImageDiscriminator256(d_dis=self.d_dis)

    def forward(self, noise):
        # pylint: disable=arguments-differ
        # pylint: disable=invalid-name
        # The arguments will differ from the base class since nn.Module is an abstract class.
        # Short variable names like x, h and c are fine in this context.

        images = self.img_gen(noise)
        pred_logits = self.img_dis(images)
        return pred_logits

class DCGAN64(nn.Module):
    """
    The DCGAN network for 64 x 64 images.
    """

    def __init__(self, d_noise, d_gen, d_dis):
        super().__init__()
        self.d_noise = d_noise
        self.d_gen = d_gen
        self.d_dis = d_dis

        self.define_module()

    def define_module(self):
        """
        Define each part of the GAN.
        """

        self.img_gen = ImageGenerator64(d_noise=self.d_noise, d_gen=self.d_gen)
        self.img_dis = ImageDiscriminator64(d_dis=self.d_dis)

    def forward(self, noise):
        # pylint: disable=arguments-differ
        # pylint: disable=invalid-name
        # The arguments will differ from the base class since nn.Module is an abstract class.
        # Short variable names like x, h and c are fine in this context.

        images = self.img_gen(noise)
        pred_logits = self.img_dis(images)
        return pred_logits

def train_dcgan(dcgan, dataloader, device, d_batch, num_epochs, print_every=100, save_results=True):
    """
    Train the DCGAN on the given dataset.
    """
    # pylint: disable=too-many-locals
    # pylint: disable=too-many-arguments
    # This number of local variables is necessary for training.
    # The arguments are necesarry for training.

    beta1 = 0.5
    d_noise = 100
    learning_rate = 0.0002
    s_noise = (d_batch, d_noise, 1, 1)
    real_labels = torch.full((d_batch,), 1, device=device)
    fake_labels = torch.full((d_batch,), 0, device=device)

    criterion = nn.BCELoss()
    opt_g = optim.Adam(dcgan.img_gen.parameters(), lr=learning_rate, betas=(beta1, 0.999))
    opt_d = optim.Adam(dcgan.img_dis.parameters(), lr=learning_rate, betas=(beta1, 0.999))

    fixed_noise = torch.randn(s_noise, device=device)

    now = datetime.datetime.now()
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = os.path.join('.', 'output', f'generated_{timestamp}')
    output_images_dir = os.path.join(output_dir, 'images')
    output_weights_dir = os.path.join(output_dir, 'weights')
    mkdir_p(output_images_dir)
    mkdir_p(output_weights_dir)

    for epoch in range(1, num_epochs+1):
        print('epoch:', epoch)

        for i, batch in enumerate(dataloader, start=1):
            opt_d.zero_grad()

            real_imgs = batch[0]
            real_imgs = real_imgs.to(device)
            pred_logits = dcgan.img_dis(real_imgs).squeeze()
            loss_d_real = criterion(pred_logits, real_labels)
            loss_d_real.backward()

            noise = torch.randn(s_noise, device=device)
            fake_imgs = dcgan.img_gen(noise)
            pred_logits = dcgan.img_dis(fake_imgs).squeeze()
            loss_d_fake = criterion(pred_logits, fake_labels)
            loss_d_fake.backward(retain_graph=True)

            opt_d.step()

            opt_g.zero_grad()

            loss_g = criterion(pred_logits, real_labels)
            loss_g.backward()

            opt_g.step()

            if i % print_every == 0:
                print(f'i: {i}| loss_d_real: {loss_d_real}, loss_d_fake: {loss_d_fake} | loss_g: {loss_g}')
                with torch.no_grad():
                    fake_imgs = dcgan.img_gen(fixed_noise)
                    pred_logits = dcgan.img_dis(fake_imgs).squeeze()
                    loss_g = criterion(pred_logits, real_labels)
                print('evaluation loss_g:', loss_g)
                image_grid = torchvision.utils.make_grid(fake_imgs, padding=2, normalize=True).cpu()
                image_grid = np.transpose(image_grid, (1, 2, 0)) # convert channels first to channels last format.
                plt.imshow(image_grid)
                if save_results:
                    plt.savefig(os.path.join(output_images_dir, f'gen_epoch_{epoch}_iter_{i}.png'))
                    torch.save(dcgan.state_dict(), os.path.join(output_weights_dir, f'gen_epoch_{epoch}_iter_{i}.pth'))

def test_dcgan_256():
    """
    Test DCGAN for 256 x 256 images.
    """
    # pylint: disable=too-many-locals
    # This number of local variables is necessary for testing.

    d_batch = 16
    d_noise = 100
    d_gen = 64
    d_dis = 64
    d_image_size = 256

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = DCGAN256(d_noise=d_noise, d_gen=d_gen, d_dis=d_dis).to(device)
    print(model)

    s_noise = (d_batch, d_noise, 1, 1)
    noise = torch.randn(s_noise, device=device)
    pred_logits = model(noise)
    print('noise:', noise.size(), 'pred_logits:', pred_logits.size())

    # print('training on celeba:')
    # celeba_dataset = get_celeba_dataset(d_image_size=d_image_size)
    # dataloader = torch.utils.data.DataLoader(celeba_dataset, batch_size=d_batch, shuffle=True, drop_last=True, num_workers=4)
    # train_dcgan(model, dataloader, device=device, d_batch=d_batch, num_epochs=2, save_results=False)

    print('training on cub2011:')
    d_max_seq_len = 18
    cub2011_dataset_dir = '../../../exp2/CUB_200_2011'
    cub2011_captions_dir = '../../../exp2/cub_with_captions'
    img_transforms = transforms.Compose([
        transforms.Resize((d_image_size, d_image_size)),
        transforms.ToTensor(),
        transforms.Normalize((.5, .5, .5), (.5, .5, .5))
    ])
    cub2011_dataset = CUB2011Dataset(dataset_dir=cub2011_dataset_dir, captions_dir=cub2011_captions_dir, img_transforms=img_transforms, d_max_seq_len=d_max_seq_len)
    dataloader = torch.utils.data.DataLoader(cub2011_dataset, batch_size=d_batch, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    train_dcgan(model, dataloader, device=device, d_batch=d_batch, num_epochs=200)

def test_dcgan_64():
    """
    Test DCGAN for 64 x 64 images.
    """
    # pylint: disable=too-many-locals
    # This number of local variables is necessary for testing.

    d_batch = 20
    d_noise = 100
    d_gen = 64
    d_dis = 64
    d_image_size = 64

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = DCGAN64(d_noise=d_noise, d_gen=d_gen, d_dis=d_dis).to(device)
    print(model)

    s_noise = (d_batch, d_noise, 1, 1)
    noise = torch.randn(s_noise, device=device)
    pred_logits = model(noise)
    print('noise:', noise.size(), 'pred_logits:', pred_logits.size())

    print('training on celeba:')
    celeba_dataset = get_celeba_dataset(d_image_size=d_image_size)
    dataloader = torch.utils.data.DataLoader(celeba_dataset, batch_size=d_batch, shuffle=True, drop_last=True, num_workers=4)
    train_dcgan(model, dataloader, device=device, d_batch=d_batch, num_epochs=2, save_results=False)

    '''
    print('training on cub2011:')
    d_max_seq_len = 18
    cub2011_dataset_dir = '../../../exp2/CUB_200_2011'
    cub2011_captions_dir = '../../../exp2/cub_with_captions'
    img_transforms = transforms.Compose([
        transforms.Resize((d_image_size, d_image_size)),
        transforms.ToTensor(),
        transforms.Normalize((.5, .5, .5), (.5, .5, .5))
    ])
    cub2011_dataset = CUB2011Dataset(dataset_dir=cub2011_dataset_dir, captions_dir=cub2011_captions_dir, img_transforms=img_transforms, d_max_seq_len=d_max_seq_len)
    dataloader = torch.utils.data.DataLoader(cub2011_dataset, batch_size=d_batch, shuffle=True, drop_last=True, num_workers=4)
    train_dcgan(model, dataloader, device=device, d_batch=d_batch, num_epochs=200)
    '''

def run_tests():
    """
    Run tests on GAN.
    """

    # test_dcgan_64()
    test_dcgan_256()


if __name__ == '__main__':
    run_tests()
