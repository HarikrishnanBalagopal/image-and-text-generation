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

    def load_weights_from_dcgan_64(self, weights_64):
        """
        Load relevant weights from pretrained DCGAN64 model.
        loading pretrained weights of dcgan 64:
        """
        layers_from_to_mapping = {
            'img_gen.model.0.weight'                    : 'img_gen.model.0.weight',
            'img_gen.model.1.weight'                    : 'img_gen.model.1.weight',
            'img_gen.model.1.bias'                      : 'img_gen.model.1.bias',
            'img_gen.model.1.running_mean'              : 'img_gen.model.1.running_mean',
            'img_gen.model.1.running_var'               : 'img_gen.model.1.running_var',
            'img_gen.model.1.num_batches_tracked'       : 'img_gen.model.1.num_batches_tracked',
            'img_gen.model.3.weight'                    : 'img_gen.model.3.weight',
            'img_gen.model.4.weight'                    : 'img_gen.model.4.weight',
            'img_gen.model.4.bias'                      : 'img_gen.model.4.bias',
            'img_gen.model.4.running_mean'              : 'img_gen.model.4.running_mean',
            'img_gen.model.4.running_var'               : 'img_gen.model.4.running_var',
            'img_gen.model.4.num_batches_tracked'       : 'img_gen.model.4.num_batches_tracked',
            'img_gen.model.6.weight'                    : 'img_gen.model.6.weight',
            'img_gen.model.7.weight'                    : 'img_gen.model.7.weight',
            'img_gen.model.7.bias'                      : 'img_gen.model.7.bias',
            'img_gen.model.7.running_mean'              : 'img_gen.model.7.running_mean',
            'img_gen.model.7.running_var'               : 'img_gen.model.7.running_var',
            'img_gen.model.7.num_batches_tracked'       : 'img_gen.model.7.num_batches_tracked',
            'img_gen.model.9.weight'                    : 'img_gen.model.9.weight',
            'img_gen.model.10.weight'                   : 'img_gen.model.10.weight',
            'img_gen.model.10.bias'                     : 'img_gen.model.10.bias',
            'img_gen.model.10.running_mean'             : 'img_gen.model.10.running_mean',
            'img_gen.model.10.running_var'              : 'img_gen.model.10.running_var',
            'img_gen.model.10.num_batches_tracked'      : 'img_gen.model.10.num_batches_tracked',

            'img_dis.model.2.weight'                    : 'img_dis.model.8.weight',
            'img_dis.model.3.weight'                    : 'img_dis.model.9.weight',
            'img_dis.model.3.bias'                      : 'img_dis.model.9.bias',
            'img_dis.model.3.running_mean'              : 'img_dis.model.9.running_mean',
            'img_dis.model.3.running_var'               : 'img_dis.model.9.running_var',
            'img_dis.model.3.num_batches_tracked'       : 'img_dis.model.9.num_batches_tracked',
            'img_dis.model.5.weight'                    : 'img_dis.model.11.weight',
            'img_dis.model.6.weight'                    : 'img_dis.model.12.weight',
            'img_dis.model.6.bias'                      : 'img_dis.model.12.bias',
            'img_dis.model.6.running_mean'              : 'img_dis.model.12.running_mean',
            'img_dis.model.6.running_var'               : 'img_dis.model.12.running_var',
            'img_dis.model.6.num_batches_tracked'       : 'img_dis.model.12.num_batches_tracked',
            'img_dis.model.8.weight'                    : 'img_dis.model.14.weight',
            'img_dis.model.9.weight'                    : 'img_dis.model.15.weight',
            'img_dis.model.9.bias'                      : 'img_dis.model.15.bias',
            'img_dis.model.9.running_mean'              : 'img_dis.model.15.running_mean',
            'img_dis.model.9.running_var'               : 'img_dis.model.15.running_var',
            'img_dis.model.9.num_batches_tracked'       : 'img_dis.model.15.num_batches_tracked',
            'img_dis.model.11.weight'                   : 'img_dis.model.17.weight'
        }

        model_dict = self.state_dict()
        filtered_dict = {layers_from_to_mapping[k]: v for k, v in weights_64.items() if k in layers_from_to_mapping}
        model_dict.update(filtered_dict)
        self.load_state_dict(model_dict)

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

def train_dcgan(dcgan, dataloader, device, d_batch, num_epochs, print_every=100, save_results=True, config_name='generated'):
    """
    Train the DCGAN on the given dataset.
    """
    # pylint: disable=too-many-locals
    # pylint: disable=too-many-arguments
    # This number of local variables is necessary for training.
    # The arguments are necesarry for training.

    beta1 = 0.5
    learning_rate = 0.0002
    d_noise = dcgan.d_noise
    s_noise = (d_batch, d_noise, 1, 1)
    real_labels = torch.full((d_batch,), 1, device=device)
    fake_labels = torch.full((d_batch,), 0, device=device)

    criterion = nn.BCELoss()
    opt_g = optim.Adam(dcgan.img_gen.parameters(), lr=learning_rate, betas=(beta1, 0.999))
    opt_d = optim.Adam(dcgan.img_dis.parameters(), lr=learning_rate, betas=(beta1, 0.999))

    fixed_noise = torch.randn(s_noise, device=device)

    now = datetime.datetime.now()
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = os.path.join('.', 'output', f'{config_name}_{timestamp}')
    output_images_dir = os.path.join(output_dir, 'images')
    output_weights_dir = os.path.join(output_dir, 'weights')
    mkdir_p(output_images_dir)
    mkdir_p(output_weights_dir)

    dis_train_losses = []
    gen_train_losses = []
    gen_eval_losses = []

    for epoch in range(1, num_epochs+1):
        print('epoch:', epoch)

        running_loss_g = 0
        running_loss_d = 0
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

            running_loss_g += loss_g.item()
            running_loss_d += loss_d_real.item() + loss_d_fake.item()

            if i % print_every == 0:
                running_loss_g /= print_every
                running_loss_d /= print_every
                print(f'i: {i}| loss_d_real: {loss_d_real}, loss_d_fake: {loss_d_fake} | loss_g: {loss_g} | running losses -> gen:{running_loss_g} | dis: {running_loss_d}')
                gen_train_losses.append(running_loss_g)
                dis_train_losses.append(running_loss_d)
                running_loss_g = 0
                running_loss_d = 0

                with torch.no_grad():
                    fake_imgs = dcgan.img_gen(fixed_noise)
                    pred_logits = dcgan.img_dis(fake_imgs).squeeze()
                    loss_g = criterion(pred_logits, real_labels)
                print('evaluation loss_g:', loss_g)
                gen_eval_losses.append(loss_g.item())
                image_grid = torchvision.utils.make_grid(fake_imgs, padding=2, normalize=True).cpu()
                image_grid = np.transpose(image_grid, (1, 2, 0)) # convert channels first to channels last format.
                plt.imshow(image_grid)
                if save_results:
                    plt.savefig(os.path.join(output_images_dir, f'gen_epoch_{epoch}_iter_{i}.png'))
                    torch.save(dcgan.state_dict(), os.path.join(output_weights_dir, f'gen_epoch_{epoch}_iter_{i}.pth'))
                    plt.close()
                    plt.plot(dis_train_losses, label='dis train')
                    plt.plot(gen_train_losses, label='gen train')
                    plt.plot(gen_eval_losses, label='gen eval')
                    plt.legend()
                    plt.savefig(os.path.join(output_images_dir, f'losses_epoch_{epoch}_iter_{i}.png'))
                    plt.close()

def test_dcgan_256():
    """
    Test DCGAN for 256 x 256 images.
    """
    # pylint: disable=too-many-locals
    # This number of local variables is necessary for testing.

    gpu_id = 1
    d_batch = 16
    d_noise = 100
    d_gen = 16
    d_dis = 16
    d_image_size = 256
    pretrained_weights_64_path = os.path.join('.', 'output', 'generated_2020_01_09_03_06_33', 'weights', 'gen_epoch_200_iter_200.pth')
    print('loading pretrained weights of dcgan 64:')
    pretrained_weights_64 = torch.load(pretrained_weights_64_path)
    # for k, v in pretrained_weights_64.items():
    #     print(f'{k:40}|{str(v.size()):40}')

    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

    model = DCGAN256(d_noise=d_noise, d_gen=d_gen, d_dis=d_dis).to(device)
    print('dcgan 256 weights:')
    model_dict = model.state_dict()
    # for k, v in model_dict.items():
    #     print(f'{k:40}|{str(v.size()):40}')

    model.load_weights_from_dcgan_64(pretrained_weights_64)
    print('loaded weights from dcgan 64')

    s_noise = (d_batch, d_noise, 1, 1)
    noise = torch.randn(s_noise, device=device)
    pred_logits = model(noise)
    print('noise:', noise.size(), 'pred_logits:', pred_logits.size())

    # print('training on celeba:')
    # celeba_dataset = get_celeba_dataset(d_image_size=d_image_size)
    # dataloader = torch.utils.data.DataLoader(celeba_dataset, batch_size=d_batch, shuffle=True, drop_last=True, num_workers=4)
    # train_dcgan(model, dataloader, device=device, d_batch=d_batch, num_epochs=2, save_results=False, config_name='dcgan_256_celeba')

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
    train_dcgan(model, dataloader, device=device, d_batch=d_batch, num_epochs=200, config_name='dcgan_256_cub2011_using_pretrained_64')

def test_dcgan_64():
    """
    Test DCGAN for 64 x 64 images.
    """
    # pylint: disable=too-many-locals
    # This number of local variables is necessary for testing.

    gpu_id = 1
    d_batch = 64 # works with 20
    d_noise = 100 # failed with 512, works with 100
    d_gen = 256 # works with 64
    d_dis = 256 # works with 64
    d_image_size = 64
    print_every = 25

    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

    model = DCGAN64(d_noise=d_noise, d_gen=d_gen, d_dis=d_dis).to(device)
    print(model)

    # s_noise = (d_batch, d_noise, 1, 1)
    # noise = torch.randn(s_noise, device=device)
    # pred_logits = model(noise)
    # print('noise:', noise.size(), 'pred_logits:', pred_logits.size())

    # print('training on celeba:')
    # celeba_dataset = get_celeba_dataset(d_image_size=d_image_size)
    # dataloader = torch.utils.data.DataLoader(celeba_dataset, batch_size=d_batch, shuffle=True, drop_last=True, num_workers=4)
    # train_dcgan(model, dataloader, device=device, d_batch=d_batch, num_epochs=2, save_results=False, config_name='dcgan_64_celeba')

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
    train_dcgan(model, dataloader, device=device, d_batch=d_batch, num_epochs=200, print_every=print_every, config_name='dcgan_64_cub2011')

def run_tests():
    """
    Run tests on GAN.
    """

    # test_dcgan_64()
    test_dcgan_256()


if __name__ == '__main__':
    run_tests()
