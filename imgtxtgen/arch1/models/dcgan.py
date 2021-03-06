"""
Deep Convolutional Generative Adversarial Network (DCGAN) is a generative model
that can be sampled to get images. The network uses 2d convolutional layers.
"""
# pylint: disable=wrong-import-order
# pylint: disable=ungrouped-imports
# The imports are ordered by length.

import os
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from torch import nn
from torch import optim
from imgtxtgen.common.utils import mkdir_p, get_timestamp
from imgtxtgen.arch1.models.image_generator import ImageGenerator64, ImageGenerator256
from imgtxtgen.arch1.models.image_discriminator import ImageDiscriminator64, ImageDiscriminator256

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

        loaded_parameter_names = layers_from_to_mapping.values()
        untrained_parameters_g = []
        untrained_parameters_d = []
        for name, parameter in self.named_parameters():
            if name in loaded_parameter_names:
                continue
            if 'gen' in name:
                untrained_parameters_g.append(parameter)
            else:
                untrained_parameters_d.append(parameter)

        untrained_parameters = untrained_parameters_g, untrained_parameters_d
        return untrained_parameters

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
        """
        Run the network forward.
        """
        # pylint: disable=arguments-differ
        # pylint: disable=invalid-name
        # The arguments will differ from the base class since nn.Module is an abstract class.
        # Short variable names like x, h and c are fine in this context.

        images = self.img_gen(noise)
        pred_logits = self.img_dis(images)
        return pred_logits

def train_dcgan(dcgan, dataloader, device, d_batch, num_epochs, output_dir, print_every=100, save_results=True, config_name='generated', learning_rate=0.0002, untrained_parameters=None):
    """
    Train the DCGAN on the given dataset.
    """
    # pylint: disable=too-many-locals
    # pylint: disable=too-many-arguments
    # This number of local variables is necessary for training.
    # The arguments are necesarry for training.

    beta1 = 0.5
    d_noise = dcgan.d_noise
    s_noise = (d_batch, d_noise, 1, 1)
    real_labels = torch.full((d_batch,), 1, device=device)
    fake_labels = torch.full((d_batch,), 0, device=device)

    criterion = nn.BCEWithLogitsLoss()
    opt_g = optim.Adam(dcgan.img_gen.parameters(), lr=learning_rate, betas=(beta1, 0.999))
    opt_d = optim.Adam(dcgan.img_dis.parameters(), lr=learning_rate, betas=(beta1, 0.999))

    fixed_noise = torch.randn(s_noise, device=device)

    output_dir = os.path.join(output_dir, f'{config_name}_{get_timestamp()}')
    output_images_dir = os.path.join(output_dir, 'images')
    output_weights_dir = os.path.join(output_dir, 'weights')
    mkdir_p(output_images_dir)
    mkdir_p(output_weights_dir)

    dis_train_losses = []
    gen_train_losses = []
    gen_eval_losses = []

    dcgan.train()

    # if we are using pretrained weights from dcgan 64 only train untrained parameters
    if untrained_parameters is not None:
        untrained_parameters_g, untrained_parameters_d = untrained_parameters
        opt_g = optim.Adam(untrained_parameters_g, lr=learning_rate, betas=(beta1, 0.999))
        opt_d = optim.Adam(untrained_parameters_d, lr=learning_rate, betas=(beta1, 0.999))

    for epoch in range(1, num_epochs+1):
        print('epoch:', epoch)
        running_loss_g = 0
        running_loss_d = 0

        for i, batch in enumerate(dataloader, start=1):
            # Get batch.
            real_imgs = batch[0]
            real_imgs = real_imgs.to(device)
            with torch.no_grad():
                noise = torch.randn(s_noise, device=device)
                fake_imgs = dcgan.img_gen(noise)

            # Train discriminator
            real_logits = dcgan.img_dis(real_imgs).squeeze()
            fake_logits = dcgan.img_dis(fake_imgs).squeeze()

            loss_d_real = criterion(real_logits, real_labels)
            loss_d_fake = criterion(fake_logits, fake_labels)
            loss_d = loss_d_real + loss_d_fake

            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()

            # Train generator
            noise = torch.randn(s_noise, device=device)
            fake_imgs = dcgan.img_gen(noise)
            fake_logits = dcgan.img_dis(fake_imgs).squeeze()
            loss_g = criterion(fake_logits, real_labels)

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
                dcgan.eval()
                with torch.no_grad():
                    fake_imgs = dcgan.img_gen(fixed_noise)
                    pred_logits = dcgan.img_dis(fake_imgs).squeeze()
                    loss_g = criterion(pred_logits, real_labels)
                dcgan.train()
                print('evaluation loss_g:', loss_g.item())
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
