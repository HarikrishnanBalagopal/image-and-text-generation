"""
Module for InfoGAN.
Architecture taken from the InfoGAN paper:
https://arxiv.org/pdf/1606.03657.pdf
Architecture code:
https://github.com/openai/InfoGAN/blob/master/infogan/models/regularized_gan.py
"""
# pylint: disable=wrong-import-order
# pylint: disable=ungrouped-imports
# The imports are ordered by length.

import torch

from torch import nn
from torch.optim import Adam
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from torch.nn.functional import log_softmax
from torch.distributions.normal import Normal
from imgtxtgen.arch6.models.generator import Generator
from imgtxtgen.common.utils import mkdir_p, get_timestamp
from imgtxtgen.arch6.models.discriminator import Discriminator

class InfoGAN(nn.Module):
    """
    InfoGAN designed for MNIST 28 x 28 grayscale images from 10 classes.
    """

    def __init__(self):
        super().__init__()
        self.define_module()

    def define_module(self):
        """
        Define each part of the InfoGAN.
        """

        self.gen = Generator()
        self.dis = Discriminator()

    def forward(self, noise, labels, rest_code):
        """
        Run the network forward.
        """
        # pylint: disable=arguments-differ
        # The arguments will differ from the base class since nn.Module is an abstract class.

        images = self.gen(noise, labels, rest_code)
        valid_logits, label_logits, rest_means = self.dis(images)
        return valid_logits, label_logits, rest_means

    def sample_images(self, noise, labels, rest_code):
        """
        Sample images from the generator.
        """

        with torch.no_grad():
            images = self.gen(noise, labels, rest_code)
        return images

def get_categorical_log_probs(xs, logits):
    """
    Categorical distribution log probabilities.
    """
    # pylint: disable=invalid-name
    # Short names like xs are fine in this function.

    log_probs = log_softmax(logits, dim=1)
    return torch.gather(log_probs, dim=1, index=xs.unsqueeze(1))

def dis_loss_function(preds, targets, is_real):
    """
    Discriminator loss function.
    """

    validity_loss_fn = nn.BCEWithLogitsLoss()
    valid_logits, label_logits, rest_means = preds
    labels, rest_code = targets

    alpha = 0.1

    if is_real:
        validity_loss = validity_loss_fn(valid_logits, torch.ones_like(valid_logits))
        # unsupervised setting, no ground truths to calculate mutual info.
        label_info = 0.
        rest_info = 0.
    else:
        validity_loss = validity_loss_fn(valid_logits, torch.zeros_like(valid_logits))
        label_info = get_categorical_log_probs(labels, logits=label_logits).mean()
        rest_info = 0. # Normal(loc=rest_means, scale=1).log_prob(rest_code).mean()

    mutual_info = label_info + rest_info

    return validity_loss - alpha * mutual_info, label_info, rest_info

def gen_loss_function(preds, targets):
    """
    Generator loss function.
    """

    validity_loss_fn = nn.BCEWithLogitsLoss()
    valid_logits, label_logits, rest_means = preds
    labels, rest_code = targets

    alpha = 0.1

    validity_loss = validity_loss_fn(valid_logits, torch.ones_like(valid_logits))
    label_info = get_categorical_log_probs(labels, logits=label_logits).mean()
    rest_info = 0. # Normal(loc=rest_means, scale=1).log_prob(rest_code).mean()

    mutual_info = label_info + rest_info

    return validity_loss - alpha * mutual_info, label_info, rest_info

def dbg_plot(dbg_dis_fake_label_info, dbg_dis_fake_rest_info, dbg_gen_label_info, dbg_gen_rest_info, epoch, i, output_dir):
    """
    Debug the mutual information estimates.
    """
    plt.close()
    plt.plot(dbg_dis_fake_label_info, label='dis fake label')
    plt.plot(dbg_dis_fake_rest_info, label='dis fake rest')
    plt.plot(dbg_gen_label_info, label='gen label')
    plt.plot(dbg_gen_rest_info, label='gen rest')
    plt.legend()
    plt.savefig(f'{output_dir}info_{epoch}_{i}.png')

def train(model, data_loader, device, d_batch, num_epochs=20, print_every=10):
    """
    Train the InfoGAN model on MNIST.
    """

    gen_opt = Adam(model.gen.parameters(), lr=1e-3)
    dis_opt = Adam(model.dis.parameters(), lr=2e-4)

    fixed_noise, fixed_labels, fixed_rest_code = model.gen.sample_latent(64, device=device)
    output_dir = f'./outputs/arch6/infogan_MNIST_{get_timestamp()}/'
    mkdir_p(output_dir)

    dbg_dis_fake_label_info = []
    dbg_dis_fake_rest_info = []

    dbg_gen_label_info = []
    dbg_gen_rest_info = []

    reg_alpha = 1. # add some noise to prevent discriminator using noise to detect latent code.

    model.train()

    for epoch in range(1, num_epochs+1):
        print('epoch:', epoch)

        for i, (real_imgs, _) in enumerate(data_loader, start=1):

            # Prepare the batch.
            real_imgs = real_imgs.to(device)
            real_imgs += reg_alpha * torch.rand_like(real_imgs, device=device) # regularization
            noise, fake_labels, rest_code = model.gen.sample_latent(d_batch, device=device)
            fake_imgs = model.sample_images(noise, fake_labels, rest_code)
            fake_imgs += reg_alpha * torch.rand_like(fake_imgs, device=device) # regularization

            #  Train Discriminator
            real_preds = model.dis(real_imgs)
            fake_preds = model.dis(fake_imgs)

            dis_loss_real, _, _ = dis_loss_function(real_preds, (None, None), is_real=True)

            dis_loss_fake, label_info, rest_info = dis_loss_function(fake_preds, (fake_labels, rest_code), is_real=False)
            dbg_dis_fake_label_info.append(label_info)
            dbg_dis_fake_rest_info.append(rest_info)

            dis_loss = dis_loss_real + dis_loss_fake

            dis_opt.zero_grad()
            dis_loss.backward()
            dis_opt.step()

            #  Train Generator
            noise, fake_labels, rest_code = model.gen.sample_latent(d_batch, device=device)
            fake_imgs = model.gen(noise, fake_labels, rest_code)
            fake_imgs += reg_alpha * torch.rand_like(fake_imgs, device=device) # regularization
            fake_preds = model.dis(fake_imgs)

            gen_loss, label_info, rest_info = gen_loss_function(fake_preds, (fake_labels, rest_code))
            dbg_gen_label_info.append(label_info)
            dbg_gen_rest_info.append(rest_info)

            gen_opt.zero_grad()
            gen_loss.backward()
            gen_opt.step()

            if i % print_every == 0:
                print('dis loss:', dis_loss)
                print('gen loss:', gen_loss)
                model.eval()
                fake_imgs = model.sample_images(fixed_noise, fixed_labels, fixed_rest_code)
                model.train()
                filename = f'{output_dir}gen_{epoch}_{i}.png'
                save_image(fake_imgs, filename)
                weights_filename = f'{output_dir}weights_{epoch}_{i}.pth'
                torch.save(model.state_dict(), weights_filename)
                dbg_plot(dbg_dis_fake_label_info, dbg_dis_fake_rest_info, dbg_gen_label_info, dbg_gen_rest_info, epoch, i, output_dir)
