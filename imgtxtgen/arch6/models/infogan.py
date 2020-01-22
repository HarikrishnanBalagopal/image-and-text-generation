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

from numpy import pi
from torch import nn
from torch.optim import Adam
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from torch.nn.functional import log_softmax
# from torch.distributions.normal import Normal
from imgtxtgen.arch6.models.generator import Generator
from imgtxtgen.common.utils import mkdir_p, get_timestamp
from imgtxtgen.arch6.models.discriminator import Discriminator, DHead, QHead

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
        self.d_head = DHead()
        self.q_head = QHead()

    def forward(self, noise, labels, rest_code):
        """
        Run the network forward.
        """
        # pylint: disable=arguments-differ
        # The arguments will differ from the base class since nn.Module is an abstract class.

        images = self.gen(noise, labels, rest_code)
        features = self.dis(images)
        valid_logits = self.d_head(features)
        label_logits, rest_means, rest_vars = self.q_head(features)
        return valid_logits, label_logits, rest_means, rest_vars

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

# THIS IS NUMERICALLY UNSTABLE, GIVES NAN AT START OF TRAINING.
# TAKEN FROM https://github.com/Natsu6767/InfoGAN-PyTorch/blob/4586919f2821b9b2e4aeff8a07c5003a5905c7f9/utils.py#L15-L28
def normal_nll_loss(xs, mu, var):
    """
    Calculate the negative log likelihood
    of normal distribution.
    This needs to be minimised.
    Treating Q(cj | x) as a factored Gaussian.
    """
    # pylint: disable=invalid-name
    # Short names like xs, mu, etc. are fine in this function.

    logli = -0.5 * (var.mul(2 * pi) + 1e-6).log() - (xs - mu).pow(2).div(var.mul(2.0) + 1e-6)
    nll = -(logli.sum(1).mean())

    return nll

# def normal_nll_loss(xs, mu, var):
#     """
#     Calculate the negative log likelihood
#     of normal distribution.
#     This needs to be minimised.
#     Treating Q(cj | x) as a factored Gaussian.
#     """
#     # pylint: disable=invalid-name
#     # Short names like xs, mu, etc. are fine in this function.

#     return Normal(loc=mu, scale=var.sqrt()).log_prob(xs).sum(dim=1).mean()

def dbg_plot(metrics, filename):
    """
    Debug the mutual information estimates.
    """
    (metrics_gen_loss, metrics_dis_loss, metrics_gen_loss_fake, metrics_gen_loss_labels, metrics_gen_loss_rest, metrics_dis_loss_real, metrics_dis_loss_fake) = metrics
    plt.close()
    plt.plot(metrics_gen_loss, label='gen_loss')
    plt.plot(metrics_dis_loss, label='dis_loss')
    plt.plot(metrics_gen_loss_fake, label='gen_loss_fake')
    plt.plot(metrics_gen_loss_labels, label='gen_loss_labels')
    plt.plot(metrics_gen_loss_rest, label='gen_loss_rest')
    plt.plot(metrics_dis_loss_real, label='dis_loss_real')
    plt.plot(metrics_dis_loss_fake, label='dis_loss_fake')
    plt.legend()
    plt.savefig(filename)

def train(model, data_loader, device, d_batch, num_epochs=20, print_every=10):
    """
    Train the InfoGAN model on MNIST.
    """

    model.train()

    gen_opt = Adam(list(model.gen.parameters()) + list(model.q_head.parameters()), lr=1e-3)
    dis_opt = Adam(list(model.dis.parameters()) + list(model.d_head.parameters()), lr=2e-4)

    fixed_noise, fixed_labels, fixed_rest_code = model.gen.sample_latent(64, device=device)
    output_dir = f'./outputs/arch6/infogan_MNIST_{get_timestamp()}/'
    mkdir_p(output_dir)

    validity_loss_fn = nn.BCEWithLogitsLoss()
    labels_loss_fn = nn.CrossEntropyLoss()

    real_targets = torch.ones(d_batch, 1, device=device)
    fake_targets = torch.zeros(d_batch, 1, device=device)

    metrics_gen_loss = []
    metrics_dis_loss = []

    metrics_gen_loss_fake = []
    metrics_gen_loss_labels = []
    metrics_gen_loss_rest = []

    metrics_dis_loss_real = []
    metrics_dis_loss_fake = []

    # reg_alpha = 1. # add some noise to prevent discriminator using noise to detect latent code.

    for epoch in range(1, num_epochs+1):
        print('epoch:', epoch)

        for i, (real_imgs, _) in enumerate(data_loader, start=1):

            # Prepare the batch.
            real_imgs = real_imgs.to(device)
            # real_imgs += reg_alpha * torch.rand_like(real_imgs, device=device) # regularization
            noise, fake_labels, rest_code = model.gen.sample_latent(d_batch, device=device)
            fake_imgs = model.gen(noise, fake_labels, rest_code)
            # fake_imgs += reg_alpha * torch.rand_like(fake_imgs, device=device) # regularization

            #  Train Discriminator
            dis_opt.zero_grad()

            real_features = model.dis(real_imgs)
            real_valid_logits = model.d_head(real_features)
            dis_loss_real = validity_loss_fn(real_valid_logits, real_targets)
            dis_loss_real.backward()

            fake_features = model.dis(fake_imgs.detach())
            fake_valid_logits = model.d_head(fake_features)
            dis_loss_fake = validity_loss_fn(fake_valid_logits, fake_targets)
            dis_loss_fake.backward()

            dis_loss = dis_loss_real + dis_loss_fake

            dis_opt.step()

            metrics_dis_loss.append(dis_loss.item())
            metrics_dis_loss_real.append(dis_loss_real.item())
            metrics_dis_loss_fake.append(dis_loss_fake.item())

            #  Train Generator
            gen_opt.zero_grad()

            fake_features = model.dis(fake_imgs)
            fake_valid_logits = model.d_head(fake_features)
            gen_loss_fake = validity_loss_fn(fake_valid_logits, real_targets)

            label_logits, rest_means, rest_vars = model.q_head(fake_features)
            gen_loss_labels = labels_loss_fn(label_logits, fake_labels)
            gen_loss_rest = normal_nll_loss(rest_code, rest_means, rest_vars) * 0.1 # scale continuous part loss to avoid overwhelming the other losses.

            gen_loss = gen_loss_fake + gen_loss_labels + gen_loss_rest
            gen_loss.backward()

            gen_opt.step()

            metrics_gen_loss.append(gen_loss.item())
            metrics_gen_loss_fake.append(gen_loss_fake.item())
            metrics_gen_loss_labels.append(gen_loss_labels.item())
            metrics_gen_loss_rest.append(gen_loss_rest.item())

            if i % print_every == 0:
                # print('dis loss:', dis_loss)
                # print('gen loss:', gen_loss)
                print('debug gen_loss_rest:', gen_loss_rest)
                model.eval()
                fake_imgs = model.sample_images(fixed_noise, fixed_labels, fixed_rest_code)
                model.train()
                imgs_filename = f'{output_dir}gen_{epoch}_{i}.png'
                save_image(fake_imgs, imgs_filename)
                weights_filename = f'{output_dir}weights_{epoch}_{i}.pth'
                torch.save(model.state_dict(), weights_filename)
                metrics = (metrics_gen_loss, metrics_dis_loss, metrics_gen_loss_fake, metrics_gen_loss_labels, metrics_gen_loss_rest, metrics_dis_loss_real, metrics_dis_loss_fake)
                metrics_filename = f'{output_dir}info_{epoch}_{i}.png'
                dbg_plot(metrics, metrics_filename)
