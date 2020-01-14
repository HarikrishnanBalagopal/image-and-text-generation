"""
GAN for generating text conditioned on images.
To be trained using reinforcement learning.
"""

import torch

from torch import nn
from torch.optim import Adam
from imgtxtgen.arch2.models.text_generator import TextGenerator
from imgtxtgen.arch2.models.text_discriminator import TextDiscriminator

class Img2TextGAN(nn.Module):
    """
    Model to generate texts conditioned on images.
    Uses a conditional generator and discriminator.
    To be trained using reinforcement learning.
    """

    def __init__(self, d_vocab, d_embed, d_hidden, d_max_seq_len, d_image_features, d_noise, end_token=0, start_token=1):
        super().__init__()
        self.d_vocab = d_vocab
        self.d_embed = d_embed
        self.d_noise = d_noise
        self.d_hidden = d_hidden
        self.end_token = end_token
        self.start_token = start_token
        self.d_max_seq_len = d_max_seq_len
        self.d_image_features = d_image_features

        self.define_module()

    def define_module(self):
        """
        Define each part of TextGAN.
        """
        gen_opts = {
            'd_vocab':self.d_vocab,
            'd_embed':self.d_embed,
            'd_noise':self.d_noise,
            'd_hidden':self.d_hidden,
            'end_token':self.end_token,
            'start_token':self.start_token,
            'd_max_seq_len':self.d_max_seq_len,
            'd_image_features':self.d_image_features
        }
        self.gen = TextGenerator(**gen_opts)
        self.dis = TextDiscriminator(d_vocab=self.d_vocab, d_embed=self.d_embed, d_hidden=self.d_hidden)

    def forward(self, images, noise):
        """
        Run the network forward.
        """
        # pylint: disable=arguments-differ
        # pylint: disable=invalid-name
        # pylint: disable=bad-whitespace
        # The arguments will differ from the base class since nn.Module is an abstract class.
        # Short variable names like x, h and c are fine in this context. The whitespace makes it more readable.
        captions, _, _ = self.gen(images, noise)
        preds = self.dis(captions)
        return captions, preds

def train_img2txt_gen(model, dataset, d_batch, device, num_rollouts=16, num_epochs=20, debug_dataset=None, print_every=10):
    """
    Train the Img2TextGAN using reinforcement learning.
    num_rollouts : Number of times to rollout at each time instance.
    """

    beta1 = 0.5
    learning_rate = 0.0002
    s_noise = (d_batch, model.d_noise)
    real_labels = torch.full((d_batch,), 1, device=device)
    fake_labels = torch.full((d_batch,), 0, device=device)
    opt_g = Adam(model.gen.parameters(), lr=learning_rate, betas=(beta1, 0.999))
    opt_d = Adam(model.dis.parameters(), lr=learning_rate, betas=(beta1, 0.999))

    criterion = nn.BCEWithLogitsLoss()

    model.train()

    for epoch in range(1, num_epochs+1):
        print('epoch:', epoch)

        for i, batch in enumerate(dataset):

            images, real_captions, _ = batch
            images, real_captions = images.to(device), real_captions.to(device)

            # train discriminator
            noise = torch.randn(s_noise, device=device)

            with torch.no_grad():
                fake_captions, fake_captions_log_probs, hiddens = model.gen(images, noise)

            if i % print_every == 0 and debug_dataset is not None:
                print('i:', i)
                print('real_captions:', debug_dataset.dict.decode(real_captions[0]))
                print('fake_captions:', debug_dataset.dict.decode(fake_captions[0]))

            loss_d_real = criterion(model.dis(real_captions), real_labels)
            loss_d_fake = criterion(model.dis(fake_captions), fake_labels)
            loss_d = loss_d_real + loss_d_fake

            print('loss_d:', loss_d)

            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()

            # train generator
            noise = torch.randn(s_noise, device=device)
            fake_captions, fake_captions_log_probs, hiddens = model.gen(images, noise)
            rewards = torch.zeros((d_batch, model.d_max_seq_len), device=device)

            with torch.no_grad():
                for len_so_far in range(model.d_max_seq_len - 1, -1, -1):
                    # len_so_far goes through all valid indices of the time dimension in reverse.
                    # We do Monte Carlo rollouts in reverse to avoid having to copy captions.
                    for _ in range(num_rollouts):
                        model.gen.text_gen.complete_sequence(fake_captions, hiddens, len_so_far) # changes captions in place.
                        rewards[:, len_so_far] += model.dis(fake_captions)

                rewards /= num_rollouts

            # fake_captions_log_probs : d_batch x d_vocab x d_max_seq_len
            # fake_captions           : d_batch x d_max_seq_len -> d_batch x 1 x d_max_seq_len
            # relevant_log_probs      : d_batch x 1 x d_max_seq_len -> d_batch x d_max_seq_len now same size as rewards.
            relevant_log_probs = torch.gather(input=fake_captions_log_probs, dim=1, index=fake_captions.unsqueeze(1)).squeeze()
            loss_g = -(relevant_log_probs * rewards).mean() # negative expected reward over all time instances and sentences in the batch.

            print('loss_g:', loss_g)

            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()
