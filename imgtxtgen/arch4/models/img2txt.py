"""
Image captioning model.
"""

import os
import torch
import numpy as np

from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from imgtxtgen.common.utils import get_timestamp, mkdir_p
from imgtxtgen.arch4.models.text_decoder import TextDecoder
from imgtxtgen.arch4.models.image_encoder import ImageEncoder

class Img2Txt(nn.Module):
    """
    Takes images and generates captions.
    """

    def __init__(self, d_embed, d_hidden, d_vocab, d_layers, d_max_seq_len=20):
        super().__init__()
        self.d_embed = d_embed
        self.d_hidden = d_hidden
        self.d_vocab = d_vocab
        self.d_layers = d_layers
        self.d_max_seq_len = d_max_seq_len

        self.define_module()

    def define_module(self):
        """
        Define each part of Img2Txt.
        """

        self.enc = ImageEncoder(self.d_embed)
        self.dec = TextDecoder(self.d_embed, self.d_hidden, self.d_vocab, self.d_layers, self.d_max_seq_len)

    def forward(self, images, captions, lengths):
        """
        Run the network.
        """
        # pylint: disable=arguments-differ
        # The arguments will differ from the base class since nn.Module is an abstract class.

        features = self.enc(images)
        outputs = self.dec(features, captions, lengths)
        return outputs

    def sample(self, images):
        """
        Sample some captions given images.
        """

        with torch.no_grad():
            features = self.enc(images)
            captions = self.dec.sample(features)
        return captions

def train_img2txt(encoder, decoder, data_loader, device, learning_rate=0.01, num_epochs=20, print_every=100, save_every=100, output_dir='.', config_name='img2txt', debug_dataset=None):
    """
    Train the image captioning model.
    """
    output_dir = os.path.join(output_dir, f'{config_name}_{get_timestamp()}')
    mkdir_p(output_dir)
    print('saving to output directory:', output_dir)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.fc.parameters()) + list(encoder.bn.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate)

    # Train the models
    total_step = len(data_loader)

    for epoch in range(1, num_epochs+1):
        for i, (images, captions, lengths, _) in enumerate(data_loader, start=1):
            # Set mini-batch dataset
            images, captions = images.to(device), captions.to(device)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

            # Forward, backward and optimize
            features = encoder(images)
            outputs = decoder(features, captions, lengths)
            loss = criterion(outputs, targets)
            encoder.zero_grad()
            decoder.zero_grad()
            loss.backward()
            optimizer.step()

            # Print log info
            if i % print_every == 0:
                print(f'Epoch [{epoch}/{num_epochs}], Step [{i}/{total_step}], Loss: {loss.item():.4f}, Perplexity: {np.exp(loss.item()):5.4f}')
                with torch.no_grad():
                    predicted = decoder.sample(features)
                print('captions[0]:', debug_dataset.dict.decode(captions[0]))
                print('predicted[0]:', debug_dataset.dict.decode(predicted[0]))

            # Save the model checkpoints
            if i % save_every == 0:
                torch.save(encoder.state_dict(), os.path.join(output_dir, f'encoder_{epoch}_{i}.pth'))
                torch.save(decoder.state_dict(), os.path.join(output_dir, f'decoder_{epoch}_{i}.pth'))
