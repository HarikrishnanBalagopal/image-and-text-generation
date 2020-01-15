"""
Image captioning model.
"""

import os
import torch
import numpy as np

from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

def train_img2txt(encoder, decoder, data_loader, device, learning_rate=0.01, num_epochs=20, print_every=100, save_every=100, output_dir='.'):
    """
    Train the image captioning model.
    """

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.fc.parameters()) + list(encoder.bn.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate)

    # Train the models
    total_step = len(data_loader)

    for epoch in range(1, num_epochs+1):
        for i, (images, captions, lengths) in enumerate(data_loader, start=1):

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

            # Save the model checkpoints
            if i % save_every == 0:
                torch.save(decoder.state_dict(), os.path.join(output_dir, 'decoder-{}-{}.ckpt'.format(epoch+1, i+1)))
                torch.save(encoder.state_dict(), os.path.join(output_dir, 'encoder-{}-{}.ckpt'.format(epoch+1, i+1)))
