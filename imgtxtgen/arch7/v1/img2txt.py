"""
1. CNN + RNN trained with MLE.
"""
import torch
import visdom as vis

from torch import nn
from torch import optim
from torchvision import transforms
from torch.nn.utils.rnn import pack_padded_sequence
from imgtxtgen.arch7.models.image_encoder import ImageEncoder1

class Img2Txt(nn.Module):
    """
    1. CNN + RNN trained with MLE.
    """

    def __init__(self, d_vocab, d_embed, d_hidden, start_token, end_token, fine_tune_encoder=False, d_max_seq_len=50):
        super().__init__()
        self.d_vocab = d_vocab
        self.d_embed = d_embed
        self.d_hidden = d_hidden
        self.end_token = end_token
        self.start_token = start_token
        self.d_max_seq_len = d_max_seq_len
        self.fine_tune_encoder = fine_tune_encoder
        self.define_module()

    def define_module(self):
        """
        Define each part of the Img2Txt network.
        """

        self.enc = ImageEncoder1()
        self.fc_h = nn.Linear(2048, self.d_hidden)
        self.embed = nn.Embedding(num_embeddings=self.d_vocab, embedding_dim=self.d_embed)
        self.rnn = nn.GRU(input_size=self.d_embed, hidden_size=self.d_hidden, num_layers=1, batch_first=True)
        self.fc_logits = nn.Linear(self.d_hidden, self.d_vocab)

    def forward(self, images, captions, cap_lens):
        """
        Forward propagation.
        :param images: a float tensor of dimensions (d_batch, 3, d_image_size, d_image_size)
        :param captions: a integer tensor of dimensions (d_batch, -1) containing the padded captions.
        :param cap_lens: a integer tensor of dimensions (d_batch,) containing the valid lengths of each caption.
        :return: encoded images (d_batch, 2048)
        """
        # pylint: disable=arguments-differ
        # The arguments will differ from the base class since nn.Module is an abstract class.

        if self.fine_tune_encoder:
            features = self.enc(images)
        else:
            with torch.no_grad():
                features = self.enc(images)
        hiddens = self.fc_h(features).unsqueeze(0) #  (num_layers * num_directions, batch, hidden_size)
        embeddings = self.embed(captions)
        embeddings = pack_padded_sequence(embeddings, cap_lens, batch_first=True)
        ouputs, _ = self.rnn(embeddings, hiddens)
        logits = self.fc_logits(ouputs[0])
        return logits

    def sample(self, images):
        """
        Sample some captions given images.
        """

        device = images.device
        d_max_seq_len = self.d_max_seq_len
        captions = []
        with torch.no_grad():
            features = self.enc(images)
            hiddens = self.fc_h(features)
            for hidden in hiddens:
                caption = []
                hidden = hidden.view(1, 1, -1)
                word = torch.full((1, 1), self.start_token, dtype=torch.long, device=device)
                while len(caption) < d_max_seq_len and word.item() != self.end_token:
                    embedding = self.embed(word)
                    ouput, hidden = self.rnn(embedding, hidden)
                    logits = self.fc_logits(ouput)
                    _, word = logits.squeeze().max(dim=0)
                    caption.append(word.item())
                    word = word.view(1, 1)
                captions.append(caption)

        return captions

def train(model, dataset, data_loader, device, num_epochs=20, print_every=25):
    """
    Train using MLE.
    """
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    losses = []
    running_loss = 0
    visualization = vis.Visdom()
    v1_env = 'Img2TxtV1'
    img_win = None
    text_win = None
    loss_win = None

    unnormalize_transform = transforms.Normalize((-1, -1, -1), (2, 2, 2))

    for epoch in range(1, num_epochs+1):
        print('epoch:', epoch)
        for i, (images, captions, cap_lens) in enumerate(data_loader, start=1):
            images, captions, cap_lens = images.to(device), captions.to(device), cap_lens.to(device)
            logits = model(images, captions, cap_lens - 1) # don't include <end> token
            targets = pack_padded_sequence(captions[:, 1:], cap_lens - 1, batch_first=True)[0] # don't include <start> token
            loss = criterion(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % print_every == 0:
                running_loss /= print_every
                print('i:', i, 'running_loss:', running_loss)
                losses.append(running_loss)
                running_loss = 0
                captions = [' '.join(dataset.decode(caption)) for caption in captions[:4]]
                predicted = model.sample(images)
                predicted = [' '.join(dataset.decode(p)) for p in predicted[:4]]

                images = torch.stack([unnormalize_transform(image) for image in images[:4]])
                img_win = visualization.images(images, env=v1_env, win=img_win)
                captions_html = '<h1>Targets:</h1>' + '<br/>'.join(captions) + '<h1>Predictions:</h1>' + '<br/>'.join(predicted)
                text_win = visualization.text(captions_html, env=v1_env, win=text_win)
                loss_win = visualization.line(Y=losses, env=v1_env, win=loss_win)
