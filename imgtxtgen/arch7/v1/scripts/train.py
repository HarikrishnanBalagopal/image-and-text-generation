"""
Train v1 on flickr30k dataset.
"""

import torch

from imgtxtgen.arch7.v1.img2txt import Img2Txt, train
from imgtxtgen.common.datasets.flickr30k import get_default_flickr30k_loader

def main():
    """
    Train v1 on flickr30k dataset.
    """

    gpu_id = 1
    d_batch = 64
    d_embed = 256
    d_hidden = 256
    d_image_size = 256
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    dataset, train_loader = get_default_flickr30k_loader(d_batch=d_batch, d_image_size=d_image_size)
    model = Img2Txt(dataset.d_vocab, d_embed, d_hidden, dataset.start_token, dataset.end_token).to(device)

    train(model, dataset, train_loader, device)

if __name__ == '__main__':
    main()
