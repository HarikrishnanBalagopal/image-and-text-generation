"""
Sample some captions from Img2Txt model.
"""
# pylint: disable=wrong-import-order
# pylint: disable=ungrouped-imports
# The imports are ordered by length.

import os
import torch
import argparse
import matplotlib.pyplot as plt

from PIL import Image
from imgtxtgen.arch4.models.img2txt import Img2Txt
from imgtxtgen.common.datasets.cub2011 import get_cub2011_data_loader
from imgtxtgen.common.utils import mkdir_p, get_standard_img_transforms

# Device configuration

def load_image(image_path):
    """
    Load an image and convert to pytorch channels last tensor format.
    """
    img_transforms = get_standard_img_transforms()
    image = Image.open(image_path)
    images = img_transforms(image).unsqueeze(0)
    return images

def parse_args():
    """
    Parse command line arguments.
    """

    parser = argparse.ArgumentParser()
    img_path = '/users/gpu/haribala/code/datasets/CUB_200_2011/CUB_200_2011/images/001.Black_footed_Albatross/' # Black_Footed_Albatross_0001_796111.jpg
    output_path = os.path.join('.', 'outputs', 'arch4', 'img2txt_samples', '001.Black_footed_Albatross')
    encoder_path = os.path.join('.', 'outputs', 'arch4', 'img2txt_2020_01_16_09_34_45', 'encoder_400_40.pth')
    decoder_path = os.path.join('.', 'outputs', 'arch4', 'img2txt_2020_01_16_09_34_45', 'decoder_400_40.pth')

    parser.add_argument('--images', type=str, default=img_path, help='Directory containing the input images.')
    parser.add_argument('--outputs', type=str, default=output_path, help='Directory to store outputs.')
    parser.add_argument('--encoder_path', type=str, default=encoder_path, help='Path to trained encoder weights.')
    parser.add_argument('--decoder_path', type=str, default=decoder_path, help='Path to trained decoder weights.')

    # Model parameters (should be same as paramters in train.py)
    parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')

    args = parser.parse_args()

    return args

def sample(args):
    """
    Sample from Img2Txt model.
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    _, dataset = get_cub2011_data_loader()
    d_vocab = len(dataset.dict)

    # Build models
    model = Img2Txt(d_embed=args.embed_size, d_hidden=args.hidden_size, d_vocab=d_vocab, d_layers=args.num_layers).eval().to(device)  # eval mode (batchnorm uses moving mean/variance)

    # Load the trained model parameters
    model.enc.load_state_dict(torch.load(args.encoder_path))
    model.dec.load_state_dict(torch.load(args.decoder_path))

    # Prepare images.
    images_dir = args.images
    assert os.path.isdir(images_dir), f'{images_dir} is not a directory.'
    img_filenames = os.listdir(images_dir)
    img_filenames = [f for f in img_filenames if os.path.splitext(f)[1] in {'.jpg', '.png'}]
    img_transforms = get_standard_img_transforms()
    images = []
    for img_filename in img_filenames:
        img_path = os.path.join(images_dir, img_filename)
        img = Image.open(img_path)
        img = img_transforms(img).to(device)
        images.append(img)
    images = torch.stack(images, dim=0)
    print('images:', images.size())

    # Generate captions.
    with torch.no_grad():
        captions = model.sample(images).cpu()
    captions = [dataset.dict.decode(c) for c in captions]

    # Print out the generated captions.
    print('The generated captions:')
    print(captions)

    # Create and save captioned images.
    outputs_dir = args.outputs
    mkdir_p(outputs_dir)
    for img_filename, caption in zip(img_filenames, captions):
        img_path = os.path.join(images_dir, img_filename)
        out_path = os.path.join(outputs_dir, img_filename)
        plt.imshow(plt.imread(img_path))
        plt.title(caption)
        plt.savefig(out_path)

if __name__ == '__main__':
    sample(parse_args())
