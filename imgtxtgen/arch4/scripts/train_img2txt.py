"""
Train image captioning model.
"""
# pylint: disable=wrong-import-order
# pylint: disable=ungrouped-imports
# The imports are ordered by length.

import os
import torch
import argparse

from imgtxtgen.arch4.models.img2txt import train_img2txt
from imgtxtgen.arch4.models.text_decoder import TextDecoder
from imgtxtgen.arch4.models.image_encoder import ImageEncoder
from imgtxtgen.common.datasets.cub2011 import get_cub2011_data_loader
from imgtxtgen.common.utils import mkdir_p, get_standard_img_transforms

def parse_args():
    """
    Parse command line arguments.
    """
    output_path = os.path.join('.', 'outputs', 'arch4')

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='ID of the GPU to use.')
    parser.add_argument('--outputs', type=str, default=output_path, help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=224, help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default='data/resized2014', help='directory for resized images')
    parser.add_argument('--caption_path', type=str, default='data/annotations/captions_train2014.json', help='path for train annotation json file')
    parser.add_argument('--print_every', type=int, default=20, help='step size for prining log info')
    parser.add_argument('--save_every', type=int, default=20, help='step size for saving trained models')

    # Model parameters
    parser.add_argument('--embed_size', type=int, default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int, default=1, help='number of layers in lstm')

    parser.add_argument('--num_epochs', type=int, default=400)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.001)

    args = parser.parse_args()

    return args

def train(args):
    """
    Train using parsed arguments.
    """
    # Device configuration
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    # Create model directory
    mkdir_p(args.outputs)

    # Image preprocessing, normalization for the pretrained resnet
    # img_transforms = transforms.Compose([
    #     transforms.RandomCrop(args.crop_size),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    # ])
    d_image_size = 256
    img_transforms = get_standard_img_transforms(d_image_size=d_image_size)

    # Build data loader
    print('loading cub2011:')
    data_loader, dataset = get_cub2011_data_loader(d_batch=args.batch_size, img_transforms=img_transforms) #, img_transforms=img_transforms)
    d_vocab = len(dataset.dict)

    # Build the models
    print('constructing models:')
    encoder = ImageEncoder(args.embed_size).to(device)
    decoder = TextDecoder(args.embed_size, args.hidden_size, d_vocab, args.num_layers).to(device)

    print('training on cub2011:')
    train_img2txt(encoder=encoder,
                  decoder=decoder,
                  data_loader=data_loader,
                  learning_rate=args.learning_rate,
                  num_epochs=args.num_epochs,
                  device=device,
                  print_every=args.print_every,
                  save_every=args.save_every,
                  output_dir=args.outputs,
                  debug_dataset=dataset)

if __name__ == '__main__':
    train(parse_args())
