"""
Train DCGAN for 256 x 256 images using pretrained weights from DCGAN for 64 x 64 images.
"""
# pylint: disable=wrong-import-order
# pylint: disable=ungrouped-imports
# The imports are ordered by length.

import os
import torch
import argparse

from torchvision import transforms
from imgtxtgen.common.datasets.cub2011 import CUB2011Dataset
from imgtxtgen.arch1.models.dcgan import DCGAN256, train_dcgan

def parse_args():
    """
    Parse command line arguments.
    """

    weights_path = os.path.join('.', 'output', 'generated_2020_01_09_03_06_33', 'weights', 'gen_epoch_200_iter_200.pth')
    dataset_path = os.path.join('..', '..', '..', 'exp2', 'CUB_200_2011')
    captions_path = os.path.join('..', '..', '..', 'exp2', 'cub_with_captions')
    outputs_path = os.path.join('.', 'output', 'arch1')

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, help='Path to DCGAN64 weights pretrained on CUB 2011 dataset.', default=weights_path)
    parser.add_argument('--dataset', type=str, help='Path to CUB2011 directory.', default=dataset_path)
    parser.add_argument('--captions', type=str, help='Path to captions directory.', default=captions_path)
    parser.add_argument('--output', type=str, help='Directory to store training outputs.', default=captions_path)

    args = parser.parse_args()

    assert os.path.isfile(args.weights), f'weights path is not a file:{args.weights}'
    assert os.path.isdir(args.dataset), f'CUB2011 path is not a directory:{args.dataset}'
    assert os.path.isdir(args.captions), f'Captions path is not a directory:{args.captions}'

    return args

def train_dcgan_256(args):
    """
    This train DCGAN256 on CUB 2011 dataset using weights from DCGAN64 model pretrained on the same dataset.
    """
    # pylint: disable=too-many-locals
    # This number of local variables is necessary for training.

    gpu_id = 1
    d_gen = 16
    d_dis = 16
    d_batch = 16
    d_noise = 100
    d_image_size = 256
    d_max_seq_len = 18
    pretrained_weights_64_path = args.weights

    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    print('running on device:', device)

    print('loading pretrained weights of dcgan 64:')
    pretrained_weights_64 = torch.load(pretrained_weights_64_path)
    # print('dcgan 64 weights:')
    # for k, v in pretrained_weights_64.items():
    #     print(f'{k:40}|{str(v.size()):40}')

    print('constructing DCGAN256 model:')
    model = DCGAN256(d_noise=d_noise, d_gen=d_gen, d_dis=d_dis).to(device)
    # print('dcgan 256 weights:')
    # model_dict = model.state_dict()
    # for k, v in model_dict.items():
    #     print(f'{k:40}|{str(v.size()):40}')

    model.load_weights_from_dcgan_64(pretrained_weights_64)
    print('loaded pretrained weights from dcgan 64')

    print('testing the model with a forward pass:')
    with torch.no_grad():
        s_noise = (d_batch, d_noise, 1, 1)
        noise = torch.randn(s_noise, device=device)
        pred_logits = model(noise)
        print('noise:', noise.size(), 'pred_logits:', pred_logits.size())

    print('training on cub2011:')
    cub2011_dataset_dir = args.dataset
    cub2011_captions_dir = args.captions
    img_transforms = transforms.Compose([
        transforms.Resize((d_image_size, d_image_size)),
        transforms.ToTensor(),
        transforms.Normalize((.5, .5, .5), (.5, .5, .5))
    ])
    cub2011_dataset = CUB2011Dataset(dataset_dir=cub2011_dataset_dir, captions_dir=cub2011_captions_dir, img_transforms=img_transforms, d_max_seq_len=d_max_seq_len)
    dataloader = torch.utils.data.DataLoader(cub2011_dataset, batch_size=d_batch, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    train_dcgan(model, dataloader, device=device, d_batch=d_batch, num_epochs=200, config_name='dcgan_256_cub2011_using_pretrained_64')

if __name__ == '__main__':
    train_dcgan_256(parse_args())
