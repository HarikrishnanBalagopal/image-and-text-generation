"""
Sample from InfoGAN.
"""

import torch
import matplotlib.pyplot as plt

from torch.nn.functional import sigmoid
from imgtxtgen.arch6.models.infogan import InfoGAN
from imgtxtgen.common.utils import mkdir_p, get_timestamp

def plot_and_save_info(images, results, filename):
    """
    Plot 8x8 grid of generated images along with a tuple caption:
    (actual label, is_real, predicted label)
    """

    images = images.squeeze(1)
    _, axs = plt.subplots(8, 8, figsize=(8, 8))
    for r in range(8):
        for c in range(8):
            idx = r * 8 + c
            ax = axs[r, c]
            ax.imshow(images[idx])
            res = results[idx]
            ax.set_title(f'{res[0]}, {res[1]}, {res[2]}')
            ax.axis('off')
    plt.tight_layout()
    plt.savefig(filename)

def sample():
    """
    Sample from InfoGAN.
    """

    gpu_id = 1
    d_batch = 64
    # class_id = 3
    # weights_path = '/users/gpu/haribala/code/image/experiments/exp7/image-and-text-generation/outputs/arch6/infogan_MNIST_2020_01_21_05_53_04/weights_10_100.pth'
    # weights_path = '/users/gpu/haribala/code/image/experiments/exp7/image-and-text-generation/outputs/arch6/infogan_MNIST_2020_01_21_17_24_58/weights_6_100.pth'
    # weights_path = '/users/gpu/haribala/code/image/experiments/exp7/image-and-text-generation/outputs/arch6/infogan_MNIST_2020_01_21_17_57_42/weights_10_900.pth'
    # weights_path = '/users/gpu/haribala/code/image/experiments/exp7/image-and-text-generation/outputs/arch6/infogan_MNIST_2020_01_21_19_34_38/weights_29_500.pth'
    # weights_path = '/users/gpu/haribala/code/image/experiments/exp7/image-and-text-generation/outputs/arch6/infogan_MNIST_2020_01_21_19_34_38/weights_1_800.pth'
    # weights_path = '/users/gpu/haribala/code/image/experiments/exp7/image-and-text-generation/outputs/arch6/infogan_MNIST_2020_01_21_20_25_08/weights_12_400.pth'
    # weights_path = '/users/gpu/haribala/code/image/experiments/exp7/image-and-text-generation/outputs/arch6/infogan_MNIST_2020_01_21_20_36_14/weights_33_300.pth'
    # weights_path = '/users/gpu/haribala/code/image/experiments/exp7/image-and-text-generation/outputs/arch6/infogan_MNIST_2020_01_22_02_54_21/weights_2_400.pth'
    weights_path = '/users/gpu/haribala/code/image/experiments/exp7/image-and-text-generation/outputs/arch6/infogan_MNIST_2020_01_22_03_06_25/weights_4_100.pth'

    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    model = InfoGAN()
    model.load_state_dict(torch.load(weights_path))
    model.eval().to(device)

    # noise = torch.randn(1, model.gen.d_noise, device=device).expand(d_batch, model.gen.d_noise)
    # labels = class_id * torch.ones(d_batch, device=device).long()
    # rest_code = torch.zeros(d_batch, 2, device=device) + 0.2
    # rest_code[:, 0] = torch.linspace(-0.2, 0.2, d_batch, device=device) # only change the 2nd continuous random variable. Seems to have learned rotation.

    noise, labels, rest_code = model.gen.sample_latent(num_samples=d_batch, device=device)
    images = model.sample_images(noise, labels, rest_code)
    valid_logits, label_logits, _ = model.dis(images)

    is_real = sigmoid(valid_logits) > 0.5
    _, pred_labels = label_logits.max(dim=1)
    results = torch.cat((labels.unsqueeze(1), is_real.long(), pred_labels.unsqueeze(1)), dim=1)
    print(results)

    output_dir = './outputs/arch6/infogan_samples/'
    mkdir_p(output_dir)
    filename = f'{output_dir}samples_{get_timestamp()}.png'
    plot_and_save_info(images.cpu(), results, filename)

if __name__ == '__main__':
    sample()
