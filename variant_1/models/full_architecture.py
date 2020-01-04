"""
The full architecture.
"""

from dcgan import DCGAN as ImageGenerator
from image_to_text import ImageToText

class FullArchitecture:
    """
    This is the full architecture.
    You can sample image, text pairs from this.
    """
    def __init__(self):
        self.image_gen = ImageGenerator()
        self.cond_text_gen = ImageToText()

    def train_for_one_epoch(self, dataset):
        """
        Train the network for one epoch on the dataset.
        """

    def evaluate(self, dataset):
        """
        Evaluate the network on the dataset.
        """

    def sample(self, num_samples):
        """
        Sample from the network.
        """

        images = self.image_gen.sample(num_samples=num_samples)
        texts = self.cond_text_gen.sample(images=images)
        return images, texts
