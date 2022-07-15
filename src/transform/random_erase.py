import math
import torch
import random

class RandomEraser(object):
    
    def __init__(self, probability=0.5, min_proportion=0.02, max_proportion=0.4, min_aspect_ration=0.3, mean=(0.4914, 0.4822, 0.4465)):
        """RandomEraser selectes a ractangle region in an image and erase its pixels.

        Args:
            probability (float, optional): prob that the erasing operation will be performed. Defaults to 0.5.
            min_proportion (float, optional): min proportion of erased area against input image. Defaults to 0.02.
            max_proportion (float, optional): max proportion of erased area against input image. Defaults to 0.4.
            min_aspect_ration (float, optional): min aspect ratio of erased area. Defaults to 0.3.
            mean (tuple, optional): erasing value. Defaults to (0.4914, 0.4822, 0.4465).
        """
        self.probability = probability
        self.mean = mean
        self.min_proportion = min_proportion
        self.max_proportion = max_proportion
        self.min_aspect_ration = min_aspect_ration

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """Performs random erasing

        Args:
            img (torch.Tensor): image

        Returns:
            torch.Tensor: either erased or not image
        """

        if random.uniform(0, 1) >= self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.min_proportion, self.max_proportion) * area
            aspect_ratio = random.uniform(self.min_aspect_ration, 1 / self.min_aspect_ration)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img