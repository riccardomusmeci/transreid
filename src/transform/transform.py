
from torchvision import transforms as T
from src.transform.random_erase import RandomEraser

def get_transforms(mode: str, img_size: list = [224, 224], horizontal_flip_p: float = 0.3, rand_erase_p: float = 0.5, pixel_mean: list = [0.485, 0.456, 0.406], pixel_std: list = [0.229, 0.224, 0.225], padding: int = 10) -> T.Compose:
    """retunrs the set of transformations to apply to images

    Args:
        mode (str): one in ["train", "test", "val"]
        img_size (list, optional): rescaling image size. Defaults to [224, 224].
        horizontal_flip_p (float, optional): random horizontal flip probability. Defaults to 0.3.
        rand_erase_p (float, optional): randome erasing probability. Defaults to 0.5.
        pixel_mean (list, optional): mean image normalization. Defaults to [0.485, 0.456, 0.406].
        pixel_std (list, optional): standard deviation image normalization. Defaults to [0.229, 0.224, 0.225].
        padding (int, optional): padding pixels. Defaults to 10.

    Returns:
        T.Compose: Composition of PyTorch transformations
    """
    if mode not in ["train", "test", "val"]:
        raise ValueError(f"mode must be one of ['train', 'test', val'].")
    
    if mode == "train":
        return T.Compose([
            T.Resize(img_size, interpolation=3),
            T.RandomHorizontalFlip(p=horizontal_flip_p),
            T.Pad(padding),
            T.RandomCrop(img_size),
            T.ToTensor(),
            T.Normalize(mean=pixel_mean, std=pixel_std),
            RandomEraser(probability=rand_erase_p),
        ])
    if mode == "val" or mode == "test":
        return T.Compose([
            T.Resize(img_size, interpolation=3),
            T.ToTensor(),
            T.Normalize(mean=pixel_mean, std=pixel_std),
        ])
