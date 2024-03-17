import io
from itertools import product
import numpy as np
from PIL import Image
import random
import torch
import zipfile


def set_seed(seed):
    """
    Set ALL the random seeds to a fixed value and
    take out any randomness from cuda kernels.

    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    return True


def set_device(device="cpu", idx=0):
    if device != "cpu":
        if torch.cuda.device_count() > idx and torch.cuda.is_available():
            print(
                "Cuda installed! Running on GPU {} {}!".format(
                    idx, torch.cuda.get_device_name(idx)
                )
            )
            device = "cuda:{}".format(idx)
        elif torch.cuda.device_count() > 0 and torch.cuda.is_available():
            print(
                "Cuda installed but only {} GPU(s) available!".format(
                    torch.cuda.device_count()
                )
            )
            print("Running on GPU 0 {}!".format(torch.cuda.get_device_name()))
            device = "cuda:0"
        else:
            device = "cpu"
            print("No GPU available! Running on CPU")
    return device


def get_images_from_zip(zip_path):
    """
    Open the zipped folder of all images to read them,
    store them in a dictionary (name: PIL Image) to be returned.

    Parameters
    ----------
    zip_path: str

        The path to the zipped folder to be opened.

    """
    imgs = {}
    with zipfile.ZipFile(zip_path, "r") as zip_file:
        imgs = {
            name[name.find("/") + 1:]: Image.open(
                io.BytesIO(zip_file.read(name)))
            for name in zip_file.namelist()
        }
        print(len(imgs))
    return imgs


def generate_combinations(dictionary):
    """
    Create combinations of values attached to each key in a given dictionary.
    Thanks to ChatGPT:
    https://chat.openai.com/share/0ce3de92-2309-43fe-be2a-84312c6a802a

    """
    keys = list(dictionary.keys())
    values = list(dictionary.values())

    # Generate all possible combinations of values
    combinations = list(product(*values))

    # Create a list of dictionaries using the combinations
    result = [dict(zip(keys, combination)) for combination in combinations]

    return result
