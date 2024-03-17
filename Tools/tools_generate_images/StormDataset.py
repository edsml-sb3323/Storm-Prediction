import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Grayscale, PILToTensor, Resize


class StormDataset(Dataset):
    def __init__(self, data_df, storm_id, num_sequence, all_images_dict,
                 img_size=366):
        """
        Initialise a StormDataset object.
        They contain a series of (X, y) where
        X, used as feature, is a series of N consecutive images and
        y, used as label, is 3 images next to the image series

        Parameters
        ----------
        data_df: pandas.DataFrame

            The DataFrame containing the data of each image
            (must have "storm_id" and "image_file_name" columns).

        storm_id: str

            The ID of a storm whose images are chosen for training the model.

        num_sequence: int

            The number of consecutive images used as training features (X).

        all_images_dict: dict

            The dictionary with the filename and PIL of each image
            as keys and values respectively.

        img_size: int

            The size of images (originally 366).

        """
        self.df = data_df[data_df["storm_id"] == storm_id]
        self.storm_id = storm_id
        self.image_file_names = self.df["image_file_name"].to_numpy()
        self.all_images_dict = all_images_dict
        self.num_sequence = num_sequence
        self.imgs_original = [self.get_original_image(f) for
                              f in self.image_file_names]
        self.imgs_resized = [
            self.resize_image(img, img_size)
            for img in self.imgs_original
            if img is not None
        ]
        # Normalise images
        self.mean = torch.mean(torch.stack(self.imgs_resized) / 255.0).item()
        self.std = torch.std(torch.stack(self.imgs_resized) / 255.0).item()
        self.imgs = [
            ((img / 255.0) - self.mean) / self.std for img in self.imgs_resized
        ]

    def get_original_image(self, img_filename):
        """
        Get the original (non-normalised) image from an image file.

        Parameter
        ---------
        img_filename: str

            The name of an image file.

        Return
        ------
        torch.Tensor

            The Tensor of an original image.

        """
        if img_filename in self.all_images_dict.keys():
            transform = Compose([Grayscale(num_output_channels=1),
                                 PILToTensor()])
            img = transform(self.all_images_dict[img_filename]).float()
            return img
        return None

    def resize_image(self, img, size):
        """Resize images to any size x size."""
        c, h, w = img.shape
        if h == w and size == h:
            return img
        # else
        transform = Resize((size, size), antialias=True)
        resized_img = transform(img)
        return resized_img

    def get_sub_series_of_images(self, start, num_img):
        """
        Get a sub-series from the series of all normalised images.

        Parameters
        ----------
        start: int

            The start index in the normalised series.

        num_img: int

            The number of images in the desired sub-series.

        Return
        ------
        torch.Tensor

            The Tensor of the desired sub-series.

        """
        imgs = torch.stack(self.imgs[start: (start + num_img)], dim=0)
        return imgs

    def __len__(self):
        """Enable len() function."""
        return len(self.df) - self.num_sequence - 2

    def __getitem__(self, idx):
        """Enable indexing and slicing."""
        if isinstance(idx, slice):
            start = idx.start or 0
            stop = idx.stop or len(self.df) - self.num_sequence - 2
            step = idx.step or 1
            return [
                (
                    self.get_sub_series_of_images(i, self.num_sequence),
                    self.get_sub_series_of_images(i + self.num_sequence, 3),
                )
                for i in range(start, stop, step)
            ]
        else:
            return (
                self.get_sub_series_of_images(idx, self.num_sequence),
                self.get_sub_series_of_images(idx + self.num_sequence, 3),
            )
