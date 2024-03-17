from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image as pil_image


class DatasetWIND(Dataset):
    """
    A custom dataset class for wind speed data.

    Args:
        df (pandas.DataFrame): The input dataframe containing the wind speed
        data.
        transform (torchvision.transforms.Compose): The data transformation to
        be applied to the images.

    Attributes:
        data (pandas.DataFrame): The input dataframe without the wind speed
        column.
        label (pandas.Series): The wind speed column from the input dataframe.
        transform (torchvision.transforms.Compose): The data transformation to
        be applied to the images.

    """

    def __init__(self, df, transform=None):
        """
        Initializes a new instance of the DatasetWIND class.

        Args:
            df (pandas.DataFrame): The input dataframe containing the wind
            speed data.
            transform (torchvision.transforms.Compose): The data
            transformation to be applied to the images.

        """
        train_df = df[df.pct_of_storm < 0.9].drop(
            ["images_per_storm", "pct_of_storm"], axis=1
        )
        val_df = df[df.pct_of_storm >= 0.9].drop(
            ["images_per_storm", "pct_of_storm"], axis=1
        )

        # Use train split if in training mode
        if transform == self.transform_train:
            self.data = train_df.drop(["wind_speed"], axis=1)
            self.label = train_df["wind_speed"]
            self.transform = transform

        # Use val split if in validation mode
        elif transform == self.transform_val:
            self.data = val_df.drop(["wind_speed"], axis=1)
            self.label = val_df["wind_speed"]
            self.transform = transform

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: The length of the dataset.

        """
        return len(self.data)

    def __getitem__(self, index):
        """
        Retrieves a sample from the dataset at the given index.

        Args:
            index (int): The index of the sample to retrieve.

        Returns:
            dict: A dictionary containing the image ID, image tensor, and
            label (if available) of the sample.

        """
        image_path = self.data.iloc[index]["image_path"]
        image = pil_image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        image_id = self.data.iloc[index]["image_file_name"]

        if self.label is not None:
            label = self.label.iloc[index]
            sample = {"image_id": image_id, "image": image, "label": label}
        else:
            sample = {"image_id": image_id, "image": image}

        return sample

    # Defining the transforms
    # Data transformations for training set
    transform_train = transforms.Compose(
        [
            transforms.RandomRotation((0, 360), expand=True),
            transforms.CenterCrop((366, 366)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225)),
        ]
    )

    # Data transformations for validation set
    transform_val = transforms.Compose(
        [
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225)),
        ]
    )
