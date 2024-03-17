import unittest
import pandas as pd
from PIL import Image
import torch
from Tools.StormDataset import StormDataset

import sys

sys.path.append("..")


class TestStormDataset(unittest.TestCase):
    def setUp(self):
        """Mock data for testing"""

        # Import test images and their data
        csv_path = "./resources/test/test_storm_images_csv.csv"
        data = pd.read_csv(csv_path)
        # Convert the data types of numeric columns to integers
        data[["relative_time", "ocean", "wind_speed"]] = data[
            ["relative_time", "ocean", "wind_speed"]
        ].astype(int)

        # Create a dictionary mapping each image filename with its PIL image
        imgs = {}
        for file in data["image_file_name"]:
            image_path = f"./resources/test/{file}"
            imgs[file] = Image.open(image_path)

        self.data_df = data
        self.storm_id = "bkh"  # Choose one of the storm ids for testing
        self.num_sequence = 2
        all_images_dict = imgs

        # Create a StormDataset instance
        self.dataset = StormDataset(
            self.data_df, self.storm_id, self.num_sequence, all_images_dict
        )

    def test_len(self):
        """Test __len__ with a single index."""
        self.assertEqual(
            len(self.dataset), len(self.dataset.df) - self.num_sequence - 2
        )

    def test_getitem_single_index(self):
        """Test __getitem__ with a single index"""
        idx = 0
        X, y = self.dataset[idx]
        self.assertEqual(X.shape, torch.Size([self.num_sequence, 1, 366, 366]))
        self.assertEqual(y.shape, torch.Size([3, 1, 366, 366]))

    def test_getitem_slice(self):
        """Test __getitem__ with a slice"""
        start, stop, step = 0, 3, 2
        sub_dataset = self.dataset[start:stop:step]
        self.assertEqual(len(sub_dataset), (stop - start - 1) // step + 1)
        X, y = sub_dataset[0]
        self.assertEqual(X.shape, torch.Size([self.num_sequence, 1, 366, 366]))
        self.assertEqual(y.shape, torch.Size([3, 1, 366, 366]))

    def test_get_original_image(self):
        """Test get_original_image function"""
        img_filename = "bkh_000.jpg"
        img = self.dataset.get_original_image(img_filename)
        self.assertEqual(img.shape, torch.Size([1, 366, 366]))

    def test_resize_image(self):
        """Test resize_image function"""
        img = torch.rand(3, 366, 366)
        resized_img = self.dataset.resize_image(img, 256)
        self.assertEqual(resized_img.shape, torch.Size([3, 256, 256]))

    def test_get_sub_series_of_images(self):
        """Test get_sub_series_of_images function"""
        start, num_img = 0, 2
        sub_series = self.dataset.get_sub_series_of_images(start, num_img)
        self.assertEqual(sub_series.shape, torch.Size([num_img, 1, 366, 366]))


if __name__ == "__main__":
    unittest.main()
