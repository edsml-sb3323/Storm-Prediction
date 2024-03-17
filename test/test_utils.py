import unittest
from unittest.mock import patch
from Tools.Utils import set_seed, set_device, get_images_from_zip, \
    generate_combinations

import sys

sys.path.append("..")


class TestUtilityFunctions(unittest.TestCase):
    def test_set_seed(self):
        """
        Test the set_seed function.
        This function sets all the random seeds to a fixed value.

        """
        seed = 42
        result = set_seed(seed)
        self.assertTrue(result)

    @patch("builtins.print")
    def test_set_device_cpu(self, mock_print):
        """
        Test the set_device function for CPU to check
        if the set_device function correctly returns 'cpu'.

        """
        device = set_device("cpu")
        mock_print.assert_not_called()  # assert that print() was not called
        self.assertEqual(device, "cpu")

    def test_get_images_from_zip(self):
        """
        Test the get_images_from_zip function.
        This function checks if the get_images_from_zip function
        correctly reads images from a zip file.

        """
        zip_path = "./resources/test/test_zipped_images.zip"
        imgs = get_images_from_zip(zip_path)
        self.assertIsInstance(imgs, dict)
        self.assertTrue(len(imgs) > 0)

    def test_generate_combinations(self):
        """
        Test the generate_combinations function.
        This function checks if the generate_combinations function
        correctly creates combinations of values.

        """
        input_dict = {"param1": [1, 2], "param2": ["a", "b"]}
        combinations = generate_combinations(input_dict)
        expected_result = [
            {"param1": 1, "param2": "a"},
            {"param1": 1, "param2": "b"},
            {"param1": 2, "param2": "a"},
            {"param1": 2, "param2": "b"},
        ]
        self.assertEqual(combinations, expected_result)


if __name__ == "__main__":
    unittest.main()
