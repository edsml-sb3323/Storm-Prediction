import torch
import numpy as np
import random
import csv
import pandas as pd


def set_seed(seed):
    """
    Set the random seed for reproducibility.

    Args:
        seed (int): The seed value to set.

    Returns:
        bool: True if the seed is set successfully.

    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    return True


def to_csv(csv_file, missing_times, missing_wing_speeds):
    """
    Writes the missing times and corresponding wing speeds to a CSV file.

    Args:
        csv_file (str): The path of the CSV file to write.
        missing_times (list): A list of missing times.
        missing_wing_speeds (list): A list of corresponding missing wing
        speeds.

    Returns:
        None
    """
    combining = list(zip(missing_times, missing_wing_speeds))
    with open(csv_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(combining)


def create_df(csv_file_name):
    """
    Reads a CSV file and performs data transformations on it.

    Parameters:
    - csv_file_name (str): The path to the CSV file.

    Returns:
    - df (pandas.DataFrame): The transformed DataFrame.

    This function reads the specified CSV file and performs the following
    transformations:
    1. Groups the data by "storm_id" and calculates the number of images per
    storm.
    2. Merges the calculated "images_per_storm" column with the original
    DataFrame.
    3. Constructs the "image_path" column by concatenating a fixed path with
    the "image_file_name" column.
    4. Calculates the percentage of each image within its corresponding storm.
    The resulting DataFrame is returned.
    """
    df = pd.read_csv(csv_file_name)
    images_per_storm = df.groupby("storm_id").size().to_frame(
        "images_per_storm")
    df = df.merge(images_per_storm, how="left", on="storm_id")
    df["image_path"] = (
        "/content/gdrive/MyDrive/edsmlproj2/all_storm_image/" + df[
            "image_file_name"]
    )
    df["pct_of_storm"] = \
        df.groupby("storm_id").cumcount() / df.images_per_storm
    return df


def create_test_df(csv_file_name):
    """
    Create a test DataFrame from a CSV file.

    Parameters:
    csv_file_name (str): The path to the CSV file.

    Returns:
    pandas.DataFrame: The created DataFrame.

    """
    df = pd.read_csv(csv_file_name)
    images_per_storm = df.groupby("storm_id").size().to_frame(
        "images_per_storm")
    df = df.merge(images_per_storm, how="left", on="storm_id")
    df["image_path"] = (
        "/content/gdrive/MyDrive/edsmlproj2/surprise_storm_image/"
        + df["image_file_name"]
    )
    df["pct_of_storm"] = \
        df.groupby("storm_id").cumcount() / df.images_per_storm
    return df
