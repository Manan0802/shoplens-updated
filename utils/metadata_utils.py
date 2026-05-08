import os
import pandas as pd


def load_metadata(csv_path):
    df = pd.read_csv(csv_path, on_bad_lines='skip')
    return df


def get_product_name(df, image_path):
    """
    Extract product name using image id
    """

    # get id from filename
    filename = os.path.basename(image_path)
    image_id = int(filename.split(".")[0])

    row = df[df["id"] == image_id]

    if len(row) > 0:
        return row.iloc[0]["productDisplayName"]
    
    return "Unknown Product"


def get_all_categories(df):
    """
    Extract unique categories from dataset
    """
    return df["articleType"].dropna().unique().tolist()