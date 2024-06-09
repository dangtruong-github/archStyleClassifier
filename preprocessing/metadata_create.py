import pandas as pd
import os
from PIL import Image

from common_functions.functions import GetParentPath


def MetadataCreator(
    config
):
    parent_path = GetParentPath(config, __file__)
    data_path = os.path.join(parent_path, "data", "arcDataset")

    metadata_img = []

    for arch_type in os.listdir(data_path):
        print(f"Metadata of {arch_type}")
        arch_type_path = os.path.join(data_path, arch_type)
        for img_path in os.listdir(arch_type_path):
            im = Image.open(img_path)
            width, height = im.size
            metadata_img.append({
                "filename": img_path,
                "style": arch_type,
                "img_size":  "{}x{}".format(width, height)
            })

    metadata_df = pd.DataFrame(metadata_img)

    metadata_filename = config["processing"]["metadata_filename"]
    path_to_save = os.path.join(parent_path, metadata_filename)

    metadata_df.to_csv(path_to_save, index=False)


def MetadataReader(
    config      
) -> pd.DataFrame:
    parent_path = GetParentPath(config, __file__)
    metadata_filename = config["processing"]["metadata_filename"]
    metadata_path = os.path.join(parent_path, metadata_filename)

    

    metadata_df = pd.read_csv(metadata_path)

    return metadata_df
