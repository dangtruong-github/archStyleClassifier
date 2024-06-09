import os


def GetParentPath(config, file):
    parent_folder_name = config["general"]["parent_folder_name"]
    parent_directory = os.path.abspath(file)

    while os.path.basename(parent_directory) != parent_folder_name:
        parent_directory = os.path.dirname(parent_directory)

    return parent_directory
