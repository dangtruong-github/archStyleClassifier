from torch.utils.data import Dataset, DataLoader


class NaiveClassificationDataset(Dataset):
    def __init__(self, type_dataset):
        if type_dataset not in ["train", "val", "test"]:
            raise ValueError(f"Type dataset {type_dataset} does not exist")

    def __len__(self):
        pass

    def __getitem__(self):
        pass


def NaiveClassificationLoader(
    config,
    type_dataset: str
) -> DataLoader:
    if type_dataset not in ["train", "val", "test"]:
        raise ValueError(f"Type dataset {type_dataset} does not exist")

    testing_mode = bool(config["general"]["test"])
    batch_size = int(config["train"]["batch_size"])

    if testing_mode:
        batch_size = int(config["train"]["batch_size_test"])

    shuffle = type_dataset == "train"

    dataset = NaiveClassificationDataset()
    loader = DataLoader(dataset, batch_size, shuffle)

    return loader
