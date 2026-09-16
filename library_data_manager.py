"""A library of functions and classes to operate on the data source.
The source data consists of a root directory containing three subdirectories named glioma,
meningioma, and pituitary, which contain T1-weighted contrast-enhanced MRI images of
brain tumors. The distribution of images is:
    glioma: 1426 slices
    meningioma: 708 slices
    pituitary: 930 slices
The root directory also contains a file called metadata.json, which correlates patients and
their images. The total number of patients is 233.

Because several slices come from the same patient, all data partitions are grouped by patient
id: a patient contributes slices to exactly one subset. Splitting by slice instead of by patient
leaks information across subsets and inflates the validation and test scores.

"""

import os
import json
import torch
import random
import numpy as np
import pandas as pd
from typing import Union

from torchvision import transforms
import torchvision.transforms.functional as tfun
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from PIL import Image

import constants as con

pd.set_option('display.max_colwidth', None)

def check_dir_items(root_dir) -> None:
    if not os.path.isdir(root_dir):
        raise NotADirectoryError("Unavailable data source at {}".format(root_dir))

    gli_dir = os.path.join(root_dir, con.IMAGES_DIR_NAME, con.GLI_DIR_NAME)
    men_dir = os.path.join(root_dir, con.IMAGES_DIR_NAME, con.MEN_DIR_NAME)
    pit_dir = os.path.join(root_dir, con.IMAGES_DIR_NAME, con.PIT_DIR_NAME)
    meta_file = os.path.join(root_dir, con.META_FILE_NAME)

    if not os.path.isdir(gli_dir):
        raise NotADirectoryError("Unavailable glioma data at {}".format(root_dir))
    if not os.path.isdir(men_dir):
        raise NotADirectoryError("Unavailable meningioma data at {}".format(root_dir))
    if not os.path.isdir(pit_dir):
        raise NotADirectoryError("Unavailable pituitary data at {}".format(root_dir))

    if not os.path.isfile(meta_file):
        raise FileNotFoundError("Unavailable metadata file at {}".format(meta_file))

def get_id_from_path(path: str) -> str:
    """Get the image id when the image path is known. If the path is /home/xuser/files/data123.png,
    then the id is 'data123'."""
    return os.path.splitext(os.path.basename(path))[0]

def get_ids_from_paths(paths: list[str]) -> list[str]:
    """Get a list of image id values from a list of path values. """
    return [get_id_from_path(path) for path in paths]

def decode_patient_id(pid: Union[list, str]) -> str:
    """Decode a patient id as stored in the metadata file. Patient ids are recorded as lists of
    character codes (e.g. [49, 49, 49, 51, 54, 54] is patient '111366')."""
    if isinstance(pid, (list, tuple)):
        return "".join(chr(int(code)) for code in pid)
    return str(pid)

def set_random_seeds(seed: int = con.RANDOM_SEED) -> None:
    """Seed every random number generator used during a run so that data selection, partitions,
    augmentations, and weight initialization are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class MRIDataSource:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.gli_dir = os.path.join(self.root_dir, con.IMAGES_DIR_NAME, con.GLI_DIR_NAME)
        self.men_dir = os.path.join(self.root_dir, con.IMAGES_DIR_NAME, con.MEN_DIR_NAME)
        self.pit_dir = os.path.join(self.root_dir, con.IMAGES_DIR_NAME, con.PIT_DIR_NAME)

        self.meta_file = os.path.join(self.root_dir, con.META_FILE_NAME)

    def get_image_paths(self, tumor_type):
        match tumor_type:
            case "glioma":
                target_dir = self.gli_dir
            case "meningioma":
                target_dir = self.men_dir
            case "pituitary":
                target_dir = self.pit_dir
            case _:
                raise ValueError("Unknown tumor type {}".format(tumor_type))

        items = sorted(os.listdir(target_dir))
        images = [item for item in items if item.split(".")[-1] in con.IMAGE_EXTENSIONS]
        paths = [os.path.join(target_dir, item) for item in images]
        return paths

    def get_image_ids(self, tumor_type):
        match tumor_type:
            case "glioma":
                target_dir = self.gli_dir
            case "meningioma":
                target_dir = self.men_dir
            case "pituitary":
                target_dir = self.pit_dir
            case _:
                raise ValueError("Unknown tumor type {}".format(tumor_type))
        items = sorted(os.listdir(target_dir))
        images = [item for item in items if item.split(".")[-1] in con.IMAGE_EXTENSIONS]
        ids = [item.split(".")[0] for item in images]
        return ids

    def get_patient_map(self) -> dict:
        """Read the metadata file and return a dictionary that maps each image id to the id of the
        patient that the image belongs to."""
        with open(self.meta_file, "r") as file_handle:
            metadata = json.load(file_handle)

        return {str(fid): decode_patient_id(pid) for fid, pid in zip(metadata["fid"], metadata["pid"])}

    def get_kfold_partitions(self, test_percent: int, kvalue: int, images_per_tumor_type: int,
                             seed: int = con.RANDOM_SEED) -> list:
        """Get a test set together with K-fold cross-validation partitions. Every partition is
        stratified by tumor type and grouped by patient, so all the slices of a given patient belong
        to a single subset.
        Args:
            test_percent (int): percentage of data assigned to the test set. Because whole patients
                are held out, the achievable fraction is the closest value of the form 1/n.
            kvalue (int): The value of K for k-fold cross-validation using all the data not included in the test set.
            images_per_tumor_type (int): The total number of images per tumor type that will be used to create folds.
            seed (int): Seed used to select images and to build the partitions.
        Returns:
            list: A list of K pandas DataFrames, one per fold. Each DataFrame has columns "path", "label",
                "pid", and "subset", where "subset" is 'trn' for train data, 'val' for validation data, and
                'tst' for test data. The test data is the same across all folds.
        """

        if images_per_tumor_type > con.MAX_IMAGES_PER_TUMOR_TYPE:
            raise ValueError(f"Max. number of images per tumor type is {con.MAX_IMAGES_PER_TUMOR_TYPE}")

        if not 0 < test_percent < 100:
            raise ValueError("The test percentage must be larger than 0 and less than 100.")

        if kvalue < 2:
            raise ValueError("The K value for cross-validation must be at least 2.")

        patient_map = self.get_patient_map()

        # Select the images for each tumor type using a dedicated generator, so that the selection
        # does not depend on the state of the global random number generator.
        rng = random.Random(seed)

        paths = []
        labels = []
        for tumor_type, label_key in (("glioma", "gli"), ("meningioma", "men"), ("pituitary", "pit")):
            type_paths = self.get_image_paths(tumor_type)
            if len(type_paths) < images_per_tumor_type:
                raise ValueError(f"Requested {images_per_tumor_type} images of type {tumor_type}, "
                                 f"but only {len(type_paths)} are available.")
            paths.extend(rng.sample(type_paths, images_per_tumor_type))
            labels.extend([con.LABELS[label_key]] * images_per_tumor_type)

        unknown_ids = [get_id_from_path(path) for path in paths if get_id_from_path(path) not in patient_map]
        if unknown_ids:
            raise KeyError(f"No patient id in {self.meta_file} for image ids such as {unknown_ids[:5]}")

        xdata = np.array(paths)
        ydata = np.array(labels)
        gdata = np.array([patient_map[get_id_from_path(path)] for path in paths])

        # Hold out the test set first. The number of splits determines the size of the test set:
        # a requested 15% becomes 1/7 of the data (14.3%).
        test_splitter = StratifiedGroupKFold(n_splits=max(2, round(100 / test_percent)), shuffle=True,
                                             random_state=seed)
        cv_index, tst_index = next(test_splitter.split(xdata, ydata, groups=gdata))

        xcv, ycv, gcv = xdata[cv_index], ydata[cv_index], gdata[cv_index]
        xtst, ytst, gtst = xdata[tst_index], ydata[tst_index], gdata[tst_index]

        print(f"Size of the test set: {len(xtst)} slices from {len(set(gtst))} patients")

        # Initialize the cross-validation splitter on the remaining data
        kf = StratifiedGroupKFold(n_splits=kvalue, shuffle=True, random_state=seed)

        # The test subset is shared across all folds
        tst_df = pd.DataFrame({"path": xtst, "label": ytst, "pid": gtst, "subset": "tst"})

        # Loop to generate one DataFrame per fold
        fold_dataframes = []

        for fold, (train_index, val_index) in enumerate(kf.split(xcv, ycv, groups=gcv)):
            print(f"--- Fold {fold + 1} ---")

            # Slice CV data into Train and Validation chunks
            xtrn, xval = xcv[train_index], xcv[val_index]
            ytrn, yval = ycv[train_index], ycv[val_index]
            gtrn, gval = gcv[train_index], gcv[val_index]

            # Guard against a patient appearing in more than one subset of this fold
            for name_a, group_a, name_b, group_b in (("train", gtrn, "validation", gval),
                                                     ("train", gtrn, "test", gtst),
                                                     ("validation", gval, "test", gtst)):
                shared = set(group_a) & set(group_b)
                if shared:
                    raise RuntimeError(f"Patients {sorted(shared)[:5]} appear in both the {name_a} "
                                       f"and {name_b} subsets of fold {fold + 1}")

            print(f"Train size: {xtrn.shape[0]} slices from {len(set(gtrn))} patients and "
                  f"Valid. size: {xval.shape[0]} slices from {len(set(gval))} patients")

            trn_df = pd.DataFrame({"path": xtrn, "label": ytrn, "pid": gtrn, "subset": "trn"})
            val_df = pd.DataFrame({"path": xval, "label": yval, "pid": gval, "subset": "val"})

            fold_df = pd.concat([trn_df, val_df, tst_df], ignore_index=True)
            fold_dataframes.append(fold_df)

        return fold_dataframes

class PartitionDataset(Dataset):
    def __init__(self, fold_data_df: pd.DataFrame, subset: str, transform=None):
        self.fold_data_df = fold_data_df
        self.subset = subset
        self.transform = transform

        if subset not in ["trn", "val", "tst"]:
            raise ValueError(f"Invalid subset: {subset}")

        self.data_df = fold_data_df[fold_data_df["subset"] == subset].copy()

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index: int):
        row_as_df = self.data_df.iloc[[index]]
        img_path = row_as_df["path"].values[0]
        img_id = get_id_from_path(img_path)
        label = int(row_as_df["label"].values[0])

        # Convert to RGB: the pretrained SWIN models expect 3 channels, and grayscale slices would
        # otherwise break the 3-channel normalization step of the transforms.
        img_data = Image.open(img_path).convert("RGB")
        if self.transform:
            img_data = self.transform(img_data)

        return {
            "image": img_data,
            "label": label,
            "id": img_id,
            "filepath": img_path,
        }


class RandomSelectRotation:
    def __init__(self):
        self.angles = [90, 180, 270]
    def __call__(self, target_image: Union[Image.Image, torch.Tensor]) -> Union[Image.Image, torch.Tensor]:
        angle = random.choice(self.angles)
        return tfun.rotate(target_image, angle)


# Training transforms
def get_train_transform():
    train_transform = transforms.Compose([
        transforms.Resize((con.IMAGE_SIZE_FOR_NN, con.IMAGE_SIZE_FOR_NN)),
        transforms.RandomResizedCrop((con.CROP_SIZE_FOR_NN, con.CROP_SIZE_FOR_NN), scale=con.CROP_SCALE),
        RandomSelectRotation(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return train_transform

# Validation transforms
def get_valid_transform():
    valid_transform = transforms.Compose([
        transforms.Resize((con.IMAGE_SIZE_FOR_NN, con.IMAGE_SIZE_FOR_NN)),
        transforms.CenterCrop((con.CROP_SIZE_FOR_NN, con.CROP_SIZE_FOR_NN)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return valid_transform


def get_data_loaders(dataset_train, dataset_valid, batch_size):
    """ Return the loader functions for TRN and VAL datasets. The function uses the dataset handlers
    obtained as instances of the PartitionDataset class.
    """
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=con.NUM_WORKERS)
    valid_loader = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False, num_workers=con.NUM_WORKERS)
    return train_loader, valid_loader


class ResultsStore:
    """Create and expose the directory structure that holds the results of a run."""

    def __init__(self, root_dir: str):
        self.root_dir = root_dir

        self._models_dir = os.path.join(self.root_dir, "Models")
        self._logs_dir = os.path.join(self.root_dir, "Logs")

        os.makedirs(self._models_dir, exist_ok=True)
        os.makedirs(self._logs_dir, exist_ok=True)

    @property
    def models_dir(self):
        return self._models_dir

    @property
    def logs_dir(self):
        return self._logs_dir




if __name__ == "__main__":
    # Run a simple test to check if the partition function works
    rdir = "/home/xuser/Datasets/DataBrainTumorMRI2/"
    src_data = MRIDataSource(rdir)

    folds = src_data.get_kfold_partitions(test_percent=10, kvalue=4, images_per_tumor_type=200)
    print("Number of folds:", len(folds))

    for i, fold_df in enumerate(folds):
        print(f"--- Fold {i + 1} ---")
        print("Total rows:", len(fold_df))
        print(fold_df["subset"].value_counts())
        print(pd.crosstab(fold_df["subset"], fold_df["label"]))
        print("Patients per subset:", fold_df.groupby("subset")["pid"].nunique().to_dict())

