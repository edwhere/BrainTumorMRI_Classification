"""A library of functions and classes to operate on the data source.
The source data consists of a root directory containing three subdirectories named glioma,
meningioma, and pituitary, which contain T1-weighted contrast-enhanced MRI images of
brain tumors. The distribution of images is:
    glioma: 1426 slices
    meningioma: 708 slices
    pituitary: 1426 slices
The root directory also contains a file called metadata,json, which correlates patients and
their images. The total number of patients is 233.

"""

import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold

import constants as con

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

        items = os.listdir(target_dir)
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
        items = os.listdir(target_dir)
        images = [item for item in items if item.split(".")[-1] in con.IMAGE_EXTENSIONS]
        ids = [item.split(".")[0] for item in images]
        return ids

    def get_kfold_partitions(self, test_percent: int, kvalue: int, images_per_tumor_type: int) -> list:
        """Get a test set together with K-fold cross-validation partitions.
        Args:
            test_percent (int): percentage of data assigned to the test set.
            kvalue (int): The value of K for k-fold cross-validation using all the data not included in the test set.
            images_per_tumor_type (int): The total number of images per tumor type.
        Returns:
            list: A list of K pandas DataFrames, one per fold. Each DataFrame has columns "id", "label",
                and "subset", where "subset" is 'trn' for train data, 'val' for validation data, and 'tst'
                for test data. The test data is the same across all folds.
        """

        if images_per_tumor_type > con.MAX_IMAGES_PER_TUMOR_TYPE:
            raise ValueError(f"Max. number of images per tumor type is {con.MAX_IMAGES_PER_TUMOR_TYPE}")

        # Get the dataset entries for partitions
        gli_ids = self.get_image_ids("glioma")
        men_ids = self.get_image_ids("meningioma")
        pit_ids = self.get_image_ids("pituitary")

        sel_gli_ids = random.sample(gli_ids, images_per_tumor_type)
        sel_men_ids = random.sample(men_ids, images_per_tumor_type)
        sel_pit_ids = random.sample(pit_ids, images_per_tumor_type)

        xdata = np.array(sel_gli_ids + sel_men_ids + sel_pit_ids)
        ydata = np.array([con.LABELS["gli"]] * images_per_tumor_type + [con.LABELS["men"]] * images_per_tumor_type +
                         [con.LABELS["pit"]] * images_per_tumor_type)

        xcv, xtst, ycv, ytst = train_test_split(xdata, ydata, test_size=test_percent/100.0,
                                                random_state=con.RANDOM_SEED, stratify=ydata)

        print(f"Size of the test set: {len(xtst)}")

        # Initialize KFold on the remaining CV data
        kf = KFold(n_splits=kvalue, shuffle=True, random_state=con.RANDOM_SEED)

        # The test subset is shared across all folds
        tst_df = pd.DataFrame({"id": xtst, "label": ytst, "subset": "tst"})

        # Loop to generate one DataFrame per fold
        fold_dataframes = []

        for fold, (train_index, val_index) in enumerate(kf.split(xcv)):
            print(f"--- Fold {fold + 1} ---")

            # Slice CV data into Train and Validation chunks
            xtrn, xval = xcv[train_index], xcv[val_index]
            ytrn, yval = ycv[train_index], ycv[val_index]

            print(f"Train size: {xtrn.shape[0]} and Valid. size: {xval.shape[0]}")

            trn_df = pd.DataFrame({"id": xtrn, "label": ytrn, "subset": "trn"})
            val_df = pd.DataFrame({"id": xval, "label": yval, "subset": "val"})

            fold_df = pd.concat([trn_df, val_df, tst_df], ignore_index=True)
            fold_dataframes.append(fold_df)

        return fold_dataframes

if __name__ == "__main__":
    # Run a simple test to check if the partition function works
    rdir = "/home/xuser/Datasets/DataBrainTumorMRI2/"
    src_data = MRIDataSource(rdir)

    folds = src_data.get_kfold_partitions(test_percent=10, kvalue=4, images_per_tumor_type=20)
    print("Number of folds:", len(folds))

    for i, fold_df in enumerate(folds):
        print(f"--- Fold {i + 1} ---")
        print("Total rows:", len(fold_df))
        print(fold_df["subset"].value_counts())
        print(fold_df)

