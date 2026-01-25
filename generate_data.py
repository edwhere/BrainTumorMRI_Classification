"""Convert the downloaded dataset into a collection of folders with each folder
containing class images.

The downloaded dataset is a collection of four folders each containing 756 data samples.
Each data sample is stored as a matlab file that can be opened using the h5py library.

The data in a h5py file is organized in groups. In the case of the MRI brain tumor dataset,
there is a single group called 'cjdata'.

A h5py group stores multiple datasets identified by keys. In the case of the MRI brain tumor
dataset, the keys are 'PID', 'image', 'label', 'tumorBorder', 'tumorMask'.

A h5py dataset is like a numpy array. It supports slicing operations like a numpy array. Some of the
dataset attributes also resemble those from a numpy array: shape, size, dtype.

The 'PID' dataset stores the 'patient ID'. It is an array of size Nx1 where N is variable.

The 'image' dataset stores the actual image. It is an array of size 512x512.

The 'label' dataset stores the class label. It is an array of size 1x1. The values are 1 for meningioma,
2 for glioma, and 3 for pituitary tumor.

The 'tumorBorder' dataset stores contour coordinates for the tumor. It is an array of size 1 x 2K, where
K is the number of coordinates. The array contains a sequence of numbers interpreted as [x1, y1, x2, y2, ...].

The 'tumorMask' stores a binary mask for the tumor region. It is an array of size 512x512.

"""

import os
import argparse
import h5py
import json
import numpy as np
import pandas as pd
from PIL import Image


CLASS_NAMES = {
    1: "meningioma",
    2: "glioma",
    3: "pituitary"
}

def parse_arguments():
    """Parse the command line arguments"""
    desc = "Convert the downloaded dataset into a collection of folders with each folder containing class images."
    parser = argparse.ArgumentParser(description=desc)
    required = parser.add_argument_group('required arguments')

    required.add_argument("-inp", "--input_dir", type=str, required=True,
                          help="Path of input directory containing the downloaded dataset (unzipped).")
    required.add_argument("-out", "--output_dir", type=str, required=True,
                          help="Path of output directory that will contain class folders.")

    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        raise NotADirectoryError("Input directory does not exist.")

    return args

class InputData:
    def __init__(self, input_dir):
        self.input_dir = input_dir
        if not os.path.exists(self.input_dir):
            raise NotADirectoryError("Input directory does not exist.")

    def get_subdirs(self):
        items = os.listdir(self.input_dir)
        folders = [item for item in items if os.path.isdir(os.path.join(self.input_dir, item))]
        subdirs = [os.path.join(self.input_dir, folder) for folder in folders]
        return subdirs

    def get_folders(self):
        items = os.listdir(self.input_dir)
        folders = [item for item in items if os.path.isdir(os.path.join(self.input_dir, item))]
        return folders

    def get_mat_file_names(self, folder: str):
        if folder not in self.get_folders():
            raise NotADirectoryError("Requested folder does not exist.")
        subdir = os.path.join(self.input_dir, folder)
        items = os.listdir(str(subdir))
        mat_files = [item for item in items if item.split(".")[-1] == "mat"]
        return mat_files

    def get_mat_file_paths(self, folder: str):
        if folder not in self.get_folders():
            raise NotADirectoryError("Requested folder does not exist.")
        file_names = self.get_mat_file_names(folder)
        subdir = os.path.join(self.input_dir, folder)
        file_paths = [os.path.join(subdir, file_name) for file_name in file_names]
        return file_paths


class OutputData:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        os.makedirs(os.path.join(self.output_dir, "Images"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "Masks"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "Borders"), exist_ok=True)

    def add_class_folders(self, class_labels: list[str]):
        for class_label in class_labels:
            subdir_img = os.path.join(self.output_dir, "Images", class_label)
            os.makedirs(subdir_img, exist_ok=True)
            subdir_msk = os.path.join(self.output_dir, "Masks", class_label)
            os.makedirs(subdir_msk, exist_ok=True)
            subdir_bor = os.path.join(self.output_dir, "Borders", class_label)
            os.makedirs(subdir_bor, exist_ok=True)

    def save_image(self, class_label: str, pixel_array: np.ndarray, image_file_name: str):
        subdir_img = os.path.join(self.output_dir, "Images", class_label)
        grayscale_array = pixel_array.astype(np.uint8) if pixel_array.dtype != np.uint8 else pixel_array
        img_grayscale = Image.fromarray(grayscale_array, mode='L')
        img_rgb = img_grayscale.convert('RGB')
        img_file_path = os.path.join(subdir_img, image_file_name)
        img_rgb.save(img_file_path)

    def save_border(self, class_label: str, border_dframe: pd.DataFrame, file_name: str):
        subdir_bor = os.path.join(self.output_dir, "Borders", class_label)
        os.makedirs(subdir_bor, exist_ok=True)
        border_dframe.to_csv(os.path.join(subdir_bor, file_name), index=False)

    def save_mask(self, class_label: str, mask_array: np.ndarray, file_name: str):
        subdir_mask = os.path.join(self.output_dir, "Masks", class_label)
        os.makedirs(subdir_mask, exist_ok=True)
        np.save(os.path.join(subdir_mask, file_name), mask_array)

    def save_metadata(self, metadata: dict):
        with open(os.path.join(self.output_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)


def main():
    """Run the main sequence of tasks."""
    args = parse_arguments()
    print("Arguments:")
    for key, value in vars(args).items():
        print(f"   {key}: {value}")

    source = InputData(args.input_dir)
    destination = OutputData(args.output_dir)
    destination.add_class_folders(class_labels=[val for _, val in CLASS_NAMES.items()])

    folders = source.get_folders()

    metadata = {
        "fid": [],
        "pid": [],
        "label": []
    }



    for folder in folders:
        print("... processing folder:", folder)
        mat_files = source.get_mat_file_paths(folder)
        for mat_file in mat_files:
            prefix_name = os.path.basename(mat_file).split(".")[0]
            metadata["fid"].append(prefix_name)

            with h5py.File(mat_file, mode="r") as f:
                # h5 files are organized in groups with each group having multiple datasets.
                # The MRI brain tumor dataset has a single group called 'cjdata'.
                # The group has keys where each key identifies a dataset.
                # The keys for the MRI brain tumor dataset are 'PID', 'image', 'label', 'tumorBorder', 'tumorMask'
                group_obj = f['cjdata']
                label_dataset = group_obj['label']
                label_value = int(label_dataset[0, 0])
                label_name = CLASS_NAMES[label_value]
                metadata["label"].append(label_name)

                pid_dataset = group_obj['PID']
                pid_value = pid_dataset[:].flatten().tolist()
                metadata["pid"].append(list(pid_value))

                image_dataset = group_obj['image']
                img_array = image_dataset[:]
                destination.save_image(class_label=label_name,
                                       pixel_array=img_array,
                                       image_file_name=prefix_name + ".png")

                tumor_border_dataset = group_obj['tumorBorder']
                tum_border = tumor_border_dataset[:].flatten()
                paired_border = list(zip(tum_border[::2].tolist(), tum_border[1::2].tolist()))
                border_data = {"x": [], "y": []}
                for elem in paired_border:
                    border_data["x"].append(elem[0])
                    border_data["y"].append(elem[1])
                border_data_df = pd.DataFrame(data=border_data)
                destination.save_border(class_label=label_name,
                                        border_dframe=border_data_df, file_name=prefix_name + ".csv")

                tumor_mask_dataset = group_obj['tumorMask']
                tum_mask = tumor_mask_dataset[:]
                destination.save_mask(class_label=label_name, mask_array=tum_mask, file_name=prefix_name + ".npy")

        destination.save_metadata(metadata=metadata)

    print("... finished the process to generate MRI brain tumor data.")


if __name__ == "__main__":
    main()
