"""Train a SWIN transformer model to categorize MRI brain tumor images in 3 classes: glioma,
meningioma, and pituitary tumors.

"""

import os
import json
import argparse
import socket

import torch
import torch.nn
import torch.optim
import pandas as pd
from tqdm.auto import tqdm
from typing import Tuple
from datetime import datetime
from time import time

import constants as con
import library_model as mod
import library_data_manager as dm
import library_saveops as sav

def parse_arguments():
    """Parse command-line arguments and return the arguments object."""
    parser = argparse.ArgumentParser(description="Train a SWIN transformer for image classification..")
    required = parser.add_argument_group("requirements")

    required.add_argument("--data_dir_path", type=str, required=True,
                          help="Path to data directory storing the images. This directory should have "
                               "an Images folder, which includes subdirectories called 'glioma', 'meningioma' and "
                               "'pituitary'. These class folders contain the actual images.")

    required.add_argument("--results_dir_path", type=str, required=True,
                          help="Path to a directory that will store the results.")

    parser.add_argument("--model_size", type=str, choices=con.MODEL_SIZES, default=con.MODEL_SIZES[0],
                        help=f"Model size (string). Select from {con.MODEL_SIZES}. Default: {con.MODEL_SIZES[0]}")

    parser.add_argument("--model_version", type=str, choices=con.MODEL_VERSIONS,
                        default=con.MODEL_VERSIONS[0],
                        help=f"Model version (string). Select from {con.MODEL_VERSIONS}. "
                             f"Default: {con.MODEL_VERSIONS[0]}")

    parser.add_argument("--epochs", type=int, default=con.EPOCHS,
                        help=f"Number of epochs (integer). Default: {con.EPOCHS}")

    parser.add_argument("--lrate", type=float, default=con.LRATE,
                        help=f"Learning rate (real number between 0 and 1. Default: {con.LRATE}")

    parser.add_argument("--batch", type=int, default=con.BATCH,
                        help=f"Batch size (integer). Default: {con.BATCH}")

    parser.add_argument("--keyword", type=str, default=con.KEYWORD,
                        help=f"A keyword included in the file name of the saved model. Default: {con.KEYWORD}")

    parser.add_argument("--gpu", action="store_true", help="Use GPU if available.")

    args = parser.parse_args()

    # Check the structure of the data directory and raise an error if any folders are missing.
    dm.check_dir_items(args.data_dir_path)

    # Check the range of the numeric input arguments.
    if args.lrate <= 0 or args.lrate >= 1:
        raise ValueError("Learning rate must be larger than 0 and less than 1.")

    if args.batch <=0:
        raise ValueError("Batch size must be larger than 0.")

    if args.epochs <=0:
        raise ValueError("Number of epochs must be larger than 0.")

    return args


class RunData:
    def __init__(self):
        self.__id = datetime.now().strftime("Y%YM%mD%dh%Hm%Ms%S")
        self.__host = socket.gethostname()
        self.__start_time = time()

    @property
    def id(self):
        return self.__id

    @property
    def host(self):
        return self.__host

    @property
    def elapsed_time(self):
        return int(time() - self.__start_time)



def select_device(use_gpu: bool) -> str:
    """Select a target device (cpu vs gpu) for operations. The use_gpu boolean flag represents
    the user intent. Returns the device ID for PyTorch operations (mps, cuda:0, or cpu).
    In the case of machines with multiple cuda GPUs, it always returns the first GPU.
    """

    # Extract available device. CUDA is checked first because a machine that offers it is the
    # intended training platform.
    if torch.cuda.is_available():
        device = "cuda:0"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # Determine device use from user input
    if use_gpu:
        match device:
            case "cpu":
                print("[Info] A GPU is not available. Using CPU instead.")
            case "mps":
                print("[Info] Using a MAC GPU")
            case "cuda:0":
                print("[Info] Using a CUDA GPU")
        result = device
    else:
        result = "cpu"

    return result


# Define a training function
def train(model, train_loader, optimizer,
          criterion, device: str) -> Tuple[float, float]:
    """Run a backpropagation algorithm using the complete training dataset (i.e., run one epoch).
    Arguments:
        model: A handler that references the model being trained.
        train_loader: A data loading function that brings in training data samples
        optimizer: A function that defines the type of optimizer
        criterion: The loss function being optimized
        device: The selected device to run a training procedure
    Return:
        A pair of numbers: per-epoch loss value and accuracy.
    """
    model.train()
    print('Running the training phase')
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
        counter += 1
        # The dataset yields a dictionary of batched fields (see dm.PartitionDataset.__getitem__)
        image, labels = data["image"], data["label"]
        image = image.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # Forward pass.
        outputs = model(image)
        # Calculate the loss.
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()
        # Calculate the accuracy.
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == labels).sum().item()
        # Backpropagation.
        loss.backward()
        # Update the weights.
        optimizer.step()

    # Loss and accuracy for the complete epoch.
    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_correct / len(train_loader.dataset))
    return epoch_loss, epoch_acc


def validate(model, valid_loader, criterion, device: str) -> Tuple[float, float]:
    """Apply a model to the validation dataset and compute the loss and accuracy values.
    Arguments:
        model: A handler that references the model being trained.
        valid_loader: A data loading function that brings in validation data samples
        criterion: The loss function
        device: The selected device to run a validation procedure
    Return:
        A pair of numbers: loss value and accuracy.
    """
    model.eval()
    print('Running the validation phase')
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0

    with torch.no_grad():
        for i, data in tqdm(enumerate(valid_loader), total=len(valid_loader)):
            counter += 1

            image, labels = data["image"], data["label"]
            image = image.to(device)
            labels = labels.to(device)
            # Forward pass.
            outputs = model(image)
            # Calculate the loss.
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()
            # Calculate the accuracy.
            _, preds = torch.max(outputs.data, 1)
            valid_running_correct += (preds == labels).sum().item()

    # Loss and accuracy for the complete epoch.
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100. * (valid_running_correct / len(valid_loader.dataset))
    return epoch_loss, epoch_acc


def save_learning_curves(train_data: dict, out_dir: str, run_id: str, keyword: str) -> str:
    """Write the per-epoch losses and accuracies of every fold to a single CSV file and return
    the path of the file."""
    records = []
    for fold_number, curves in train_data.items():
        for epoch_index in range(len(curves["trn_loss"])):
            records.append({
                "fold": fold_number,
                "epoch": epoch_index + 1,
                "trn_loss": curves["trn_loss"][epoch_index],
                "trn_acc": curves["trn_acc"][epoch_index],
                "val_loss": curves["val_loss"][epoch_index],
                "val_acc": curves["val_acc"][epoch_index],
            })

    prefix = "_".join(part for part in (keyword, run_id) if part)
    file_path = os.path.join(out_dir, f"{prefix}_{con.RESULTS_FILE_NAME}")
    pd.DataFrame(records).to_csv(file_path, index=False)
    return file_path


def save_run_metadata(args, run_data: "RunData", target_device: str, fold_results: list,
                      out_dir: str) -> str:
    """Write the description of a run (host, timing, arguments, and per-fold outcomes) to a JSON
    file and return the path of the file."""
    metadata = {
        "run_id": run_data.id,
        "host": run_data.host,
        "elapsed_seconds": run_data.elapsed_time,
        "device": target_device,
        "torch_version": torch.__version__,
        "random_seed": con.RANDOM_SEED,
        "labels": con.ORDERED_LABELS,
        "images_per_tumor_type": con.IMAGES_PER_TUMOR_TYPE,
        "arguments": vars(args),
        "folds": fold_results,
    }

    prefix = "_".join(part for part in (args.keyword, run_data.id) if part)
    file_path = os.path.join(out_dir, f"{prefix}_{con.RUN_META_FILE_NAME}")
    with open(file_path, "w") as file_handle:
        json.dump(metadata, file_handle, indent=4)
    return file_path


def main():
    """Main sequence of operations."""
    # [1] Collect user-defined arguments
    args = parse_arguments()

    # [2] Instantiate a RunData object and make the run reproducible
    run_data = RunData()
    dm.set_random_seeds(con.RANDOM_SEED)

    # [3] Create a directory to store results
    results = dm.ResultsStore(args.results_dir_path)

    # [4] Determine target device
    target_device = select_device(args.gpu)

    # [5] Create a reference to the data source
    data_src = dm.MRIDataSource(root_dir=args.data_dir_path)

    # [6] Train models using K-fold cross-validation
    fold_number = 0
    train_data = {}
    fold_results = []
    for fold_df in data_src.get_kfold_partitions(test_percent=15, kvalue=5,
                                                 images_per_tumor_type=con.IMAGES_PER_TUMOR_TYPE):
        fold_number += 1
        # [6.1] Get dataset and loader representations for train (trn) and validation (val) data
        trn_dataset = dm.PartitionDataset(fold_data_df=fold_df, subset='trn', transform=dm.get_train_transform())
        val_dataset = dm.PartitionDataset(fold_data_df=fold_df, subset='val', transform=dm.get_valid_transform())

        trn_loader, val_loader = dm.get_data_loaders(dataset_train=trn_dataset,
                                                     dataset_valid=val_dataset,
                                                     batch_size=args.batch)

        # [6.2] Create a model reference and transfer it to the target device
        model = mod.swin_classifier(size=args.model_size, version=args.model_version,
                                    num_classes=len(con.ORDERED_LABELS))
        model = model.to(target_device)

        # [6.3] Define operational functions for training/validation
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate)
        criterion = torch.nn.CrossEntropyLoss()

        # [6.4] Define lists to keep track of losses and accuracies
        train_data[fold_number] = {
            "trn_loss": [],
            "trn_acc": [],
            "val_loss": [],
            "val_acc": []
        }

        # [6.5] Initialize a model storage object
        mod_store = sav.BestModelStore()

        # [6.6] Perform training procedures for all epochs
        for epoch in range(args.epochs):
            print(f"Epoch {epoch + 1} of {args.epochs}")
            trn_epoch_loss, trn_epoch_acc = train(model, trn_loader, optimizer, criterion, target_device)
            val_epoch_loss, val_epoch_acc = validate(model, val_loader, criterion, target_device)

            train_data[fold_number]["trn_loss"].append(trn_epoch_loss)
            train_data[fold_number]["trn_acc"].append(trn_epoch_acc)
            train_data[fold_number]["val_loss"].append(val_epoch_loss)
            train_data[fold_number]["val_acc"].append(val_epoch_acc)

            print(f"fold: {fold_number}, trn loss: {trn_epoch_loss:.3f}, trn acc: {trn_epoch_acc:.3f}, "
                  f"val loss: {val_epoch_loss:.3f}, val acc: {val_epoch_acc:.3f}")

            mod_store(current_val_acc=val_epoch_acc, fold_num=fold_number, epoch=epoch, model=model,
                      out_dir=results.models_dir, run_id=run_data.id, labels=con.ORDERED_LABELS,
                      keyword=args.keyword)

            print('-' * 50)

        # [6.7] Save the model obtained after the last epoch of this fold and record its outcome
        final_model_path = sav.save_model(model_type="final", fold_num=fold_number, epoch=args.epochs - 1,
                                          model=model, out_dir=results.models_dir, run_id=run_data.id,
                                          labels=con.ORDERED_LABELS, keyword=args.keyword)

        fold_results.append({
            "fold": fold_number,
            "best_val_acc": mod_store.best_val_acc,
            "best_epoch": None if mod_store.best_epoch is None else mod_store.best_epoch + 1,
            "final_val_acc": train_data[fold_number]["val_acc"][-1],
            "final_model_path": final_model_path,
        })

        # [6.8] Release the model of this fold before the next one is created, so that the memory
        # of the target device does not accumulate one model per fold.
        del model, optimizer, trn_loader, val_loader, trn_dataset, val_dataset
        if target_device.startswith("cuda"):
            torch.cuda.empty_cache()

    # [7] Save run metadata and learning curves
    curves_path = save_learning_curves(train_data=train_data, out_dir=results.logs_dir,
                                       run_id=run_data.id, keyword=args.keyword)
    metadata_path = save_run_metadata(args=args, run_data=run_data, target_device=target_device,
                                      fold_results=fold_results, out_dir=results.logs_dir)

    mean_best_acc = sum(item["best_val_acc"] for item in fold_results) / len(fold_results)
    print(f"Mean best validation accuracy across {len(fold_results)} folds: {mean_best_acc:.3f}")
    print(f"Models saved in: {results.models_dir}")
    print(f"Learning curves saved in: {curves_path}")
    print(f"Run metadata saved in: {metadata_path}")
    print(f"Elapsed time: {run_data.elapsed_time} seconds")

    print('Finished training a SWIN transformer model.')

if __name__ == "__main__":
    main()
