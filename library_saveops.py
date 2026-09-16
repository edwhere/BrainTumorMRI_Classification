
import os
import torch
import torch.nn
from typing import Union


class BestModelStore:
    """ Class to save the best model during training. If the current epoch's accuracy is better
    than the stored validation accuracy, the current model is better and will be saved.
    """

    def __init__(self):
        self.best_val_acc = 0
        self.best_epoch = None

    def __call__(self, current_val_acc: float, fold_num: int, epoch: int, model: torch.nn.Module,
                 out_dir: str, run_id: str, labels: list, keyword: str = ""):

        if current_val_acc > self.best_val_acc:
            self.best_val_acc = current_val_acc
            self.best_epoch = epoch
            print(f"\nBest validation acc: {self.best_val_acc}. Saving model for epoch {epoch + 1}\n")

            save_model(model_type="best", fold_num=fold_num, epoch=epoch,
                       model=model, out_dir=out_dir, run_id=run_id, labels=labels, keyword=keyword)


def save_model(model_type: str, epoch: int, model: torch.nn.Module,
               out_dir: str, run_id: str, labels: list, fold_num: Union[int, None]=None,
               keyword: str = "") -> str:
    """ Function to save a trained model to an output directory.
    Args:
        model_type (str): Type of model to be saved. It can be 'best' for an optimal model or 'final' if it is the
            last model after all epochs.
        epoch (int): The epoch that generated the model.
        model (torch.nn.Module): The model to be saved.
        out_dir (str): The output directory where models are saved.
        run_id (str): An id of the current run.
        labels (list): A list of class names ordered by class index. The list must be picklable, so
            views such as the result of dict.keys() are not valid input.
        fold_num (int): The fold number if models are generated using K-fold cross validation. It can be
            set to None.
        keyword (str): An optional keyword included in the file name of the saved model.
    Return:
        The path of the saved model file.
    """

    if model_type not in ["best", "final"]:
        raise ValueError(f"In save_model(), the model_type must be either 'best' or 'final' instead of {model_type}")

    fold_txt = "foldX" if fold_num is None else f"fold{fold_num}"

    name_parts = [part for part in (model_type, keyword, fold_txt, run_id) if part]
    file_name = "_".join(name_parts) + ".pth"

    stored_data = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'labels': list(labels)
    }

    file_path = str(os.path.join(out_dir, file_name))
    torch.save(stored_data, file_path)
    return file_path
