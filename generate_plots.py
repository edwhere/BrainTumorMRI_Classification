"""Reads the directory with results from swin_train.py and prepares figures for visualizing the results"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Suffixes of the files written by swin_train.py into the Logs directory. The part of the name
# before the suffix identifies the run.
CURVES_SUFFIX = '_learn_curves.csv'
METADATA_SUFFIX = '_run_metadata.json'

# Chart styling. Folds take colors from this list in order and the list is never cycled: the hues
# are a validated categorical palette whose order is what keeps neighboring folds distinguishable,
# including under color-vision deficiency. The marker shapes repeat that identity, so a fold can
# also be told apart without relying on its color.
FOLD_COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
FOLD_MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']

SURFACE_COLOR = '#fcfcfb'
TEXT_PRIMARY = '#0b0b0b'
TEXT_SECONDARY = '#52514e'
GRID_COLOR = '#e3e2de'

# Pastel wash behind the box of the boxplot. It is a background, not a data color: it is much
# lighter than any of the fold colors drawn on top of it.
BOX_FACE_COLOR = '#cde2fb'

FIGURE_SIZE = (19, 7)

# Fixed accuracy range shared by the three subplots
ACCURACY_LIMITS = (60, 100)

# Overlaid dots of the boxplot: diameter in points, horizontal spacing between two dots in units of
# the x axis, and the amplitude of the random jitter added to that spacing.
BOX_DOT_SIZE = 15
BOX_DOT_SPACING = 0.025
BOX_DOT_JITTER = 0.008
BOX_DOT_JITTER_SEED = 42

TITLE_FONTSIZE = 20
LABEL_FONTSIZE = 18
TICK_FONTSIZE = 16
LEGEND_FONTSIZE = 16
RUN_LABEL_FONTSIZE = 15

class ResultsData:
    def __init__(self, results_dir):
        self.results_dir = results_dir
        if not os.path.isdir(self.results_dir):
            raise NotADirectoryError('Results directory does not exist: {}'.format(results_dir))

        self.logs_dir = os.path.join(self.results_dir, 'Logs')
        if not os.path.isdir(self.logs_dir):
            raise NotADirectoryError('Logs directory does not exist: {}'.format(self.logs_dir))

        items = os.listdir(self.logs_dir)
        learn_curves_files = [item for item in items if item.endswith(CURVES_SUFFIX)]
        if len(learn_curves_files) == 0:
            raise FileNotFoundError('The learn-curve file is missing from the Logs directory')

        # A Logs directory accumulates one pair of files per run. Select the most recent learn-curve
        # file and then the metadata file of that same run, so that curves and metadata never
        # describe two different runs.
        curves_name = max(learn_curves_files,
                          key=lambda item: os.path.getmtime(os.path.join(self.logs_dir, item)))
        run_prefix = curves_name[:-len(CURVES_SUFFIX)]
        metadata_name = run_prefix + METADATA_SUFFIX

        self.run_prefix = run_prefix
        self.learn_curves_filepath = os.path.join(self.logs_dir, curves_name)
        self.metadata_filepath = os.path.join(self.logs_dir, metadata_name)

        if not os.path.isfile(self.metadata_filepath):
            raise FileNotFoundError(
                'The metadata file {} is missing from the Logs directory'.format(metadata_name))

        with open(self.metadata_filepath, 'r') as f:
            self.metadata_dict = json.load(f)

        self.learn_curves_df = pd.read_csv(self.learn_curves_filepath)

    def get_id(self):
        """Return the run_id for the selected results. The ID identifies the training run. It is a
        timestamp in hex format."""
        return self.metadata_dict["run_id"]

    def get_number_folds(self) -> int:
        """Return the number of folds described in the metadata json file."""
        return len(self.metadata_dict["folds"])

    def get_fold_indices(self) -> list[int]:
        """Return the indices of the folds described in the metadata json file. Typically, the list
        should have values from 1 to N, where N is the number of folds used during training."""
        indices = []
        for fold_dict in self.metadata_dict["folds"]:
            indices.append(fold_dict["fold"])
        return indices

    def get_fold_best_val_acc_scores(self) -> list[tuple[int, float]]:
        """Read the metadata json file to extract the best validation accuracy values per fold.
        Return a list of tuples of the form (fold_index, best_acc_value).
        """
        scores = []
        for fold_dict in self.metadata_dict["folds"]:
            scores.append((fold_dict["fold"], fold_dict["best_val_acc"]))
        return scores

    def get_learn_curves_per_fold(self, fold_index: int) -> pd.DataFrame:
        """Read the learning-curves dataframe. Return all train/validation accuracy and loss values for
        the selected fold."""
        return self.learn_curves_df[self.learn_curves_df["fold"] == fold_index]

    def get_curves(self, column: str) -> pd.DataFrame:
        """Read one column of the learning-curves dataframe and return a dataframe with that column
        for every fold. The returned dataframe has one column per fold and is indexed by epoch.
        The folds are the ones described in the metadata file.
        """
        fold_indices = self.get_fold_indices()
        single_columns = []

        for fold_index in fold_indices:
            fold_data_df = self.get_learn_curves_per_fold(fold_index)
            if fold_data_df.empty:
                raise ValueError('Fold {} of the metadata file has no rows in {}'.format(
                    fold_index, os.path.basename(self.learn_curves_filepath)))

            # Index by epoch. Concatenation aligns columns on the index, so fold columns that kept
            # their original row labels would not overlap and would produce NaN values.
            single_columns.append(fold_data_df.set_index('epoch')[column])

        return pd.concat(single_columns, axis=1, keys=fold_indices)

    def get_trn_acc_curves(self) -> pd.DataFrame:
        """Read the train accuracy values from the learning-curves dataframe. Return a dataframe
        with multiple columns (one per fold). The columns contain the train accuracy values."""
        return self.get_curves('trn_acc')

    def get_val_acc_curves(self) -> pd.DataFrame:
        """Read the validation accuracy values from the learning-curves dataframe. Return a dataframe
        with multiple columns (one per fold). The columns contain the validation accuracy values."""
        return self.get_curves('val_acc')

    def get_trn_loss_curves(self) -> pd.DataFrame:
        """Read the train loss values from the learning-curves dataframe. Return a dataframe
        with multiple columns (one per fold). The columns contain the train loss values."""
        return self.get_curves('trn_loss')

    def get_val_loss_curves(self) -> pd.DataFrame:
        """Read the validation loss values from the learning-curves dataframe. Return a dataframe
        with multiple columns (one per fold). The columns contain the validation loss values."""
        return self.get_curves('val_loss')



def get_fold_style(position: int) -> tuple[str, str]:
    """Return the (color, marker) pair for the fold that occupies the given position of the fold
    list. Colors are assigned in a fixed order and are never reused, so a chart is limited to as
    many folds as there are colors."""
    if position >= len(FOLD_COLORS):
        raise ValueError('Cannot plot more than {} folds in one figure. Reusing colors would make '
                         'two folds indistinguishable: plot the folds as separate figures '
                         'instead.'.format(len(FOLD_COLORS)))
    return FOLD_COLORS[position], FOLD_MARKERS[position]


def style_axes(axes: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    """Apply the common look to a subplot: recessive grid and axes, and text that never wears a
    data color."""
    axes.set_facecolor(SURFACE_COLOR)
    axes.set_title(title, color=TEXT_PRIMARY, fontsize=TITLE_FONTSIZE, pad=14)
    axes.set_xlabel(xlabel, color=TEXT_SECONDARY, fontsize=LABEL_FONTSIZE, labelpad=10)
    axes.set_ylabel(ylabel, color=TEXT_SECONDARY, fontsize=LABEL_FONTSIZE, labelpad=10)
    axes.tick_params(colors=TEXT_SECONDARY, labelsize=TICK_FONTSIZE, length=0)

    # Hairline solid gridlines, one step off the surface, drawn under the data
    axes.grid(axis='y', color=GRID_COLOR, linewidth=1, linestyle='-')
    axes.set_axisbelow(True)

    for name, spine in axes.spines.items():
        if name in ('top', 'right'):
            spine.set_visible(False)
        else:
            spine.set_color(GRID_COLOR)
            spine.set_linewidth(1)


def style_legend(legend) -> None:
    """Apply the common look to a legend box."""
    legend.get_frame().set_facecolor(SURFACE_COLOR)
    legend.get_frame().set_edgecolor(GRID_COLOR)
    legend.get_frame().set_linewidth(1)
    for text in legend.get_texts():
        text.set_color(TEXT_SECONDARY)
    if legend.get_title() is not None:
        legend.get_title().set_color(TEXT_SECONDARY)


def plot_accuracy_curves(axes: plt.Axes, curves_df: pd.DataFrame, title: str) -> None:
    """Draw one accuracy curve per fold on the given subplot. The dataframe has one column per
    fold and is indexed by epoch (see ResultsData.get_curves)."""
    epochs = curves_df.index.to_numpy()

    for position, fold_index in enumerate(curves_df.columns):
        color, marker = get_fold_style(position)
        axes.plot(epochs, curves_df[fold_index].to_numpy(),
                  color=color, linewidth=2, solid_joinstyle='round', solid_capstyle='round',
                  marker=marker, markersize=6,
                  # A ring in the surface color keeps markers legible where curves cross
                  markeredgecolor=SURFACE_COLOR, markeredgewidth=2,
                  label='Fold {}'.format(fold_index), zorder=3)

    style_axes(axes, title, 'Epoch', 'Accuracy (%)')
    axes.set_xticks(epochs)
    style_legend(axes.legend(title='Folds', loc='best',
                             fontsize=LEGEND_FONTSIZE, title_fontsize=LEGEND_FONTSIZE))


def plot_best_scores_boxplot(axes: plt.Axes, best_val_acc_scores: list[float],
                             fold_indices: list[int]) -> None:
    """Draw the distribution of the best validation accuracy scores as a boxplot, with the
    individual per-fold scores overlaid as dots, the median as a solid line, and the mean as a
    dashed line."""
    scores = list(best_val_acc_scores)

    axes.boxplot(scores, widths=0.3, showmeans=True, meanline=True,
                 # Every score is drawn as a dot below, so fliers would be drawn twice
                 showfliers=False,
                 # patch_artist draws the box as a filled patch instead of a plain outline
                 patch_artist=True,
                 boxprops=dict(facecolor=BOX_FACE_COLOR, edgecolor=TEXT_SECONDARY, linewidth=1),
                 medianprops=dict(color=TEXT_PRIMARY, linewidth=2),
                 meanprops=dict(color=TEXT_SECONDARY, linewidth=2, linestyle='--'),
                 whiskerprops=dict(color=TEXT_SECONDARY, linewidth=1),
                 capprops=dict(color=TEXT_SECONDARY, linewidth=1))

    # Spread the dots horizontally, so that folds with nearly equal scores remain visible as
    # separate circles. The even base spacing is wider than the diameter of a circle and does the
    # separating; the random jitter on top of it breaks up the regularity of a perfect row. The
    # generator is seeded, so the same scores always produce the same figure.
    count = len(scores)
    base_offsets = (np.arange(count) - (count - 1) / 2) * BOX_DOT_SPACING
    jitter = np.random.default_rng(BOX_DOT_JITTER_SEED).uniform(-BOX_DOT_JITTER, BOX_DOT_JITTER, count)
    offsets = base_offsets + jitter

    for position, (fold_index, score) in enumerate(zip(fold_indices, scores)):
        color, _ = get_fold_style(position)
        axes.plot(1 + offsets[position], score, linestyle='none',
                  marker='o', markersize=BOX_DOT_SIZE, color=color,
                  # A ring in the surface color separates a dot from the box fill and from a
                  # neighboring dot of a similar score
                  markeredgecolor=SURFACE_COLOR, markeredgewidth=2,
                  label='Fold {}'.format(fold_index), zorder=3)

    style_axes(axes, 'Best validation accuracy per fold', '', 'Accuracy (%)')
    axes.set_xticks([1], ['{} folds'.format(len(scores))])

    # The legend explains the two summary lines. The dots take their color from the fold, which the
    # legends of the accuracy subplots identify.
    handles = [Line2D([], [], color=TEXT_PRIMARY, linewidth=2, linestyle='-', label='median'),
               Line2D([], [], color=TEXT_SECONDARY, linewidth=2, linestyle='--', label='mean')]
    style_legend(axes.legend(handles=handles, loc='best', fontsize=LEGEND_FONTSIZE))


def plot_results(trn_acc_df: pd.DataFrame, val_acc_df: pd.DataFrame,
                 best_val_acc_scores: list[float], fold_indices: list[int],
                 run_label: str = '') -> plt.Figure:
    """Build the figure that summarizes a training run: train accuracy curves, validation accuracy
    curves, and the distribution of the best validation accuracy of each fold."""
    figure, axes_list = plt.subplots(1, 3, figsize=FIGURE_SIZE)
    figure.patch.set_facecolor(SURFACE_COLOR)

    plot_accuracy_curves(axes_list[0], trn_acc_df, 'Train accuracy')
    plot_accuracy_curves(axes_list[1], val_acc_df, 'Validation accuracy')
    plot_best_scores_boxplot(axes_list[2], best_val_acc_scores, fold_indices)

    # The three subplots share one fixed accuracy range, so that every value in the figure is read
    # against the same scale.
    for axes in axes_list:
        axes.set_ylim(*ACCURACY_LIMITS)

    if run_label:
        figure.suptitle(run_label, color=TEXT_SECONDARY, fontsize=RUN_LABEL_FONTSIZE, x=0.01, ha='left')

    figure.tight_layout()
    return figure


def parse_arguments():
    """Parses command line arguments"""
    parser = argparse.ArgumentParser(description='Plot results obtained from training a SWIN transformer.')
    required = parser.add_argument_group('required arguments')

    required.add_argument('-res', '--results_dir', type=str, required=True,
                          help='Path to a directory with results of training a SWIN transformer. This directory '
                               'contains subdirectories called Logs and Models.')

    parser.add_argument('-save', '--save_dir', type=str, default=None,
                        help="Path to a directory to save the figure. If omitted, the figures are displayed on "
                             "the screen, but not saved.")

    args = parser.parse_args()
    return args

def main():
    """Main function"""
    args = parse_arguments()
    results = ResultsData(args.results_dir)

    trn_acc_df = results.get_trn_acc_curves()
    val_acc_df = results.get_val_acc_curves()

    best_val_acc_scores = [pair[1] for pair in results.get_fold_best_val_acc_scores()]

    # plot results
    figure = plot_results(trn_acc_df=trn_acc_df, val_acc_df=val_acc_df,
                          best_val_acc_scores=best_val_acc_scores,
                          fold_indices=results.get_fold_indices(),
                          run_label=results.run_prefix)

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        id_value = results.get_id()
        figure_path = os.path.join(args.save_dir, 'swin_figures_{}.png'.format(id_value))
        figure.savefig(figure_path, dpi=150, facecolor=figure.get_facecolor())
        print('Saving figure to {}'.format(figure_path))

    print("\nBest accuracies:")
    for pair in results.get_fold_best_val_acc_scores():
        print(f"fold {pair[0]}: {pair[1]}")

    plt.show()


if __name__ == '__main__':
    main()
