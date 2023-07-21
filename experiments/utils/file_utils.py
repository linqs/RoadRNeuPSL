import os

EXPERIMENT_SUMMARY_FILENAME = "experiment_summary.csv"
TRAINED_MODEL_FILENAME = "trained_model_parameters.pt"
TRAINING_CONVERGENCE_FILENAME = "training_convergence.csv"
TRAINING_SUMMARY_FILENAME = "training_summary.csv"
TEST_EVALUATION_FILENAME = "test_evaluation.csv"


def check_cached_file(out_file: str):
    """
    Check if the file with the provided path already exists.
    :param out_file: The path to the file.
    :return: True if the output directory contains an out.txt file indicating the experiment has been ran.
    """
    return os.path.exists(out_file)


def make_run_dir(out_directory: str):
    """
    Make the run output directory. If the directory exists, do nothing.
    :param out_directory: The path to the run output directory.
    """
    os.makedirs(out_directory, exist_ok=True)
