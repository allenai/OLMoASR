# %%
import pandas as pd
import numpy as np
import glob
import os
import json

# %%
def load_data(csv_file: str) -> np.ndarray:
    """
    Loads the WER values from a CSV file.

    Args:
        csv_file (str): Path to the CSV file.
        column_name (str): The name of the column containing WER values.

    Returns:
        np.ndarray: Array of WER values.
    """
    data = pd.read_csv(csv_file)
    return data[["wer", "ref_length"]].values


def weighted_avg(data):
    wer = np.array([value[0] for value in data])
    ref_length = np.array([value[-1] for value in data])
    avg = np.sum(wer * ref_length) / np.sum(ref_length)
    return avg


def bootstrap_sampling(data: np.ndarray, n_bootstrap: int = 1000) -> np.ndarray:
    """
    Performs bootstrapping on the given data.

    Args:
        data (np.ndarray): The original data array.
        n_bootstrap (int): Number of bootstrap samples to generate.

    Returns:
        np.ndarray: Array of bootstrap sample means.
    """
    sample_size = len(data)
    bootstrap_samples = [
        data[np.random.choice(list(range(len(data))), size=sample_size, replace=True)]
        for _ in range(n_bootstrap)
    ]
    bootstrap_means = [weighted_avg(sample) * 100 for sample in bootstrap_samples]
    return np.array(bootstrap_means)


def compute_statistics(bootstrap_means: np.ndarray) -> dict:
    """
    Computes mean, standard deviation, and standard error from bootstrap means.

    Args:
        bootstrap_means (np.ndarray): Array of bootstrap sample means.

    Returns:
        dict: A dictionary containing mean, standard deviation, and standard error.
    """
    mean_of_means = np.mean(bootstrap_means)
    std_dev = np.std(bootstrap_means)
    std_err = std_dev / np.sqrt(len(bootstrap_means))  # Standard error of the mean
    result = {
        "mean": mean_of_means,
        "std_dev": std_dev,
        "std_err": std_err,
        "bootstrap_means": bootstrap_means,
    }

    return result


def get_error_bars(csv_file: str, n_bootstrap: int = 1000) -> dict:
    """
    Computes error bars for WER values from a CSV file.

    Args:
        csv_file (str): Path to the CSV file containing WER values.
        n_bootstrap (int): Number of bootstrap samples to generate.

    Returns:
        dict: A dictionary containing mean, standard deviation, standard error, and bootstrap means.
    """
    data = load_data(csv_file)
    bootstrap_means = bootstrap_sampling(data, n_bootstrap)
    return compute_statistics(bootstrap_means)


def get_error_bars_overall(data_dir: str, n_bootstrap: int = 1000) -> dict:
    """
    Computes error bars for WER values from multiple CSV files in a directory.

    Args:
        data_dir (str): Path to the directory containing CSV files with WER values.
        n_bootstrap (int): Number of bootstrap samples to generate.

    Returns:
        dict: A dictionary containing mean, standard deviation, standard error, and bootstrap means.
    """
    csv_files = glob.glob(f"{data_dir}/*.csv")
    data_by_eval = [load_data(csv_file) for csv_file in csv_files]
    all_bootstrap_means = []
    for i, data in enumerate(data_by_eval):
        bootstrap_means = bootstrap_sampling(data, n_bootstrap)
        stats = compute_statistics(bootstrap_means)
        all_bootstrap_means.append(stats["bootstrap_means"])
        print(
            f"""Stats for {os.path.basename(csv_files[i].split('/')[-1]).split('_')[0]}: 
                Mean: {stats['mean']:.2f}, 
                Std Dev: {stats['std_dev']:.2f},
                Std Err: {stats['std_err']:.2f}\n"""
        )
    all_bootstrap_means = np.concatenate(all_bootstrap_means, axis=0)
    print(f"{all_bootstrap_means.shape=}\n")
    overall_mean = np.mean(all_bootstrap_means)
    overall_std_dev = np.std(all_bootstrap_means)
    overall_std_err = overall_std_dev / np.sqrt(
        len(all_bootstrap_means)
    )  # Standard error of the mean
    result = {
        "mean": overall_mean,
        "std_dev": overall_std_dev,
        "std_err": overall_std_err,
    }
    return result


# %%
long_model_stats = []
for p in glob.glob("logs/data/error_bars/main/*"):
    print(f"Processing {os.path.basename(p).split('/')[-1]}")
    data_dir = f"{p}/long_form"
    stats = get_error_bars_overall(data_dir, n_bootstrap=1000)
    stats["model"] = os.path.basename(p).split('/')[-1]
    long_model_stats.append(stats)
    print(
        f"""Overall Stats:
        Mean: {stats['mean']:.2f}, 
        Std Dev: {stats['std_dev']:.2f},
        Std Err: {stats['std_err']:.2f}\n\n"""
    )

print(long_model_stats)

with open("logs/data/error_bars/stats/long_main_olmoasr_stats.jsonl", "w") as f:
    for stats in long_model_stats:
        f.write(json.dumps(stats) + "\n")
# %%
