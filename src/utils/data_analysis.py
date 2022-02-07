import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

def dat_to_df(file_path: str) -> pd.DataFrame:
    
    """Loads a .dat files as a dataframe 

    Args:
        file_path: Path to the .dat file to load

    Returns:
        The dataframe

    """
    
    cols = ["iter",
            "point",
            "energy",
            "fidelity",
            "variance",
            "corr_length",
            "const_kernel",
            "std_energies",
            "average_distances",
            "nit",
            "time_opt_bayes",
            "time_qaoa",
            "time_opt_kernel",
            "time_step"]
    
    depth = int(file_path.split("_")[1])
    angle_labels = sum([[f"gamma_{j+1}", f"beta_{j+1}"] for j in range(depth)], [])
    columns = [cols[0]] + angle_labels + cols[2:]
    df = pd.DataFrame()
    content = np.loadtxt(file_path)
    content = content[:-1, :]
    tmp_df = pd.DataFrame(content, columns=columns)
    df = df.append(tmp_df)
                 
    return df


def save_plots(file_path: str, gs_energy: int, problem: str = None) -> None:
    
    """Given a .dat file, saves the corresponding plot 

    Args:
        file_path: Path to the .dat file to load
        gs_energy: True ground state of the Hamiltonian
        problem: Problem name, necessary to plot the best current benchmark (at the moment of writing, only "MAX-CUT" is supported)

    Returns:
        None

    """

    current_best_benchmark = None
    if problem == "MAX-CUT":
        current_best_benchmark = 0.9326

    _, axs = plt.subplots(2, 4, figsize=(12,6))
    plt.tight_layout()
    
    axs = sum(axs.tolist(), [])
    
    df = dat_to_df(file_path)
    
    file_path = Path(file_path)
    file_stem = file_path.stem
    folder_name = file_path.parents[0]
    p, points, warmup, train, trial, graph = list(map(int, file_stem.split("_")[1::2]))
    
    df["ratio"] = df["energy"]/gs_energy
    #best_point = df_data["energy"].idxmin()

    for i, col in enumerate(['energy',
                             'fidelity',
                             'variance',
                             'ratio',
                             'corr_length',
                             'const_kernel',
                             'std_energies',
                             'average_distances']):
        if col == 'ratio':
            if current_best_benchmark is not None:
                axs[i].plot(range(len(df[col])), [current_best_benchmark] * len(range(len(df[col]))), label = "benchmark", color = "red")
                axs[i].legend(loc="lower right")
    
        axs[i].plot(df[col], linestyle='dashed', marker='o', color='blue', markersize=3)
        axs[i].set_title(col, fontsize=13)
    
    plt.savefig(str(folder_name / file_stem) + ".pdf")