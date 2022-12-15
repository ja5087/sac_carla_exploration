# Import libraries
import os
import re

import matplotlib.pyplot as plt
import pandas as pd

# Valid experimental parameters
VALID_EXPERIMENT_PARAMETERS = {
    "drop_ln_spec": {
        "name": "DroQ + SpecNorm",
        "color": "green",
    },
    "drop_spec": {
        "name": "Dropout + SpecNorm",
        "color": "brown",
    },
    "droq": {
        "name": "DroQ",
        "color": "yellow",
    },
    "redq": {
        "name": "ReDQ",
        "color": "aqua",
    },
    "l1": {
        "name": "L1",
        "color": "black",
    },
    "l2_0_1": {
        "name": "L2 (0.1)",
        "color": "salmon",
    },
    "l2": {
        "name": "L2 (0.01)",
        "color": "red",
    },
    "l2_0_001": {
        "name": "L2 (0.001)",
        "color": "darkred",
    },
    "l1_0_5": {
        "name": "L1 (0.5)",
        "color": "lightblue",
    },
    "l1_0_05": {
        "name": "L1 (0.05)",
        "color": "blue",
    },
    "l1_0_005": {
        "name": "L1 (0.005)",
        "color": "darkblue",
    },
    "spec": {
        "name": "SpecNorm",
        "color": "orange",
    },
    "baseline": {
        "name": "SAC",
        "color": "gray",
    }
}

VALID_EXPERIMENTS = {
    "ant": "Ant-v2",
    "hopper": "Hopper-v2",
    "humanoid": "Humanoid-v2",
    "walker": "Walker2d-v2"
}

EXPERIMENT_FILTERS = {
    "all": {
        "set": set(VALID_EXPERIMENT_PARAMETERS.keys()),
        "name": "All Experiments"
    },
    "l2": {
        "set": set(["baseline", "l2_0_001", "l2", "l2_0_1"]),
        "name": "SAC vs L2"
    },
    "l1": {
        "set": set(["baseline", "l1_0_5", "l1_0_05", "l1_0_005"]),
        "name": "SAC vs L1"
    },
    "specnorm": {
        "set": set(["baseline", "spec"]),
        "name": "SAC vs SpecNorm"
    }
}

RUNS_DIRECTORY = "./runs"
GRAPHS_DIRECTORY = "./graphs"
EXPERIMENT_NAME_REGEX = re.compile(r"sac_([\w_]+)_(\w+)$")
WINDOW_LENGTH = 7

# Storing all experiments, keyed by
# their experiment names (see VALID_EXPERIMENTS)
all_experiments = {}
for experiment in VALID_EXPERIMENTS:
    all_experiments[experiment] = {}

for file in os.listdir(RUNS_DIRECTORY):
    exp_match = EXPERIMENT_NAME_REGEX.fullmatch(file)
    if exp_match is not None:
        # print("Found parameter", exp_match[1], "for experiment", exp_match[2])
        if exp_match[2] in all_experiments:
            # Gather the proper experiment, interpret the csv
            # then place inside the all_experiments list.
            contents = None
            should_input = True
            for path_name, _, path_files in os.walk(os.path.join(RUNS_DIRECTORY, file)):
                if 'progress.txt' in path_files:
                    try:
                        contents = pd.read_csv(
                            os.path.join(path_name, 'progress.txt'), sep='\t')
                    except Exception as e:
                        should_input = False
                        print("Could not read", path_name, e)

            if should_input:
                all_experiments[exp_match[2]][exp_match[1]] = contents

# Sort out all experiments
for experiment in all_experiments:
    print("experiment", experiment, "has", len(
        all_experiments[experiment]), "experiments")
    print("-> parameters:", sorted(all_experiments[experiment].keys()))

    for filter in EXPERIMENT_FILTERS:
        plt.title(VALID_EXPERIMENTS[experiment] + " (" +
                  EXPERIMENT_FILTERS[filter]["name"] + ")")

        for key in sorted(all_experiments[experiment]):
            if key in EXPERIMENT_FILTERS[filter]["set"]:
                df = all_experiments[experiment][key]
                x = df['TotalEnvInteracts'].rolling(
                    window=WINDOW_LENGTH).mean()
                y = df['AverageEpRet'].rolling(window=WINDOW_LENGTH).mean()
                y_std = df['StdEpRet'].rolling(window=WINDOW_LENGTH).mean()
                plt.fill_between(
                    x, y-y_std, y+y_std, color=VALID_EXPERIMENT_PARAMETERS[key]["color"], alpha=0.1)
                plt.plot(x, y,
                         label=VALID_EXPERIMENT_PARAMETERS[key]["name"],
                         color=VALID_EXPERIMENT_PARAMETERS[key]["color"])

        plt.legend()
        plt.xlabel("Total Environment Interactions")
        plt.ylabel("Average Episode Return")
        plt.savefig(os.path.join(GRAPHS_DIRECTORY,
                    experiment + '_' + filter) + '.pdf')
        plt.close()
