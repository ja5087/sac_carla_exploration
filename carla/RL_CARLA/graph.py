# Import libraries
import os
import re

import matplotlib.pyplot as plt
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Valid experimental parameters
VALID_EXPERIMENT_PARAMETERS = {
    "droq": {
        "name": "DroQ (a=3, p=0.1)",
        "color": "purple",
    },
    "droq_0_01": {
        "name": "DroQ (a=3, p=0.01)",
        "color": "violet",
    },
    "droq_a1": {
        "name": "DroQ (a=1, p=0.1)",
        "color": "deeppink",
    },
    "redq": {
        "name": "ReDQ (a=3, ensemble 20)",
        "color": "darkorange",
    },
    "redq_c10": {
        "name": "ReDQ (a=3, ensemble 10)",
        "color": "lightsalmon",
    },
    "redq_c20_a1": {
        "name": "ReDQ (a=1, ensemble 20)",
        "color": "orangered",
    },
    "sac_a1": {
        "name": "SAC (a=1, UTD 1)",
        "color": "aqua",
    },
    "sac_a1_utd20": {
        "name": "SAC (a=1, UTD 20)",
        "color": "steelblue",
    },
    "sac_a3_utd20": {
        "name": "SAC (a=3, UTD 20)",
        "color": "dodgerblue",
    },
    "sac_baseline": {
        "name": "SAC (a=3, UTD 1)",
        "color": "slategray",
    },
}

EXPERIMENT_FILTERS = {
    "carla_q1": {
        "set": set(["droq", "redq", "sac_a3_utd20", "sac_baseline"]),
        "name": "SAC vs Other Variants",
        "cutoff": 50000,
    },
    "carla_q2": {
        "set": set(["redq", "redq_c10"]),
        "name": "ReDQ With Various Ensemble Sizes",
        "cutoff": 50000,
    },
    "carla_q3": {
        "set": set(["droq", "droq_0_01"]),
        "name": "DroQ With Various Dropout Rate",
        "cutoff": 50000,
    },
    "carla_q4": {
        "set": set(["droq_a1", "redq_c20_a1", "sac_a1_utd20", "sac_baseline"]),
        "name": "SAC, ReDQ, DroQ With Varying Number of Parallel Agents",
        "cutoff": 50000,
    },
    "carla_longrun": {
        "set": set(["sac_a3_utd20", "sac_baseline"]),
        "name": "SAC with varying UTD",
        "cutoff": 250000,
    }
}

RUNS_DIRECTORY = "./results"
GRAPHS_DIRECTORY = "./graphs"
EXPERIMENT_NAME_REGEX = re.compile(r"([\w_]+)$")
WINDOW_LENGTH = 7

carla_experiments = {}

for file in os.listdir(RUNS_DIRECTORY):
    exp_match = EXPERIMENT_NAME_REGEX.fullmatch(file)
    if exp_match is not None:
        # Gather the proper experiment, interpret the csv
        # then place inside the all_experiments list.
        contents = None
        should_input = True

        for path_name, _, _ in os.walk(os.path.join(RUNS_DIRECTORY, file)):
            pass

        try:
            event_acc = EventAccumulator(path_name)
            event_acc.Reload()

            contents = pd.DataFrame([
                (w, s, v) for w, s, v in event_acc.Scalars("eval/episode_reward")
            ], columns=['wall_time', 'step', 'eval'])
        except Exception as e:
            should_input = False
            print("Could not read", path_name, e)

        if should_input:
            carla_experiments[exp_match[1]] = contents

# Sort out all experiments
for filter in EXPERIMENT_FILTERS:
    plt.title(EXPERIMENT_FILTERS[filter]["name"])

    for key in sorted(carla_experiments):
        if key in EXPERIMENT_FILTERS[filter]["set"]:
            df = carla_experiments[key]
            x = df['step'].rolling(window=WINDOW_LENGTH).mean()
            y = df['eval'].rolling(window=WINDOW_LENGTH).mean()
            plt.xlim(0, EXPERIMENT_FILTERS[filter]["cutoff"])
            plt.plot(x, y,
                     label=VALID_EXPERIMENT_PARAMETERS[key]["name"],
                     color=VALID_EXPERIMENT_PARAMETERS[key]["color"])
    plt.xlabel("Number of Environment Steps")
    plt.ylabel("Average Episode Reward")
    plt.legend()
    plt.savefig(os.path.join(GRAPHS_DIRECTORY, filter) + '.pdf')
    plt.close()
