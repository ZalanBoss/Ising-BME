import os
import glob
import numpy as np

file_pattern = "data/non_zero_h/*.npz"
all_files = glob.glob(file_pattern)

dataset = []

for file_path in all_files:
    base_name = os.path.basename(file_path).replace(".npz", "")
    parts = base_name.split("_")

    beta = float(parts[0])
    h = float(parts[1])
    above_tc = parts[2] == "above"

    with np.load(file_path) as data:
        dataset.append({
            "beta": beta,
            "h": h,
            "above_tc": above_tc,
            "sigma": data["sigma"],
        })


#print(dataset[0]["sigma"])
