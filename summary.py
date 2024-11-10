from glob import glob
import numpy as np
import os
import json
import csv
import sys

DEFAULT_MESH_DICT = {
    "completeness": -1,
    "accuracy": -1,
    "chamfer-L1": -1,
    "f-score": -1,
    "f-score-15": -1,
    "f-score-20": -1,
}

# root = "output/original/TanksAndTemple"
# root = "output/ablation/TanksAndTemple_residual"
# root = "output/ablation/GlossySynthetic_residual"
# root = "output/original/GlossySynthetic"
root = sys.argv[1]

f_names = sorted(glob(os.path.join(root, "*")))
summary = {}
for f_name in f_names:
    results_path = os.path.join(f_name, "results.json")
    if not os.path.exists(results_path):
        continue
    with open(results_path) as json_file:
        contents = json.load(json_file)
        # method = sorted(contents.keys(), key=lambda method : int(method.split('_')[-1]))[-1]
        method = sorted(contents.keys(), key=lambda method: float(contents[method]["PSNR"]))[-1]
        try:
            summary["Method"].append(os.path.basename(f_name))
            summary["PSNR"].append(contents[method]["PSNR"])
            summary["SSIM"].append(contents[method]["SSIM"])
            summary["LPIPS"].append(contents[method]["LPIPS"])
        except:
            summary["Method"] = [os.path.basename(f_name)]
            summary["PSNR"] = [contents[method]["PSNR"]]
            summary["SSIM"] = [contents[method]["SSIM"]]
            summary["LPIPS"] = [contents[method]["LPIPS"]]

    if os.path.exists(os.path.join(f_name, "results_mesh.json")):
        with open(os.path.join(f_name, "results_mesh.json")) as json_file:
            contents = json.load(json_file)
            for key, value in DEFAULT_MESH_DICT.items():
                summary[key].append(contents[method].get(key, value))

summary["Method"].append("Avg.")
summary["PSNR"].append(np.mean(summary["PSNR"]))
summary["SSIM"].append(np.mean(summary["SSIM"]))
summary["LPIPS"].append(np.mean(summary["LPIPS"]))

for key in DEFAULT_MESH_DICT.keys():
    summary[key].append(np.mean(summary[key]))

with open(os.path.join(root, "summary.csv"), "w") as file_obj:
    writer_obj = csv.writer(file_obj)
    writer_obj.writerow(["Method"] + summary["Method"])
    writer_obj.writerow(["PSNR"] + summary["PSNR"])
    writer_obj.writerow(["SSIM"] + summary["SSIM"])
    writer_obj.writerow(["LPIPS"] + summary["LPIPS"])
    for key in DEFAULT_MESH_DICT.keys():
        writer_obj.writerow([key] + summary[key])
