import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import tqdm
import yaml

from utils import io, recorder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre", type=str, help="Path of the predictions to evaluate.")
    parser.add_argument("--root-info", default="env/splitted_ovcamo.yaml", type=str)
    cfg = parser.parse_args()

    with open(cfg.root_info, mode="r") as f:
        cfg.root_info = yaml.safe_load(f)

    with open(cfg.root_info["OVCamo_CLASS_JSON_PATH"], mode="r", encoding="utf-8") as f:
        class_infos = json.load(f)
    with open(cfg.root_info["OVCamo_SAMPLE_JSON_PATH"], mode="r", encoding="utf-8") as f:
        sample_infos = json.load(f)

    dataset_classes = []
    for class_info in class_infos:
        if class_info["split"] == "test":
            dataset_classes.append(class_info["name"])

    total_data_paths = {}
    for sample_info in sample_infos:
        class_name = sample_info["base_class"]
        if class_name not in dataset_classes:
            continue

        unique_id = sample_info["unique_id"]
        gt_suffix = os.path.splitext(sample_info["mask"])[1]
        gt_path = os.path.join(cfg.root_info["OVCamo_TE_MASK_DIR"], unique_id + gt_suffix)
        total_data_paths[unique_id + gt_suffix] = (class_name, gt_path)
    print(f"[TestSet] {len(total_data_paths)} Samples, {len(dataset_classes)} Classes")

    metricer = recorder.OVCOSMetricer(class_names=dataset_classes)
    pre_paths = list(Path(cfg.pre).iterdir())
    for pre_path in tqdm.tqdm(pre_paths, total=len(pre_paths), ncols=79, desc="Evaluating"):
        pre_path = pre_path.as_posix()
        pre_cls, msk_name = re.findall(r"^\[(.*?)\](.*?)$", os.path.basename(pre_path))[0]
        pre = io.read_gray_array(pre_path, dtype=np.uint8)
        if msk_name not in total_data_paths:
            raise KeyError(f"[Error] No the corresponding mask: {pre_path}")

        gt_cls, gt_path = total_data_paths[msk_name]
        gt = io.read_gray_array(gt_path, thr=0)

        metricer.step(pre=pre, gt=(gt * 255).astype(np.uint8), pre_cls=pre_cls, gt_cls=gt_cls, gt_path=gt_path)
    avg_ovcos_results = metricer.show()
    print(avg_ovcos_results)


if __name__ == "__main__":
    main()
