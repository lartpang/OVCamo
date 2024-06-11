import argparse
import json
import shutil
from pathlib import Path

import yaml
from tqdm import tqdm


def get_defined_root(path: str, root_info: dict):
    defined_root = None
    for key, _defined_root in root_info.items():
        if key in path:
            defined_root = _defined_root
            break
    if defined_root is None:
        raise KeyError(path, root_info.keys())
    return key, defined_root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-info", type=str, required=True, help="dataset.yaml")
    parser.add_argument("--class-info", type=str, required=True, help="Json file storing class information.")
    parser.add_argument("--sample-info", type=str, required=True, help="Json file storing sample information.")
    parser.add_argument("--train-root", type=str, required=True, help="Dir to store the data of training split.")
    parser.add_argument("--test-root", type=str, required=True, help="Dir to store the data of testing split.")
    args = parser.parse_args()

    train_root = Path(args.train_root)
    test_root = Path(args.test_root)

    with open(args.root_info, encoding="utf-8", mode="r") as f:
        root_info = yaml.safe_load(f)
    with open(args.class_info, encoding="utf-8", mode="r") as f:
        class_infos = json.load(f)
    with open(args.sample_info, encoding="utf-8", mode="r") as f:
        sample_infos = json.load(f)

    for sample_info in tqdm(sample_infos, total=len(sample_infos), ncols=78):
        base_class = sample_info["base_class"]
        for class_info in class_infos:
            if class_info["name"] == base_class:
                break

        sample_split = class_info["split"]
        image_root_key, image_root = get_defined_root(sample_info["image"], root_info)
        mask_root_key, mask_root = get_defined_root(sample_info["mask"], root_info)
        image_path = Path(sample_info["image"].replace(image_root_key, image_root))
        mask_path = Path(sample_info["mask"].replace(mask_root_key, mask_root))
        if sample_split == "train":
            new_image_path = train_root.joinpath("image", sample_info["unique_id"] + image_path.suffix)
            new_mask_path = train_root.joinpath("mask", sample_info["unique_id"] + mask_path.suffix)
        elif sample_split == "test":
            new_image_path = test_root.joinpath("image", sample_info["unique_id"] + image_path.suffix)
            new_mask_path = test_root.joinpath("mask", sample_info["unique_id"] + mask_path.suffix)
        else:
            raise ValueError(sample_split)

        new_image_path.parent.mkdir(parents=True, exist_ok=True)
        new_mask_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(image_path, new_image_path)
        shutil.copy(mask_path, new_mask_path)


if __name__ == "__main__":
    main()
