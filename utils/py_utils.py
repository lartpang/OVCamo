# -*- coding: utf-8 -*-
import copy
import os
import shutil
from collections import OrderedDict
from datetime import datetime


def construct_path(output_dir: str, exp_name: str) -> dict:
    proj_root = os.path.join(output_dir, exp_name)
    exp_idx = 0
    exp_output_dir = os.path.join(proj_root, f"exp_{exp_idx}")
    while os.path.exists(exp_output_dir):
        exp_idx += 1
        exp_output_dir = os.path.join(proj_root, f"exp_{exp_idx}")

    tb_path = os.path.join(exp_output_dir, "tb")
    save_path = os.path.join(exp_output_dir, "pre")
    pth_path = os.path.join(exp_output_dir, "pth")

    final_full_model_path = os.path.join(pth_path, "checkpoint_final.pth")
    final_state_path = os.path.join(pth_path, "state_final.pth")

    log_path = os.path.join(exp_output_dir, f"log_{str(datetime.now())[:10]}.txt")
    cfg_copy_path = os.path.join(exp_output_dir, "config.py")
    trainer_copy_path = os.path.join(exp_output_dir, "trainer.txt")

    path_config = {
        "output_dir": output_dir,
        "pth_log": exp_output_dir,
        "tb": tb_path,
        "save": save_path,
        "pth": pth_path,
        "final_full_net": final_full_model_path,
        "final_state_net": final_state_path,
        "log": log_path,
        "cfg_copy": cfg_copy_path,
        "trainer_copy": trainer_copy_path,
    }

    return path_config


def construct_exp_name(model_name: str, cfg: dict):
    focus_item = OrderedDict(
        {
            "train/batch_size": "bs",
            "train/lr": "lr",
            "train/num_epochs": "e",
            "train/num_iters": "i",
            "train/input_hw": "hw",
            "train/optimizer/mode": "opm",
            "train/optimizer/group_mode": "opgm",
            "train/scheduler/mode": "sc",
            "train/scheduler/warmup/num_iters": "wu",
        }
    )
    config = copy.deepcopy(cfg)

    def _format_item(_i):
        if isinstance(_i, bool):
            _i = "" if _i else "false"
        elif isinstance(_i, (int, float)):
            if _i == 0:
                _i = "false"
        elif isinstance(_i, (list, tuple)):
            _i = "-".join([str(x) for x in _i]) if _i else "??"  # 只是判断是否非空
        elif isinstance(_i, str):
            if "_" in _i:
                _i = _i.replace("_", "").lower()
        elif _i is None:
            _i = "none"
        # else: other types and values will be returned directly
        return _i

    if (epoch_based := config.train.get("epoch_based", None)) is not None and (not epoch_based):
        focus_item.pop("train/num_epochs")
    else:
        # 默认基于epoch
        focus_item.pop("train/num_iters")

    exp_names = [model_name]
    for key, alias in focus_item.items():
        item = get_value_recurse(keys=key.split("/"), info=config)
        formatted_item = _format_item(item)
        if formatted_item == "false":
            continue
        exp_names.append(f"{alias.upper()}{formatted_item}")

    info = config.get("info", None)
    if info:
        exp_names.append(f"INFO{info.lower()}")

    return "_".join(exp_names)


def pre_mkdir(path_config):
    # 提前创建好记录文件，避免自动创建的时候触发文件创建事件
    check_mkdir(path_config["pth_log"])
    make_log(path_config["log"], f"=== log {datetime.now()} ===")

    # 提前创建好存储预测结果和存放模型的文件夹
    check_mkdir(path_config["save"])
    check_mkdir(path_config["pth"])


def check_mkdir(dir_name, delete_if_exists=False):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    else:
        if delete_if_exists:
            print(f"{dir_name} will be re-created!!!")
            shutil.rmtree(dir_name)
            os.makedirs(dir_name)


def make_log(path, context):
    with open(path, "a") as log:
        log.write(f"{context}\n")


def get_value_recurse(keys: list, info: dict):
    curr_key, sub_keys = keys[0], keys[1:]

    if (sub_info := info.get(curr_key, "NoKey")) == "NoKey":
        raise KeyError(f"{curr_key} must be contained in {info}")

    if sub_keys:
        return get_value_recurse(keys=sub_keys, info=sub_info)
    else:
        return sub_info


def get_defined_root(path: str, root_info: dict):
    defined_root = None
    for key, _defined_root in root_info.items():
        if path.startswith(key):
            defined_root = _defined_root
            break
    if defined_root is None:
        raise KeyError(path, root_info.keys())
    return key, defined_root
