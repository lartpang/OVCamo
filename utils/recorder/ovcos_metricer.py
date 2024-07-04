# -*- coding: utf-8 -*-
from collections import defaultdict

import numpy as np
from py_sod_metrics.sod_metrics import MAE, Emeasure, Fmeasure, Smeasure, WeightedFmeasure
from py_sod_metrics.utils import TYPE, get_adaptive_threshold, prepare_data


class Smeasure(Smeasure):
    def __init__(self, alpha: float = 0.5):
        self.sms = []
        self.alpha = alpha

    def step(self, pred: np.ndarray, gt: np.ndarray, pre_cls: str, gt_cls: str):
        pred, gt = prepare_data(pred=pred, gt=gt)
        sm = self.cal_sm(pred, gt)

        sm = np.asarray(sm).reshape(1, 1)
        if pre_cls != gt_cls:
            sm.fill(0)

        self.sms.append(sm)

    def get_results(self):
        sm = np.concatenate(self.sms, axis=0, dtype=TYPE)  # N,1
        return dict(sm=sm)


class MAE(MAE):
    def __init__(self):
        self.maes = []

    def step(self, pred: np.ndarray, gt: np.ndarray, pre_cls: str, gt_cls: str):
        pred, gt = prepare_data(pred, gt)
        mae = self.cal_mae(pred, gt)

        mae = np.asarray(mae).reshape(1, 1)
        if pre_cls != gt_cls:
            mae.fill(1)

        self.maes.append(mae)

    def get_results(self):
        mae = np.concatenate(self.maes, axis=0, dtype=TYPE)  # N,1
        return dict(mae=mae)


class WeightedFmeasure(WeightedFmeasure):
    def __init__(self, beta: float = 1):
        self.beta = beta
        self.weighted_fms = []

    def step(self, pred: np.ndarray, gt: np.ndarray, pre_cls: str, gt_cls: str):
        pred, gt = prepare_data(pred=pred, gt=gt)
        if np.all(~gt):
            wfm = 0
        else:
            wfm = self.cal_wfm(pred, gt)

        wfm = np.asarray(wfm).reshape(1, 1)
        if pre_cls != gt_cls:
            wfm.fill(0)

        self.weighted_fms.append(wfm)
        return wfm

    def get_results(self):
        wfm = np.concatenate(self.weighted_fms, axis=0, dtype=TYPE)  # N,1
        return dict(wfm=wfm)


class Fmeasure(Fmeasure):
    def __init__(self, beta: float = 0.3):
        self.beta = beta
        self.adaptive_fms = []
        self.changeable_fms = []

    def step(self, pred: np.ndarray, gt: np.ndarray, pre_cls: str, gt_cls: str):
        pred, gt = prepare_data(pred, gt)
        adaptive_fm = self.cal_adaptive_fm(pred=pred, gt=gt)
        precision, recall, changeable_fm = self.cal_pr(pred=pred, gt=gt)

        adaptive_fm = np.asarray(adaptive_fm).reshape(1, 1)
        changeable_fm = np.asarray(changeable_fm).reshape(1, -1)
        if pre_cls != gt_cls:
            adaptive_fm.fill(0)
            changeable_fm.fill(0)

        self.adaptive_fms.append(adaptive_fm)
        self.changeable_fms.append(changeable_fm)

    def get_results(self):
        adaptive_fm = np.concatenate(self.adaptive_fms, axis=0, dtype=TYPE)  # N,1
        changeable_fm = np.concatenate(self.changeable_fms, axis=0, dtype=TYPE)  # N,256
        return dict(fm=dict(adp=adaptive_fm, curve=changeable_fm))


class Emeasure(Emeasure):
    def __init__(self):
        self.adaptive_ems = []
        self.changeable_ems = []

    def step(self, pred: np.ndarray, gt: np.ndarray, pre_cls: str, gt_cls: str):
        pred, gt = prepare_data(pred=pred, gt=gt)
        self.gt_fg_numel = np.count_nonzero(gt)
        self.gt_size = gt.shape[0] * gt.shape[1]
        adaptive_em = self.cal_adaptive_em(pred, gt)
        changeable_em = self.cal_changeable_em(pred, gt)

        adaptive_em = np.asarray(adaptive_em).reshape(1, 1)
        changeable_em = np.asarray(changeable_em).reshape(1, -1)
        if pre_cls != gt_cls:
            adaptive_em.fill(0)
            changeable_em.fill(0)

        self.adaptive_ems.append(adaptive_em)
        self.changeable_ems.append(changeable_em)

    def get_results(self):
        adaptive_em = np.concatenate(self.adaptive_ems, axis=0, dtype=TYPE)
        changeable_em = np.concatenate(self.changeable_ems, axis=0, dtype=TYPE)
        return dict(em=dict(adp=adaptive_em, curve=changeable_em))


class IOU:
    def __init__(self):
        self.adaptive_ious = []
        self.changeable_ious = []

    def step(self, pred: np.ndarray, gt: np.ndarray, pre_cls: str, gt_cls: str):
        pred, gt = prepare_data(pred, gt)

        adaptive_iou = self.cal_adaptive_iou(pred=pred, gt=gt)
        changeable_iou = self.cal_changeable_iou(pred=pred, gt=gt)

        adaptive_iou = np.asarray(adaptive_iou).reshape(1, 1)
        changeable_iou = np.asarray(changeable_iou).reshape(1, -1)
        if pre_cls != gt_cls:
            adaptive_iou.fill(0)
            changeable_iou.fill(0)

        self.adaptive_ious.append(adaptive_iou)
        self.changeable_ious.append(changeable_iou)

    def cal_adaptive_iou(self, pred: np.ndarray, gt: np.ndarray) -> float:
        # ``np.count_nonzero`` is faster and better
        adaptive_threshold = get_adaptive_threshold(pred, max_value=1)
        binary_predcition = pred >= adaptive_threshold
        assert binary_predcition.dtype == gt.dtype, (binary_predcition.dtype, gt.dtype)

        union = np.count_nonzero(np.bitwise_or(binary_predcition, gt))
        if union == 0:
            return 0
        else:
            inter = np.count_nonzero(np.bitwise_and(binary_predcition, gt))
            return inter / union

    def cal_changeable_iou(self, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
        FG = np.count_nonzero(gt)  # 真实前景, FG=(TPs+FNs)
        # BG = gt.size - FG  # 真实背景, BG=(TNs+FPs)
        # 1. 获取预测结果在真值前背景区域中的直方图
        pred: np.ndarray = (pred * 255).astype(np.uint8)
        bins: np.ndarray = np.linspace(0, 256, 257)
        tp_hist, _ = np.histogram(pred[gt], bins=bins)  # 最后一个bin为[255, 256]
        fp_hist, _ = np.histogram(pred[~gt], bins=bins)

        # 2. 使用累积直方图（Cumulative Histogram）获得对应真值前背景中大于不同阈值的像素数量
        # 这里使用累加（cumsum）就是为了一次性得出 >=不同阈值 的像素数量, 这里仅计算了前景区域
        tp_w_thrs = np.cumsum(np.flip(tp_hist))  # >= 255, >= 254, ... >= 1, >= 0
        fp_w_thrs = np.cumsum(np.flip(fp_hist))

        # 3. 计算对应的TP,FP,TN,FN
        TPs = tp_w_thrs  # 前景 预测为 前景
        FPs = fp_w_thrs  # 背景 预测为 前景
        FNs = FG - TPs  # 前景 预测为 背景
        # TNs = BG - FPs  # 背景 预测为 背景

        changeable_iou = np.array(TPs + FNs + FPs, dtype=TYPE)
        np.divide(TPs, changeable_iou, out=changeable_iou, where=changeable_iou != 0)
        return changeable_iou

    def get_results(self):
        adaptive_iou = np.concatenate(self.adaptive_ious, axis=0, dtype=TYPE)  # N,1
        changeable_iou = np.concatenate(self.changeable_ious, axis=0, dtype=TYPE)  # N,256
        return dict(iou=dict(adp=adaptive_iou, curve=changeable_iou))


def ndarray_to_basetype(data):
    """
    将单独的ndarray，或者tuple，list或者dict中的ndarray转化为基本数据类型，
    即列表(.tolist())和python标量
    """

    def _to_list_or_scalar(item):
        listed_item = item.tolist()
        if isinstance(listed_item, list) and len(listed_item) == 1:
            listed_item = listed_item[0]
        return listed_item

    if isinstance(data, (tuple, list)):
        results = [_to_list_or_scalar(item) for item in data]
    elif isinstance(data, dict):
        results = {k: _to_list_or_scalar(item) for k, item in data.items()}
    elif isinstance(data, (int, float)):
        results = data
    else:
        assert isinstance(data, np.ndarray), type(data)
        results = _to_list_or_scalar(data)
    return results


def round_w_zero_padding(x, bit_width):
    x = str(round(x, bit_width))
    x += "0" * (bit_width - len(x.split(".")[-1]))
    return x


METRIC_MAPPING = {
    "mae": MAE,
    "em": Emeasure,
    "sm": Smeasure,
    "wfm": WeightedFmeasure,
    "fm": Fmeasure,
    "iou": IOU,
}


class OVCOSMetricer:
    suppoted_metrics = sorted(METRIC_MAPPING.keys())

    def __init__(self, class_names, metric_names=("sm", "wfm", "mae", "fm", "em", "iou")):
        self.class_names = class_names
        if not metric_names:
            metric_names = self.suppoted_metrics
        assert set(metric_names).issubset(self.suppoted_metrics), f"Only support: {self.suppoted_metrics}"

        self.metric_objs = {n: METRIC_MAPPING[n]() for n in metric_names}

    def step(self, pre: np.ndarray, gt: np.ndarray, pre_cls: str, gt_cls: str, gt_path: str = None):
        assert pre.shape == gt.shape, (pre.shape, gt.shape, gt_path)
        assert pre.dtype == gt.dtype == np.uint8, (pre.dtype, gt.dtype, gt_path)
        return {n: obj.step(pre, gt, pre_cls, gt_cls) for n, obj in self.metric_objs.items()}

    def _get_raw_results(self) -> dict:
        """{metric_name: array([num_classes])}"""
        numerical_results = defaultdict(dict)
        for m_name, m_obj in self.metric_objs.items():
            results = m_obj.get_results()[m_name]  # np.ndarray

            # class_results: N,1/256
            if m_name in ("wfm", "sm", "mae"):
                avg_res = results.mean()  # N,1 -> 0
                numerical_results[m_name] = avg_res
            elif m_name in ("fm", "em", "iou"):
                # N,256->256
                adp_res = results["adp"].mean()
                max_res = results["curve"].mean(axis=0)
                avg_res = results["curve"].mean(axis=0)

                numerical_results[f"adp{m_name}"] = adp_res
                numerical_results[f"max{m_name}"] = max_res.max()
                numerical_results[f"avg{m_name}"] = avg_res.mean()
            else:
                raise NotImplementedError(m_name)
        return numerical_results  # metric_name, value

    def show(self, num_bits: int = 3):
        """{metric_name: {avg_value_of_all_classes}}"""
        avg_results = self._get_raw_results()  # metric_name, value

        if num_bits is not None and isinstance(num_bits, int):
            avg_results = {k: v.round(num_bits) for k, v in avg_results.items()}

        avg_results = ndarray_to_basetype(avg_results)
        return avg_results
