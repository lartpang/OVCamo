# -*- coding: utf-8 -*-
from collections import defaultdict

import numpy as np
from py_sod_metrics.sod_metrics import MAE, Emeasure, Fmeasure, Smeasure, WeightedFmeasure
from py_sod_metrics.utils import TYPE, get_adaptive_threshold, prepare_data


class ClasswiseSmeasure(Smeasure):
    def __init__(self, class_names: tuple, alpha: float = 0.5):
        self.class_names = class_names
        self.sms = {name: [] for name in class_names}
        self.alpha = alpha

    def step(self, pred: np.ndarray, gt: np.ndarray, pre_cls: str, gt_cls: str):
        pred, gt = prepare_data(pred=pred, gt=gt)
        sm = self.cal_sm(pred, gt)

        sm = np.asarray(sm).reshape(1, 1)
        if pre_cls != gt_cls:
            sm.fill(0)

        self.sms[gt_cls].append(sm)

    def get_results(self):
        sm = {}
        for name in self.class_names:
            if not self.sms[name]:
                continue

            sm[name] = np.concatenate(self.sms[name], axis=0, dtype=TYPE)
        return dict(sm=sm)


class ClasswiseMAE(MAE):
    def __init__(self, class_names: tuple):
        self.class_names = class_names
        self.maes = {name: [] for name in class_names}

    def step(self, pred: np.ndarray, gt: np.ndarray, pre_cls: str, gt_cls: str):
        pred, gt = prepare_data(pred, gt)
        mae = self.cal_mae(pred, gt)

        mae = np.asarray(mae).reshape(1, 1)
        if pre_cls != gt_cls:
            mae.fill(1)

        self.maes[gt_cls].append(mae)

    def get_results(self):
        mae = {}
        for name in self.class_names:
            if not self.maes[name]:
                continue

            mae[name] = np.concatenate(self.maes[name], axis=0, dtype=TYPE)
        return dict(mae=mae)


class ClasswiseWeightedFmeasure(WeightedFmeasure):
    def __init__(self, class_names: tuple, beta: float = 1):
        self.class_names = class_names
        self.beta = beta
        self.weighted_fms = {name: [] for name in class_names}

    def step(self, pred: np.ndarray, gt: np.ndarray, pre_cls: str, gt_cls: str):
        pred, gt = prepare_data(pred=pred, gt=gt)
        if np.all(~gt):
            wfm = 0
        else:
            wfm = self.cal_wfm(pred, gt)

        wfm = np.asarray(wfm).reshape(1, 1)
        if pre_cls != gt_cls:
            wfm.fill(0)

        self.weighted_fms[gt_cls].append(wfm)
        return wfm

    def get_results(self):
        wfm = {}
        for name in self.class_names:
            if not self.weighted_fms[name]:
                continue

            wfm[name] = np.concatenate(self.weighted_fms[name], axis=0, dtype=TYPE)
        return dict(wfm=wfm)


class ClasswiseFmeasure(Fmeasure):
    def __init__(self, class_names: tuple, beta: float = 0.3):
        self.beta = beta
        self.class_names = class_names
        self.precisions = {name: [] for name in class_names}
        self.recalls = {name: [] for name in class_names}
        self.adaptive_fms = {name: [] for name in class_names}
        self.changeable_fms = {name: [] for name in class_names}

    def step(self, pred: np.ndarray, gt: np.ndarray, pre_cls: str, gt_cls: str):
        pred, gt = prepare_data(pred, gt)
        adaptive_fm = self.cal_adaptive_fm(pred=pred, gt=gt)
        precision, recall, changeable_fm = self.cal_pr(pred=pred, gt=gt)

        adaptive_fm = np.asarray(adaptive_fm).reshape(1, 1)
        precision = np.asarray(precision).reshape(1, -1)
        recall = np.asarray(recall).reshape(1, -1)
        changeable_fm = np.asarray(changeable_fm).reshape(1, -1)
        if pre_cls != gt_cls:
            adaptive_fm.fill(0)
            precision.fill(0)
            recall.fill(0)
            changeable_fm.fill(0)

        self.adaptive_fms[gt_cls].append(adaptive_fm)
        self.precisions[gt_cls].append(precision)
        self.recalls[gt_cls].append(recall)
        self.changeable_fms[gt_cls].append(changeable_fm)

    def get_results(self):
        adaptive_fm = {}
        changeable_fm = {}
        precision = {}
        recall = {}
        for name in self.class_names:
            if not self.adaptive_fms[name]:
                continue

            adaptive_fm[name] = np.concatenate(self.adaptive_fms[name], axis=0, dtype=TYPE)  # N,1
            precision[name] = np.concatenate(self.precisions[name], axis=0, dtype=TYPE)  # N,256
            recall[name] = np.concatenate(self.recalls[name], axis=0, dtype=TYPE)
            changeable_fm[name] = np.concatenate(self.changeable_fms[name], axis=0, dtype=TYPE)
        return dict(
            fm=dict(adp=adaptive_fm, curve=changeable_fm),
            pr=dict(p=precision, r=recall),
        )


class ClasswiseEmeasure(Emeasure):
    def __init__(self, class_names: tuple):
        self.class_names = class_names
        self.adaptive_ems = {name: [] for name in class_names}
        self.changeable_ems = {name: [] for name in class_names}

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

        self.adaptive_ems[gt_cls].append(adaptive_em)
        self.changeable_ems[gt_cls].append(changeable_em)

    def get_results(self):
        adaptive_em = {}
        changeable_em = {}
        for name in self.class_names:
            if not self.adaptive_ems[name]:
                continue

            adaptive_em[name] = np.concatenate(self.adaptive_ems[name], axis=0, dtype=TYPE)
            changeable_em[name] = np.concatenate(self.changeable_ems[name], axis=0, dtype=TYPE)
        return dict(em=dict(adp=adaptive_em, curve=changeable_em))


class ClasswiseIOU:
    def __init__(self, class_names: tuple):
        self.class_names = class_names
        self.adaptive_ious = {name: [] for name in class_names}
        self.changeable_ious = {name: [] for name in class_names}

    def step(self, pred: np.ndarray, gt: np.ndarray, pre_cls: str, gt_cls: str):
        pred, gt = prepare_data(pred, gt)

        adaptive_iou = self.cal_adaptive_iou(pred=pred, gt=gt)
        changeable_iou = self.cal_changeable_iou(pred=pred, gt=gt)

        adaptive_iou = np.asarray(adaptive_iou).reshape(1, 1)
        changeable_iou = np.asarray(changeable_iou).reshape(1, -1)
        if pre_cls != gt_cls:
            adaptive_iou.fill(0)
            changeable_iou.fill(0)

        self.adaptive_ious[gt_cls].append(adaptive_iou)
        self.changeable_ious[gt_cls].append(changeable_iou)

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
        # class_name: metric_values
        adaptive_iou = {}
        changeable_iou = {}
        for name in self.class_names:
            if not self.adaptive_ious[name]:
                continue

            adaptive_iou[name] = np.concatenate(self.adaptive_ious[name], axis=0, dtype=TYPE)
            changeable_iou[name] = np.concatenate(self.changeable_ious[name], axis=0, dtype=TYPE)
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
    "mae": ClasswiseMAE,
    "em": ClasswiseEmeasure,
    "sm": ClasswiseSmeasure,
    "wfm": ClasswiseWeightedFmeasure,
    "fm": ClasswiseFmeasure,
    "iou": ClasswiseIOU,
}


class OVCOSMetricer:
    suppoted_metrics = sorted(METRIC_MAPPING.keys())

    def __init__(self, class_names, metric_names=("sm", "wfm", "mae", "fm", "em", "iou")):
        self.class_names = class_names
        if not metric_names:
            metric_names = self.suppoted_metrics
        assert set(metric_names).issubset(self.suppoted_metrics), f"Only support: {self.suppoted_metrics}"

        self.metric_objs = {n: METRIC_MAPPING[n](class_names=class_names) for n in metric_names}

    def step(self, pre: np.ndarray, gt: np.ndarray, pre_cls: str, gt_cls: str, gt_path: str = None):
        assert pre.shape == gt.shape, (pre.shape, gt.shape, gt_path)
        assert pre.dtype == gt.dtype == np.uint8, (pre.dtype, gt.dtype, gt_path)
        return {n: obj.step(pre, gt, pre_cls, gt_cls) for n, obj in self.metric_objs.items()}

    def _get_raw_results(self) -> dict:
        """{metric_name: array([num_classes])}"""
        numerical_results = defaultdict(dict)
        for m_name, m_obj in self.metric_objs.items():
            info = m_obj.get_results()  # class_name as the key
            results = info[m_name]  # class_name: metric_values

            # class_results: N,1/256
            if m_name in ("wfm", "sm", "mae"):
                for class_name, class_results in results.items():
                    res = class_results.mean()  # N,1 -> 0
                    numerical_results[m_name][class_name] = res
            elif m_name in ("fm", "em", "iou"):
                for class_name in self.class_names:
                    # N,256->256
                    if class_name not in results["adp"]:
                        continue

                    adp_res = results["adp"][class_name].mean()
                    max_res = results["curve"][class_name].mean(axis=0)
                    avg_res = results["curve"][class_name].mean(axis=0)

                    numerical_results[f"adp{m_name}"][class_name] = adp_res
                    numerical_results[f"max{m_name}"][class_name] = max_res.max()
                    numerical_results[f"avg{m_name}"][class_name] = avg_res.mean()
            else:
                raise NotImplementedError(m_name)
        return numerical_results  # metric_name, class_name, value

    def show(self, num_bits: int = 3):
        """{metric_name: {avg_value_of_all_classes}}"""
        raw_results = self._get_raw_results()  # metric_name, class_name, value

        # average results of all classes for each metric
        avg_results = {k: np.array(list(v.values())).mean() for k, v in raw_results.items()}

        if num_bits is not None and isinstance(num_bits, int):
            avg_results = {k: v.round(num_bits) for k, v in avg_results.items()}

        avg_results = ndarray_to_basetype(avg_results)
        return avg_results
