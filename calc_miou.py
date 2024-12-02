import numpy as np
import os

def fast_hist(a, b, n):
    """
    计算混淆矩阵
    a: 标签，形状为(H×W,)
    b: 预测结果，形状为(H×W,)
    n: 类别总数
    """
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    """
    计算每个类别的IoU
    """
    print('Defect class IoU as follows:')
    print(np.diag(hist)[1:] / np.maximum((hist.sum(1) + hist.sum(0) - np.diag(hist))[1:], 1))
    return np.diag(hist)[1:] / np.maximum((hist.sum(1) + hist.sum(0) - np.diag(hist))[1:], 1)


def per_class_PA(hist):
    """
    计算每个类别的准确率
    """
    return np.diag(hist) / np.maximum(hist.sum(1), 1)


def compute_mIoU(gt_dir, pred_dir):
    """
    计算mIoU和mPA
    gt_dir: 真实标签文件夹
    pred_dir: 预测结果文件夹
    npy_name_list: 文件名列表
    num_classes: 类别总数
    """
    num_classes = 4
    print('Num classes', num_classes)
    hist = np.zeros((num_classes, num_classes))

    # 提取文件编号
    npy_name_list = [f.split('.')[0] for f in os.listdir(pred_dir) if f.endswith('.npy')]

    # 修改命名规则以匹配文件名
    gt_npy_files = [os.path.join(gt_dir, f"{x}.npy") for x in npy_name_list]
    pred_npy_files = [os.path.join(pred_dir, f"{x}.npy") for x in npy_name_list]

    for ind in range(len(gt_npy_files)):
        # 检查文件是否存在
        if not os.path.isfile(gt_npy_files[ind]):
            print(f"Ground truth file not found: {gt_npy_files[ind]}")
            continue

        if not os.path.isfile(pred_npy_files[ind]):
            print(f"Prediction file not found: {pred_npy_files[ind]}")
            continue

        # 读取npy文件
        pred = np.load(pred_npy_files[ind])
        label = np.load(gt_npy_files[ind])

        # 检查预测和标签的尺寸是否一致
        if len(label.flatten()) != len(pred.flatten()):
            print('Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(
                len(label.flatten()), len(pred.flatten()), gt_npy_files[ind],
                pred_npy_files[ind]))
            continue

        # 计算并累加hist矩阵
        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)

    # 计算最终的mIoU和mPA
    mIoUs = per_class_iu(hist)
    mPA = per_class_PA(hist)

    # 输出所有类别的平均mIoU和mPA
    print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 4)) +
          '; mPA: ' + str(round(np.nanmean(mPA) * 100, 4)))

    return mIoUs


if __name__ == "__main__":
    # 训练好的模型的路径

    score = 0
    ####计算class IoU分数####
    pred_dir = 'test_predictions/'
    base_dir = 'baseline_predictions/'
    gt_dir = 'test_ground_truths/'
    improvement_threshold = 0.06

    # 计算mIoU和mPA
    print("Baseline predictions")
    base_IoU = compute_mIoU(gt_dir, base_dir)
    print({f'base ious:'}, base_IoU)
    print("Ours predictions")
    pre_IoU = compute_mIoU(gt_dir, pred_dir)
    thr = 130

    for pre, base in zip(pre_IoU, base_IoU):
        delta = pre - base
        if delta >= improvement_threshold:
            score_class = 100
        else:
            if delta <= 0:
                score_class = 0
            else:
                score_class = 40 + (thr * delta) ** 2
        print(f"score：{score_class}")
        score += score_class
    print(f"Class IoU scores：{score}")



