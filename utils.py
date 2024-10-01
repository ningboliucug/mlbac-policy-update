from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, confusion_matrix
import numpy as np
import pandas as pd
import scipy.sparse
from scipy.special import rel_entr
from scipy.spatial.distance import jensenshannon
from copy import deepcopy
from scipy.sparse import vstack, csr_matrix
import time
from sklearn.model_selection import StratifiedKFold
from concurrent.futures import ProcessPoolExecutor
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import log_loss
from base import LogisticRegression
from collections import Counter

def compute_label_proportions(Y):
    # 计算每个标签的数量
    label_counts = Counter(Y)
    
    # 计算总标签数
    total_count = sum(label_counts.values())
    
    # 计算每个标签的比例
    label_proportions = {label: count / total_count for label, count in label_counts.items()}
    
    return label_proportions

def compute_gradient_sum(model, x, y, weights):
    if scipy.sparse.issparse(x):
        x = x.toarray()
    y = np.array(y)

    # 确保 x 是 2D
    if x.ndim == 1:
        x = x[np.newaxis, :]

    # Get predictions using predict_proba
    predictions = model.predict_proba(x)[:, 1]  # Probability of class 1

    # 根据 y 生成每个样本的权重
    sample_weights = np.array([weights[label] for label in y])

    # 加权梯度计算
    weighted_errors = (predictions - y) * sample_weights
    gradients_sum = np.dot(weighted_errors[:, np.newaxis].T, x)

    return gradients_sum

def compute_weights(Y):
    # 如果 Y 是 Pandas Series，则转换为 NumPy 数组
    if isinstance(Y, pd.Series):
        Y = Y.values

    class_counts = np.bincount(Y)
    total_samples = len(Y)
    num_classes = len(class_counts)
    
    # 计算每个类别的权重
    weights = total_samples / (num_classes * class_counts)
    
    return weights
# 用于调整x_sample_mod,y_sample_mod的比例
def adjust_class_distribution(X_mod, Y_mod, x_sample_mod, y_sample_mod):
    # 找出 Y_mod 中的主要类别及其比例
    majority_class = Y_mod.value_counts().idxmax()
    majority_proportion = Y_mod.value_counts(normalize=True).max()

    # 计算 y_sample_mod 中主要类别的当前比例
    majority_count_in_sample = np.sum(y_sample_mod == majority_class)
    majority_proportion_in_sample = majority_count_in_sample / len(y_sample_mod)

    # 创建 x_sample_mod 和 y_sample_mod 的副本以避免修改原始数据
    x_sample_mod_adjusted = x_sample_mod.copy()
    y_sample_mod_adjusted = np.copy(y_sample_mod)

    Y_mod_reset_index = Y_mod.reset_index(drop=True)

    # 计算需要添加的样本总数以达到目标比例
    total_samples_needed = int(len(y_sample_mod) / (1 - majority_proportion) - len(y_sample_mod))
    additional_majority_samples_count = max(0, int(total_samples_needed * majority_proportion - majority_count_in_sample))
    additional_minority_samples_count = total_samples_needed - additional_majority_samples_count

    # 添加主要类别的样本
    if additional_majority_samples_count > 0:
        additional_indices = np.random.choice(Y_mod_reset_index[Y_mod_reset_index == majority_class].index.values, additional_majority_samples_count, replace=False)
        x_additional = X_mod[additional_indices, :]
        y_additional = Y_mod_reset_index.iloc[additional_indices].values

        x_sample_mod_adjusted = vstack([x_sample_mod_adjusted, x_additional])
        y_sample_mod_adjusted = np.append(y_sample_mod_adjusted, y_additional)

    # 添加少数类别的样本
    if additional_minority_samples_count > 0:
        additional_indices = np.random.choice(Y_mod_reset_index[Y_mod_reset_index != majority_class].index.values, additional_minority_samples_count, replace=False)
        x_additional = X_mod[additional_indices, :]
        y_additional = Y_mod_reset_index.iloc[additional_indices].values

        x_sample_mod_adjusted = vstack([x_sample_mod_adjusted, x_additional])
        y_sample_mod_adjusted = np.append(y_sample_mod_adjusted, y_additional)
    #print_label_proportions(y_sample_mod_adjusted)
    return x_sample_mod_adjusted, y_sample_mod_adjusted

def balance_class_distribution(X_mod, Y_mod, x_sample_mod, y_sample_mod):
    # 确定 y_sample_mod 中的每个类别的数量
    class_counts = np.bincount(y_sample_mod)
    minority_class = np.argmin(class_counts)
    majority_class = np.argmax(class_counts)

    # 需要添加的少数类别样本数量
    additional_samples_count = class_counts[majority_class] - class_counts[minority_class]

    # 创建 x_sample_mod 和 y_sample_mod 的副本以避免修改原始数据
    x_sample_mod_adjusted = x_sample_mod.copy()
    y_sample_mod_adjusted = np.copy(y_sample_mod)

    # 确保 Y_mod 的索引与 X_mod 一致
    Y_mod_reset_index = Y_mod.reset_index(drop=True)

    # 添加少数类别的样本
    if additional_samples_count > 0:
        additional_indices = np.random.choice(Y_mod_reset_index[Y_mod_reset_index == minority_class].index.values, additional_samples_count, replace=False)
        x_additional = X_mod[additional_indices, :]
        y_additional = Y_mod_reset_index.iloc[additional_indices].values

        x_sample_mod_adjusted = vstack([x_sample_mod_adjusted, x_additional])
        y_sample_mod_adjusted = np.append(y_sample_mod_adjusted, y_additional)
    print_label_proportions(y_sample_mod_adjusted)
    return x_sample_mod_adjusted, y_sample_mod_adjusted

def balance_class_distribution_to_match_full_set(X_mod, Y_mod, x_sample_mod, y_sample_mod):
    # 计算目标和当前样本分布
    target_counts = np.bincount(Y_mod)
    target_proportions = target_counts / np.sum(target_counts)
    
    # 调整样本以匹配全集的分布
    current_counts = np.bincount(y_sample_mod)
    adjustments = target_proportions * len(y_sample_mod) - current_counts
    
    for class_label, adjustment in enumerate(adjustments):
        if adjustment > 0:  # 需要添加样本
            available_indices = np.where(Y_mod == class_label)[0]
            additional_indices = np.random.choice(available_indices, int(adjustment), replace=False)
            x_additional = X_mod[additional_indices]
            y_sample_mod = np.append(y_sample_mod, Y_mod[additional_indices])
            x_sample_mod = vstack([x_sample_mod, x_additional])
        elif adjustment < 0:  # 需要删除样本
            remove_indices = np.where(y_sample_mod == class_label)[0][:int(-adjustment)]
            # 为稀疏矩阵生成掩码，排除特定索引
            mask = np.ones(len(y_sample_mod), dtype=bool)
            mask[remove_indices] = False
            y_sample_mod = y_sample_mod[mask]
            x_sample_mod = x_sample_mod[mask]
    #print("FT:")
    #print_label_proportions(y_sample_mod)
    return x_sample_mod, y_sample_mod


def print_label_proportions(y_sample_mod):
    # Calculate the count of each class in y_sample_mod
    class_counts = np.bincount(y_sample_mod)
    total_samples = np.sum(class_counts)
    
    # Calculate the proportion of each class
    class_proportions = class_counts / total_samples
    
    # Print class proportions
    for class_label in range(len(class_proportions)):
        print(f"Proportion of class {class_label}: {class_proportions[class_label]:.2f}")

def sample_with_equal_labels_test(X_test, Y_test, k):
    if k % 2 != 0:
        raise ValueError("k must be an even number to ensure a 1:1 label ratio.")
    
    # 确保有足够的样本
    label_counts = Counter(Y_test)
    if label_counts[0] < k // 2 or label_counts[1] < k // 2:
        raise ValueError("Not enough samples to ensure a 1:1 label ratio.")
    
    # 从标签为 0 和标签为 1 的样本中分别选取 k/2 个样本
    class_0_indices = np.where(Y_test == 0)[0]
    class_1_indices = np.where(Y_test == 1)[0]

    selected_class_0_indices = np.random.choice(class_0_indices, k // 2, replace=False)
    selected_class_1_indices = np.random.choice(class_1_indices, k // 2, replace=False)

    selected_indices = np.concatenate([selected_class_0_indices, selected_class_1_indices])
    np.random.shuffle(selected_indices)  # 打乱顺序

    X_sampled = X_test[selected_indices]
    Y_sampled = Y_test[selected_indices]
    Y_sampled = np.array(Y_sampled)
    return X_sampled, Y_sampled

def sample_with_equal_labels(X_test_enc, Y_test, x_sample_mod_enc, y_sample_mod):
    # 计算 x_sample_mod_enc 中每个标签的数量
    labels, counts = np.unique(y_sample_mod, return_counts=True)
    label_counts = dict(zip(labels, counts))
    
    # 目标是确保合并后每个标签的数量相同
    target_count = max(counts)

    # 确定每个标签需要从 X_test_enc 中选取的样本数量
    samples_needed = {label: target_count - count for label, count in label_counts.items()}

    selected_indices = []

    for label, count in samples_needed.items():
        if count > 0:
            # 从 X_test_enc 中找到对应标签的样本索引
            label_indices = np.where(Y_test == label)[0]
            
            # 检查是否有足够的样本
            if len(label_indices) < count:
                raise ValueError(f"Not enough samples in X_test_enc for label {label}")
            
            # 从对应标签的样本中随机选取相应数量的样本
            random_indices = np.random.choice(label_indices, count, replace=False)
            selected_indices.extend(random_indices)

    # 从 X_test_enc 中选取这些样本
    X_test_sampled = X_test_enc[selected_indices, :]

    # 将选取的样本与 x_sample_mod_enc 垂直堆叠
    X_proba = vstack([X_test_sampled, x_sample_mod_enc])
    
    # 合并后的标签
    y_proba = np.concatenate([Y_test[selected_indices], y_sample_mod])

    return X_proba, y_proba


def _clip_gradient(gradient, clipping_threshold):
    norm = np.linalg.norm(gradient)
    print("norm")
    print(norm)
    if norm > clipping_threshold:
        return gradient * (clipping_threshold / norm)
    else:
        return gradient

def compute_gradient_sum_clipped(model, x, y, weights, clip_threshold):
    if scipy.sparse.issparse(x):
        x = x.toarray()
    y = np.array(y)

    # 确保 x 是 2D
    if x.ndim == 1:
        x = x[np.newaxis, :]

    # Get predictions using predict_proba
    predictions = model.predict_proba(x)[:, 1]  # Probability of class 1

    # 根据 y 生成每个样本的权重
    sample_weights = np.array([weights[label] for label in y])

    # 加权梯度计算
    gradients_sum = np.zeros_like(x[0])  # 初始化梯度和
    for i in range(len(y)):
        error = (predictions[i] - y[i]) * sample_weights[i]
        gradient = error * x[i]
        clipped_gradient = _clip_gradient(gradient, clip_threshold)
        gradients_sum += clipped_gradient

    return gradients_sum