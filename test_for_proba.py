from sklearn.model_selection import GridSearchCV
import pandas as pd
import data_modify as dm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from base import DeepNeuralNetwork, LogisticRegressionModel
from unlearners import First_Order, Fine_Tuning, BFRT, SISA_DNN, SISA_LR
import sys
from scipy.sparse import vstack, csr_matrix
import torch
from tqdm import tqdm
import concurrent.futures
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
from sklearn.model_selection import train_test_split
import scipy.sparse
from torch.utils.data import DataLoader, Dataset
import csv
import time
from sklearn.utils import resample

def stratified_sample(X_test, Y_test, X_sample_mod, Y_sample_mod, factor=50):
    # 计算第二个数据集中每个标签的数量
    unique_labels, label_counts = np.unique(Y_sample_mod.cpu().numpy(), return_counts=True)
    
    # 计算需要采样的数量
    total_samples = len(Y_sample_mod) * factor
    
    # 计算每个标签在采样后需要的数量
    label_ratios = label_counts / len(Y_sample_mod)
    target_label_counts = (label_ratios * total_samples).astype(int)
    
    # 确保总样本数符合要求
    if target_label_counts.sum() < total_samples:
        target_label_counts[0] += total_samples - target_label_counts.sum()
    
    # 进行分层采样
    X_sampled_list = []
    Y_sampled_list = []
    
    for label, count in zip(unique_labels, target_label_counts):
        X_label = X_test[Y_test == label]
        Y_label = Y_test[Y_test == label]
        X_sampled, Y_sampled = resample(X_label.cpu().numpy(), Y_label.cpu().numpy(), n_samples=count, random_state=42)
        X_sampled_list.append(torch.tensor(X_sampled, dtype=torch.float32).to(X_test.device))
        Y_sampled_list.append(torch.tensor(Y_sampled, dtype=torch.float32).to(Y_test.device))
    
    X_sampled_final = torch.cat(X_sampled_list, dim=0)
    Y_sampled_final = torch.cat(Y_sampled_list, dim=0)
    
    return X_sampled_final, Y_sampled_final

def calculate_class_weight(Y_train):
    #class_counts = np.bincount(Y_train.astype(int))
    class_counts = np.bincount(Y_train.cpu().numpy().astype(int))
    total_count = len(Y_train)
    weights = total_count / class_counts
    class_weight = weights[1] / (weights[0] + weights[1])  # Assuming binary classification with labels 0 and 1
    #return class_weight
    return torch.tensor([class_weight], dtype=torch.float32)

def balanced_sampling(X_mod, Y_mod, x_sample_mod, y_sample_mod):
    # 计算Y_mod和y_sample_mod中的标签比例
    y_mod_ratio = np.sum(Y_mod) / len(Y_mod)
    y_sample_mod_ratio = np.sum(y_sample_mod) / len(y_sample_mod)

    # 确定需要采样的类别和采样数量
    x_sample_mod_length = x_sample_mod.shape[0]

    if y_mod_ratio > y_sample_mod_ratio:
        target_class = 1
        num_samples_to_balance = int(x_sample_mod_length * (y_mod_ratio / (1 - y_mod_ratio)))
    else:
        target_class = 0
        num_samples_to_balance = int(x_sample_mod_length * ((1 - y_mod_ratio) / y_mod_ratio))

    # 从X_mod和Y_mod中按比例采样
    target_indices = np.where(Y_mod == target_class)[0]

    # 确保采样数量不超过可用数据数量
    num_samples_to_balance = min(len(target_indices), num_samples_to_balance)

    sampled_target_indices = np.random.choice(target_indices, size=num_samples_to_balance, replace=False)

    sampled_indices = sampled_target_indices

    # 创建采样后的数据集
    X_ft = X_mod[sampled_indices]
    Y_ft = Y_mod.iloc[sampled_indices].copy()

    # 将修改后的样本并入并移除重复项
    X_ft = scipy.sparse.vstack([X_ft, x_sample_mod])  # 合并稀疏矩阵
    Y_ft = pd.concat([Y_ft, y_sample_mod])

    # 确保 X_ft 和 Y_ft 的长度相同
    if X_ft.shape[0] != Y_ft.shape[0]:
        raise ValueError("The number of samples in X_ft and Y_ft must be the same.")

    return X_ft, Y_ft

def data_engineering(dataset_path, modification_type, sample_num):
    # 获取数据
    data = pd.read_csv(dataset_path)
    Y = data['ACTION']
    X = data.drop('ACTION', axis=1)

    # 分割数据集
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=40)
    X_train = X_train.reset_index(drop = True)
    Y_train = Y_train.reset_index(drop = True)
    X_test = X_test.reset_index(drop = True)
    Y_test = Y_test.reset_index(drop = True)

    # 选取样本修改数据集
    if modification_type == 'mod_features_and_labels':
        X_mod, Y_mod, x_sample_mod, y_sample_mod, x_sample, y_sample, indices_sample = dm.modify_features_and_labels(X_train, Y_train, n=sample_num, k=1, p=3)
        #X_mod, Y_mod, x_sample_mod, y_sample_mod, x_sample, y_sample, indices_sample = dm.modify_features_and_labels_balanced(X_train, Y_train, n=sample_num, k=1, p=3)
    elif modification_type == 'mod_features_only':
        X_mod, Y_mod, x_sample_mod, y_sample_mod, x_sample, y_sample, indices_sample = dm.modify_features(X_train, Y_train, n=sample_num, k=1, p=3)
    elif modification_type == 'mod_labels_only':
        X_mod, Y_mod, x_sample_mod, y_sample_mod, x_sample, y_sample, indices_sample = dm.modify_labels(X_train, Y_train, n=sample_num)
    else:
        raise ValueError("Invalid modification type")

    #print(type(x_sample_mod))

    # 独热编码
    one_hot_enc = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
    one_hot_encoder = one_hot_enc.fit(X)
    X_test_enc = one_hot_encoder.transform(X_test)
    X_train_enc = one_hot_encoder.transform(X_train)
    X_mod_enc = one_hot_encoder.transform(X_mod)
    x_sample_enc = one_hot_encoder.transform(x_sample)
    x_sample_mod_enc = one_hot_encoder.transform(x_sample_mod)
    
    # 生成用于测试各个模型输出概率的数据，在这里生成的主要目的是让测试数据统一
    # (1)确定从 X_test_enc 中选取的样本数量
    num_samples = int(x_sample_mod_enc.shape[0] * 1)
    # 从 X_test_enc 中随机选取相同数量的样本
    random_indices = np.random.choice(X_test_enc.shape[0], num_samples, replace=False)
    X_test_sampled = X_test_enc[random_indices, :]
    # 将选取的样本与 x_sample_mod_enc 垂直堆叠
    X_proba = vstack([X_test_sampled, x_sample_mod_enc])
    y_proba = np.concatenate([Y_test[random_indices], y_sample_mod])
    return X_train_enc, Y_train, X_test_enc, Y_test, X_mod_enc, Y_mod, x_sample_mod_enc, y_sample_mod, x_sample_enc, y_sample, indices_sample, X_proba, y_proba

def get_all_performance(models, retrainer, X_test, Y_test, X_sample_mod, Y_sample_mod, X_proba):
    performance_dict = {}
    
    for model_name, model in models.items():
        # 调用每个模型的 evaluate_model 方法
        performance = model.evaluate_model(retrainer, X_test, Y_test, X_sample_mod, Y_sample_mod, X_proba)
        performance_dict[model_name] = performance
    
    return performance_dict

def new_for_ppc(num_runs, num_points, sample_num, sisa_C, n_shards, dataset_path, modification_type, data_type, base_select, model='all'):
    result_file = "performance_" + data_type + "_" + modification_type + "_" + base_select + "_" + str(sample_num) + ".csv"
    print(result_file)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # 定义参数范围
    if base_select == 'LR':
        fo_tao_max = 25
        ft_C_max = 0.08          
        fr_max = 1
        rr_max = 0.5
    elif base_select == 'DNN':
        fo_tao_max = 3.5
        ft_C_max = 0.012
        fr_max = 0.04 
        rr_max = 0.014     
    fo_tao_values = np.linspace(0, fo_tao_max, num_points)
    ft_C_values = np.linspace(0, ft_C_max, num_points)
    forgetting_rate_values = np.linspace(0, fr_max, num_points)
    retuning_rate_values = np.linspace(0, rr_max, num_points)
    ppc_models = ['BFRT', 'First_Order', 'Fine_Tuning'] 

    # 初始化累积结果存储
    cumulative_results = pd.DataFrame(columns=["Model", "Test_AUC", "FPR", "Test_Accuracy", "Mod_Sample_Accuracy", "MIA AUC.", "JS Div.", "Parameter", "Run", "Run_Time"])

    # 外层循环，执行 num_runs 次
    #for run in range(num_runs):
    for run in tqdm(range(num_runs), desc="Total Running"):
        print(f"Run {run+1}/{num_runs}")
        run_results = pd.DataFrame(columns=["Model", "Test_AUC", "FPR", "Test_Accuracy", "Mod_Sample_Accuracy", "MIA AUC.", "JS Div.", "Parameter", "Run", "Run_Time"])
        
        # 数据工程
        X_train_enc, Y_train, X_test_enc, Y_test, X_mod_enc, Y_mod, x_sample_mod_enc, y_sample_mod, x_sample_enc, y_sample, indices_sample, X_proba, y_proba = data_engineering(dataset_path, modification_type, sample_num)
        # for SISA
        changed_data = {idx: (x_sample_mod_enc[i].toarray(), y_sample_mod.iloc[i]) for i, idx in enumerate(indices_sample)}
        sampling_start_time = time.time()
        X_rt2, Y_rt2 = balanced_sampling(X_mod_enc, Y_mod, x_sample_mod_enc, y_sample_mod)
        sampling_end_time = time.time()
        sampling_time = sampling_end_time - sampling_start_time

        X_train_enc = torch.tensor(X_train_enc.toarray(), dtype=torch.float32).to(device)
        Y_train = torch.tensor(Y_train.to_numpy(), dtype=torch.float32).to(device)
        X_test_enc = torch.tensor(X_test_enc.toarray(), dtype=torch.float32).to(device)
        Y_test = torch.tensor(Y_test.to_numpy(), dtype=torch.float32).to(device)
        X_mod_enc = torch.tensor(X_mod_enc.toarray(), dtype=torch.float32).to(device)
        Y_mod = torch.tensor(Y_mod.to_numpy(), dtype=torch.float32).to(device)
        x_sample_mod_enc = torch.tensor(x_sample_mod_enc.toarray(), dtype=torch.float32).to(device)
        y_sample_mod = torch.tensor(y_sample_mod.to_numpy(), dtype=torch.float32).to(device)
        x_sample_enc = torch.tensor(x_sample_enc.toarray(), dtype=torch.float32).to(device)
        y_sample = torch.tensor(y_sample.to_numpy(), dtype=torch.float32).to(device)
        X_proba = torch.tensor(X_proba.toarray(), dtype=torch.float32).to(device)
        X_rt2_t = torch.tensor(X_rt2.toarray(), dtype=torch.float32)
        Y_rt2_t = torch.tensor(Y_rt2.values, dtype=torch.float32)

        # 训练基础模型
        static_models = {}
        dynamic_models = {}
        runtime = {}
        input_dim = X_train_enc.shape[1]
        #print("Training base model.....")
        if base_select == 'LR':
            base_model = LogisticRegressionModel(input_dim).to(device)
            base_model.train_model(X_train_enc, Y_train)
        elif base_select == 'DNN':
            base_model = DeepNeuralNetwork(input_dim).to(device)
            base_model.train_model(X_train_enc, Y_train)
        else:
            raise ValueError(f"Unsupported base model type: {base_select}")

        # 重训练模型
        #print("Training retrainer model.....")
        if base_select == 'LR':
            start_time = time.time()  # 开始计时
            retrainer = LogisticRegressionModel(input_dim).to(device)       
            retrainer.train_model(X_mod_enc, Y_mod)
            end_time = time.time()  # 结束计时
            runtime['Retrainer'] = end_time - start_time
        elif base_select == 'DNN': 
            start_time = time.time()  # 开始计时
            retrainer = DeepNeuralNetwork(input_dim).to(device)       
            retrainer.train_model(X_mod_enc, Y_mod)
            end_time = time.time()  # 结束计时
            runtime['Retrainer'] = end_time - start_time
        static_models['Retrainer'] = retrainer

        if model == 'all' or model == 'SISA':
        #if model == 'SISA':
            #print("Training SISA model.....")
            if base_select == "LR":
                start_time = time.time()  # 开始计时
                sisa = SISA_LR(n_shards, input_dim)
                sisa.training(X_train_enc, Y_train)
                sisa.retrain_affected_shards(X_train_enc, Y_train, changed_data)
                end_time = time.time()  # 结束计时
                runtime['SISA'] = end_time - start_time
                static_models['SISA'] = sisa
            elif base_select == "DNN":
                start_time = time.time()  # 开始计时
                input_dim = X_train_enc.shape[1]
                sisa_dnn = SISA_DNN(n_shards, input_dim)
                sisa_dnn.training(X_train_enc, Y_train)
                sisa_dnn.retrain_affected_shards(X_train_enc, Y_train, changed_data)
                end_time = time.time()  # 结束计时
                runtime['SISA'] = end_time - start_time
                static_models['SISA'] = sisa_dnn

        # 计算并保存静态模型的性能
        static_performance = get_all_performance(static_models, retrainer, X_test_enc, Y_test, x_sample_mod_enc, y_sample_mod, X_proba)

        for static_model in static_models.keys():
            new_row = pd.DataFrame({
                "Model": [static_model],
                "Test_AUC": [static_performance[static_model]['AUC']],
                "FPR": [static_performance[static_model]['FPR']],
                "Test_Accuracy": [static_performance[static_model]['Test Accuracy']],
                "Mod_Sample_Accuracy": [static_performance[static_model]['Sample Accuracy']],
                "MIA AUC.": [static_performance[static_model]['MIA_AUC']],
                "JS Div.": [static_performance[static_model]['JS Div.']],
                "Parameter": [None],
                "Run": [run + 1],
                "Run_Time": [runtime.get(static_model, None)]  # 添加运行时间
            }).dropna(axis=1, how='all')
            run_results = pd.concat([run_results, new_row], ignore_index=True)
        

        weights = calculate_class_weight(Y_mod)

        for i in tqdm(range(num_points), desc="Running"):
            fo_tao = fo_tao_values[i]
            ft_C = ft_C_values[i]
            forgetting_rate = forgetting_rate_values[i]
            retuning_rate = retuning_rate_values[i]
            
            if model == 'all' or model == 'First_Order':
                #print("Training First Order model.....")
                start_time = time.time()  # 开始计时
                fo = First_Order(base_model, base_select, fo_tao, weights)
                fo_model = fo.training(X_mod_enc, Y_mod, x_sample_enc, y_sample, x_sample_mod_enc, y_sample_mod)
                end_time = time.time()  # 结束计时
                runtime['First_Order'] = end_time - start_time
                dynamic_models['First_Order'] = fo_model

            if model == 'all' or model == 'Fine_Tuning':
                #print("Training Fine Tuning model.....")
                start_time = time.time()  # 开始计时
                ft = Fine_Tuning(base_model, base_select, ft_C, weights)
                ft_model = ft.training(X_mod_enc, Y_mod, x_sample_enc, y_sample, x_sample_mod_enc, y_sample_mod)
                end_time = time.time()  # 结束计时
                runtime['Fine_Tuning'] = end_time - start_time
                dynamic_models['Fine_Tuning'] = ft_model

            if model == 'all' or model == 'BFRT':  
                #print("Training BFRT model.....")
                start_time = time.time()  # 开始计时 
                bfrt = BFRT(base_model, base_select, forgetting_rate, retuning_rate, weights)
                bfrt_model = bfrt.training(X_mod_enc, Y_mod, x_sample_enc, y_sample, x_sample_mod_enc, y_sample_mod, X_rt2_t, Y_rt2_t)
                end_time = time.time()  # 结束计时
                runtime['BFRT'] = end_time - start_time + sampling_time
                dynamic_models['BFRT'] = bfrt_model

            # 获取所有动态模型的性能
            start_time = time.time()  # 开始计时 
            performance = get_all_performance(dynamic_models, retrainer, X_test_enc, Y_test, x_sample_mod_enc, y_sample_mod, X_proba)
            end_time = time.time()  # 结束计时
            #print(end_time - start_time)
            # 遍历三个动态模型，提取结果并添加到 DataFrame 中
            for ppc_model in ppc_models:
                if ppc_model in performance:
                    new_row = pd.DataFrame({
                        "Model": [ppc_model],
                        "Test_AUC": [performance[ppc_model]['AUC']],
                        "FPR": [performance[ppc_model]['FPR']],
                        "Test_Accuracy": [performance[ppc_model]['Test Accuracy']],
                        "Mod_Sample_Accuracy": [performance[ppc_model]['Sample Accuracy']],
                        "MIA AUC.": [performance[ppc_model]['MIA_AUC']],
                        "JS Div.": [performance[ppc_model]['JS Div.']],
                        "Parameter": [fo_tao if ppc_model == "First_Order" else ft_C if ppc_model == "Fine_Tuning" else forgetting_rate],
                        "Run": [run + 1],
                        "Run_Time": [runtime.get(ppc_model, None)]  # 添加运行时间
                    }).dropna(axis=1, how='all')
                    run_results = pd.concat([run_results, new_row], ignore_index=True)
                else:
                    print(f"Warning: No performance data for {ppc_model} model")

        # 将每次的结果累积到 cumulative_results
        cumulative_results = pd.concat([cumulative_results, run_results], ignore_index=True)

    # 保存所有累积结果至 CSV 文件
    cumulative_results.to_csv(result_file, index=False)
    print(f"All results saved to {result_file}")


def test_model_performance(data_engineering_fn, dataset_path, modification_type, sample_num):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # 调用数据工程函数获取数据
    X_train_enc, Y_train, X_test_enc, Y_test, X_mod_enc, Y_mod, x_sample_mod_enc, y_sample_mod, x_sample_enc, y_sample, indices_sample, X_proba, y_proba = data_engineering_fn(dataset_path, modification_type, sample_num)

    X_train_enc = torch.tensor(X_train_enc.toarray(), dtype=torch.float32).to(device)
    Y_train = torch.tensor(Y_train.to_numpy(), dtype=torch.float32).to(device)
    X_test_enc = torch.tensor(X_test_enc.toarray(), dtype=torch.float32).to(device)
    Y_test = torch.tensor(Y_test.to_numpy(), dtype=torch.float32).to(device)
    X_mod_enc = torch.tensor(X_mod_enc.toarray(), dtype=torch.float32).to(device)
    Y_mod = torch.tensor(Y_mod.to_numpy(), dtype=torch.float32).to(device)
    x_sample_mod_enc = torch.tensor(x_sample_mod_enc.toarray(), dtype=torch.float32).to(device)
    y_sample_mod = torch.tensor(y_sample_mod.to_numpy(), dtype=torch.float32).to(device)
    X_proba = torch.tensor(X_proba.toarray(), dtype=torch.float32).to(device)
    #X_mia, _ = stratified_sample(X_test_enc, Y_test, x_sample_mod_enc, y_sample_mod, factor=50)

    # 初始化并训练初始模型
    input_dim = X_train_enc.shape[1]
    model = DeepNeuralNetwork(input_dim)
    model.train_model(X_train_enc, Y_train)

    # 使用 X_test_enc 和 Y_test 进行性能评估
    retrained_model = DeepNeuralNetwork(input_dim)
    retrained_model.train_model(X_mod_enc, Y_mod)
    
    evaluation_results = model.evaluate_model(
        retrained_model,
        X_test_enc, Y_test,
        x_sample_mod_enc, y_sample_mod,
        X_proba
    )

    # 输出评估结果
    print("Model Evaluation Results:")
    for key, value in evaluation_results.items():
        print(f"{key}: {value}")
    
    return evaluation_results




# 全局参数设置
uci_realworld = ('/Users/liuningbo/Desktop/1_under_writing/5-MLforAC/dataSet/amzn-uci-anon-access-samples/cleaned_uci-2.0.csv', 'uci')
kaggle_realworld = ('/Users/liuningbo/Desktop/1_under_writing/5-MLforAC/dataSet/amazon-employee-access-challenge/train.csv', 'kaggle')
dataset = kaggle_realworld
dataset_path = dataset[0]
data_type = dataset[1]

modification_type = 'mod_features_and_labels'
#modification_type = 'mod_labels_only'
#modification_type = 'mod_features_only'
num_runs = 1
num_points = 1000
sample_num = 50
base_select = 'LR'
n_shards = 3
sisa_C = 0.6

new_for_ppc(num_runs, num_points, sample_num, sisa_C, n_shards, dataset_path, modification_type, data_type, base_select, "all")
#sample_num_list = [10, 20, 30, 50, 80, 120]
#for sample_num in sample_num_list:
#    print(sample_num)
#   new_for_ppc(num_runs, num_points, sample_num, sisa_C, n_shards, dataset_path, modification_type, data_type, base_select, "all")
