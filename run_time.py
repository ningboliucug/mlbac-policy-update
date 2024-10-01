import numpy as np
import pandas as pd

# 定义变量
data_type = "kaggle"
mod_type = "mod_labels_only"  # S1: mod_features_and_labels S2: mod_labels_only
model_type = "LR"
sample_size = 120

# 读取CSV文件
file_path = f'/Users/liuningbo/Desktop/1_under_writing/5-MLforAC/experiment_DNN/second_part/performance_{data_type}_{mod_type}_{model_type}_{sample_size}.csv'
data = pd.read_csv(file_path)

# 定义模型列表
models = ['Retrainer', 'BFRT', 'First_Order', 'Fine_Tuning', 'SISA']

# 初始化结果字典
results = {}

for model in models:
    # 过滤出该模型的数据
    model_specific_data = data[data['Model'] == model]
    
    if model_specific_data.empty:
        print(f"No data found for {model}")
        continue
    
    # 计算运行时间的均值
    avg_run_time = model_specific_data['Run_Time'].mean()
    results[model] = avg_run_time

# 输出每个模型的平均运行时间
for model, avg_run_time in results.items():
    print(f"Model: {model}")
    print(f"  Average Run Time: {avg_run_time:.4f} seconds")
    print()
