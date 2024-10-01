import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from matplotlib.font_manager import FontProperties
import seaborn as sns

# Try to load the specified external font; fallback to default if not found
try:
    font_path = "/Users/liuningbo/Desktop/1_under_writing/2-RBAC-IPFS/FigureByPython/fonts/Calibri.ttf"  # Replace with your font path
    font_properties = FontProperties(fname=font_path, size=26)  # Adjust size as needed
    font_properties_bigger = FontProperties(fname=font_path, size=32)
except IOError:
    print("Font not found, using default.")
    font_properties = FontProperties(size=15)
sns.set_palette('colorblind')
colors_palette = sns.color_palette()

# 定义变量
data_type = "uci"
mod_type = "mod_labels_only"  # S1: mod_features_and_labels S2: mod_labels_only
model_type = "DNN"
sample_size = 80

# Read the CSV file
file_path = f'/Users/liuningbo/Desktop/1_under_writing/5-MLforAC/experiment_DNN/second_part/performance_{data_type}_{mod_type}_{model_type}_{sample_size}_cleaned.csv'
data = pd.read_csv(file_path)

# 定义一个函数，生成均匀分布的采样点，并基于附近概念计算每轮次的最大值
def aggregate_nearby_max_values_per_run(data, accuracy_column, auc_column, uniform_acc_values):
    """
    针对每个模型的 Run 列进行分组，采样每个均匀生成的横坐标的最大 Test_AUC 值。
    返回每轮次的数据，并确保横坐标统一。
    """
    result_per_run = []

    # 计算两个相邻刻度之间的差值（作为"附近"的阈值）
    threshold = (uniform_acc_values.max() - uniform_acc_values.min()) / len(uniform_acc_values)

    # 按照实验轮次 (Run) 分组
    for run in data['Run'].unique():
        run_data = data[data['Run'] == run]
        result = []

        # 遍历均匀生成的横坐标
        for acc in uniform_acc_values:
            # 找到差值绝对值小于当前阈值的数据行
            nearby_rows = run_data[
                (run_data[accuracy_column] >= acc - threshold) &
                (run_data[accuracy_column] <= acc + threshold)
            ]

            # 如果存在附近的数据行，计算Test_AUC的最大值
            if not nearby_rows.empty:
                max_auc = nearby_rows[auc_column].max()
                result.append(max_auc)
            else:
                result.append(np.nan)  # 如果没有数据，则标记为缺失值

        result_per_run.append(result)

    return np.array(result_per_run)  # 返回每轮次的采样结果 (每轮一行)

# Extract data for each model
models = ['BFRT', 'First_Order', 'Fine_Tuning']
model_data = {}

for model in models:
    model_specific_data = data[data['Model'] == model]
    
    # 获取Mod_Sample_Accuracy的最大值和最小值，生成900个固定的横坐标
    min_acc = model_specific_data['Mod_Sample_Accuracy'].min()
    max_acc = model_specific_data['Mod_Sample_Accuracy'].max()
    uniform_acc_values = np.linspace(min_acc, max_acc, 900)
    
    # 获取每轮次的最大值矩阵 (rows: run, columns: 900 points)
    result_per_run = aggregate_nearby_max_values_per_run(
        model_specific_data,
        accuracy_column='Mod_Sample_Accuracy',
        auc_column='Test_AUC',
        uniform_acc_values=uniform_acc_values
    )
    
    # 对每个采样点计算所有轮次的均值和标准差，忽略缺失值
    final_means = np.nanmean(result_per_run, axis=0)
    final_stds = np.nanstd(result_per_run, axis=0)
    
    # 舍弃那些所有轮次都缺失的数据点
    valid_indices = ~np.isnan(final_means)
    final_means = final_means[valid_indices]
    final_stds = final_stds[valid_indices]
    uniform_acc_values = uniform_acc_values[valid_indices]
    
    model_data[model] = {
        'Mod_Sample_Accuracy': uniform_acc_values,
        'Test_AUC_mean': final_means,
        'Test_AUC_std': final_stds
    }

# Define function to calculate MDO
def calculate_mdo(A_ACC_values, B_ACC_values):
    distances = np.sqrt((1 - np.array(A_ACC_values))**2 + (1 - np.array(B_ACC_values))**2)
    min_index = np.argmin(distances)
    return min_index, distances[min_index]

# Prepare to plot
plt.figure(figsize=(10, 8))

# Determine the common x-axis range
max_x = max(model_data[model]['Mod_Sample_Accuracy'].max() for model in models)
min_x = min(model_data[model]['Mod_Sample_Accuracy'].min() for model in models)

# For each model, plot smoothed curve and calculate MDO
colors = {'BFRT': colors_palette[0], 'First_Order': colors_palette[2], 'Fine_Tuning': colors_palette[3]}
legend_labels = {'BFRT': 'BFR', 'First_Order': 'First-Order', 'Fine_Tuning': 'Incremental-Learning'}

mdo_results = {}

for model in models:
    A_ACC_values = model_data[model]['Mod_Sample_Accuracy']
    B_ACC_mean_values = model_data[model]['Test_AUC_mean']
    B_ACC_std_values = model_data[model]['Test_AUC_std']
    
    # Use interpolation for smoothing (linear interpolation)
    interp_A_ACC = np.linspace(min_x, max_x, 500)  # Use common x range
    interp_mean = interp1d(A_ACC_values, B_ACC_mean_values, kind='linear', bounds_error=False, fill_value="extrapolate")
    interp_std = interp1d(A_ACC_values, B_ACC_std_values, kind='linear', bounds_error=False, fill_value="extrapolate")
    
    smooth_B_ACC_mean = interp_mean(interp_A_ACC)
    smooth_B_ACC_std = interp_std(interp_A_ACC)
    
    # Calculate range: smoothed mean ± smoothed std
    smooth_B_ACC_lower = smooth_B_ACC_mean - smooth_B_ACC_std
    smooth_B_ACC_upper = smooth_B_ACC_mean + smooth_B_ACC_std
    
    # Plot smoothed mean curve
    plt.plot(interp_A_ACC, smooth_B_ACC_mean, label=legend_labels[model], color=colors[model], linewidth=2.1)
    plt.fill_between(interp_A_ACC, smooth_B_ACC_lower, smooth_B_ACC_upper, color=colors[model], alpha=0.2)
    
    # Calculate MDO
    mdo_index, mdo_value = calculate_mdo(interp_A_ACC, smooth_B_ACC_mean)
    mdo_results[model] = (mdo_index, mdo_value)
    
    # Adjust MDO label position dynamically to avoid overlap
    # Offset the label by a small amount based on the value of the MDO point
    if interp_A_ACC[mdo_index] < 0.5:
        label_offset_x = 0.05  # Move to the right
        label_offset_y = -0.05  # Move downward
    else:
        label_offset_x = -0.05  # Move to the left
        label_offset_y = 0.05   # Move upward
    
    # Mark MDO point
    plt.scatter(interp_A_ACC[mdo_index], smooth_B_ACC_mean[mdo_index], color=colors[model], marker='o')
    plt.plot([interp_A_ACC[mdo_index], 1], [smooth_B_ACC_mean[mdo_index], 1], linestyle='--', color=colors[model])
    
    # Add MDO text with dynamic offset
    plt.text(interp_A_ACC[mdo_index] + label_offset_x, 
             smooth_B_ACC_mean[mdo_index] + label_offset_y, 
             f"MDO: {mdo_value:.3f}", 
             color=colors[model], ha='center', fontproperties=font_properties_bigger)

# Mark the point (1,1)
plt.scatter(1, 1, color='black', marker='x')

# Set labels with font properties
plt.xlabel('Accuracy on $Z_{policy}$', fontproperties=font_properties_bigger)
plt.ylabel('AUC on $Z_{test}$', fontproperties=font_properties_bigger)

# Set x and y ticks with font properties
plt.xticks(fontproperties=font_properties_bigger)
plt.yticks(fontproperties=font_properties_bigger)

# Legend with font properties
plt.legend(loc='upper left', bbox_to_anchor=(-0.02, 1.025), prop=font_properties)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

output_file_name = f"aft_{model_type}_{data_type}_{mod_type}_{sample_size}.png"
plt.savefig(f'/Users/liuningbo/Desktop/1_under_writing/5-MLforAC/menuscript_latex/实验图片/{output_file_name}', bbox_inches='tight', dpi=100)
plt.show()
