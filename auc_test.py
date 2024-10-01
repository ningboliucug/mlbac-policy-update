import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties

# 设置字体
font_path = "/Users/liuningbo/Desktop/1_under_writing/2-RBAC-IPFS/FigureByPython/fonts/Calibri.ttf"
font_properties = FontProperties(fname=font_path, size=28)

sns.set_palette('colorblind')
colors = sns.color_palette()

# 定义变量
data_type = "kaggle"
mod_type = "mod_labels_only"  # S1: mod_features_and_labels S2: mod_labels_only
model_type = "DNN"

# 样本数量对应的文件名
sample_sizes = [10, 20, 30, 50, 80, 120]
file_paths = [f'/Users/liuningbo/Desktop/1_under_writing/5-MLforAC/experiment_DNN/second_part/results/performance_{data_type}_{mod_type}_{model_type}_{size}.csv' for size in sample_sizes]

# 定义要分析的模型和样式
styles = {
    'BFRT': {'color': colors[0], 'marker': 'o', 'linestyle': '-', 'linewidth': 3, 'markersize': 13},
    'First_Order': {'color': colors[2], 'marker': '^', 'linestyle': '-', 'linewidth': 3, 'markersize': 13},
    'Fine_Tuning': {'color': colors[3], 'marker': 'v', 'linestyle': '-', 'linewidth': 3, 'markersize': 13},
    'SISA': {'color': colors[1], 'marker': '>', 'linestyle': '-', 'linewidth': 3, 'markersize': 13},
    'Retrainer': {'color': 'gray', 'marker': 'o', 'linestyle': '--', 'linewidth': 3, 'markersize': 13}
}

# 准备存储结果的字典
results = {model: {'Test_AUC': []} for model in styles}

# 遍历每个文件并提取数据
for file_path, sample_size in zip(file_paths, sample_sizes):
    # 读取CSV文件
    data = pd.read_csv(file_path)
    
    for model in styles:
        if model in ['Retrainer', 'SISA']:
            # 对于 Retrainer 和 SISA 模型，直接计算 Mod_Sample_Accuracy 的均值
            model_specific_data = data[data['Model'] == model]
            avg_mod_sample_acc = model_specific_data['Test_AUC'].mean() if not model_specific_data.empty else np.nan
            results[model]['Test_AUC'].append(avg_mod_sample_acc)
        else:
            # 过滤出其他模型的数据
            model_specific_data = data[(data['Model'] == model) & 
                                       (np.abs(data['Mod_Sample_Accuracy'] - 0.8) <= 0.01)]
            
            # 找到每一轮中的 Mod_Sample_Accuracy 最大值
            if not model_specific_data.empty:
                max_mod_sample_acc_per_run = model_specific_data.groupby('Run')['Test_AUC'].max()
                # 计算最大值的均值
                mean_max_mod_sample_acc = max_mod_sample_acc_per_run.mean()
                results[model]['Test_AUC'].append(mean_max_mod_sample_acc)
            else:
                results[model]['Test_AUC'].append(np.nan)

# 创建图形
plt.figure()

# 绘制 Mod_Sample_Accuracy 的折线图
for model, style in styles.items():
    plt.plot(sample_sizes, results[model]['Test_AUC'], label=model, **style)

# 设置标题和标签
plt.xlabel('Number of PUTs', fontproperties=font_properties)
plt.ylabel('AUC on $Z_{test}$', fontproperties=font_properties)

# 自定义横坐标，仅显示 10, 30, 50, 80, 120
custom_ticks = [10, 30, 50, 80, 120]
plt.xticks(custom_ticks, [str(size) for size in custom_ticks], fontproperties=font_properties)

# 获取当前的 Axes 对象
ax = plt.gca()
# 手动设置 y 轴的刻度值
yticks = np.linspace(start=0.5, stop=0.9, num=5)  # 设置 5 个刻度
ax.set_yticks(yticks)
# 设置刻度的字体
ax.set_yticklabels([f'{y:.1f}' for y in yticks], fontproperties=font_properties)

# 添加网格线
plt.grid(True, linestyle='--', alpha=0.5)

# 调整布局并显示图形
output_file_name = f"auc_{model_type}_{data_type}_{mod_type}.png"
plt.savefig(f'/Users/liuningbo/Desktop/1_under_writing/5-MLforAC/menuscript_latex/实验图片/{output_file_name}', bbox_inches='tight', dpi=100)
plt.show()
