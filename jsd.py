import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties

# 定义变量
data_type = "uci"
mod_type = "mod_labels_only" # S1: mod_features_and_labels S2: mod_labels_only
model_type = "LR"

# 样本数量对应的文件名
sample_sizes = [10, 20, 30]
file_paths = [f'/Users/liuningbo/Desktop/1_under_writing/5-MLforAC/experiment_DNN/second_part/results/performance_{data_type}_{mod_type}_{model_type}_{size}.csv' for size in sample_sizes]

# 定义要分析的模型（不包括 Retrainer）
models = ['SISA', 'BFRT', 'Fine_Tuning', 'First_Order']

# 初始化字典用于存储 JS Div. 结果
results = {model: {'JS Div.': []} for model in models}

# 遍历每个文件并提取数据
for file_path, sample_size in zip(file_paths, sample_sizes):
    data = pd.read_csv(file_path)
    
    for model in models:
        # 过滤出该模型的数据
        model_specific_data = data[data['Model'] == model]
        
        # 如果是 BFRT、First_Order、Fine_Tuning，需要根据每一轮获取数据
        if model in ['BFRT', 'First_Order', 'Fine_Tuning']:
            run_groups = model_specific_data.groupby('Run')
            best_js_div_values = []
            
            for run, run_data in run_groups:
                if run_data.empty:
                    continue
                
                # 找到 Mod_Sample_Accuracy 接近 target_acc 的数据
                target_acc = 0.8  # 根据需要调整
                filtered_data = run_data[np.abs(run_data['Mod_Sample_Accuracy'] - target_acc) <= 0.01]
                
                if filtered_data.empty:
                    continue
                
                # 找到 Test_AUC 的最大值对应的行
                max_test_auc = filtered_data['Test_AUC'].max()
                best_data = filtered_data[filtered_data['Test_AUC'] == max_test_auc]
                
                # 计算 JS Div.
                avg_js_div = best_data['JS Div.'].mean()
                best_js_div_values.append(avg_js_div)
            
            # 存储每个 sample_size 下的均值
            js_div_mean = np.nanmean(best_js_div_values) if best_js_div_values else np.nan
            results[model]['JS Div.'].append(js_div_mean)
        else:
            # 对于 SISA，直接计算 JS Div. 的均值
            avg_js_div = model_specific_data['JS Div.'].mean() if not model_specific_data.empty else np.nan
            results[model]['JS Div.'].append(avg_js_div)

# 设置字体
font_path = "/Users/liuningbo/Desktop/1_under_writing/2-RBAC-IPFS/FigureByPython/fonts/Calibri.ttf"
font_properties = FontProperties(fname=font_path, size=28)

# 设置调色板
sns.set_palette('colorblind')
colors = sns.color_palette()

# 模型颜色映射
model_colors = {
    'BFRT': colors[0],
    'SISA': colors[1],
    'First_Order': colors[2],
    'Fine_Tuning': colors[3],
}

# 准备绘图数据
data_list = []
for i, sample_size in enumerate(sample_sizes):
    for model in models:
        js_div_value = results[model]['JS Div.'][i]
        if not np.isnan(js_div_value):  # 确保值有效
            data_list.append({
                'Sample Size': sample_size,
                'Model': model,
                'JS Div.': js_div_value
            })

plot_df = pd.DataFrame(data_list)

# 创建图表
plt.figure()
ax = sns.barplot(
    data=plot_df, 
    x='Sample Size', 
    y='JS Div.', 
    hue='Model', 
    palette=model_colors,  # 使用指定的颜色映射
    edgecolor='black', 
    linewidth=2
)

# 设置坐标轴标签和标题
ax.set_xlabel('Number of PUTs', fontproperties=font_properties)
ax.set_ylabel('JSD', fontproperties=font_properties)

# 设置横纵坐标刻度字体
ax.set_xticklabels(ax.get_xticklabels(), fontproperties=font_properties)

# 将纵坐标固定为 0, 0.5, 1.0
ax.set_yticks([0, 0.4, 0.8])
ax.set_yticklabels([f"{y:.1f}" for y in [0, 0.4, 0.8]], fontproperties=font_properties)



# 设置图例
#plt.legend(prop=font_properties)
ax.get_legend().remove()
# 添加网格
plt.grid(True, linestyle='--', alpha=0.5)

# 调整布局并显示图形
plt.tight_layout()
output_file_name = f"jsd_{model_type}_{data_type}_{mod_type}.png"
plt.savefig(f'/Users/liuningbo/Desktop/1_under_writing/5-MLforAC/menuscript_latex/实验图片/{output_file_name}', bbox_inches='tight', dpi=100)
plt.show()
