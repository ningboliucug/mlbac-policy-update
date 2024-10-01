import pandas as pd

def clean_data(file_path, threshold_combinations):
    """
    清理 CSV 数据，基于多个 threshold_A 和 threshold_B 组合逐步执行删除操作。
    
    参数:
    - file_path: 要清理的 CSV 文件路径
    - threshold_combinations: 阈值组合的列表，形式为 [(A1, B1), (A2, B2), ...]
    """
    # 读取数据
    data = pd.read_csv(file_path)

    # 遍历每个 (threshold_A, threshold_B) 组合，逐步删除不符合条件的行
    for threshold_A, threshold_B in threshold_combinations:
        print(f"Applying threshold_A={threshold_A} and threshold_B={threshold_B}")
        # 删除 Mod_Sample_Accuracy 小于 threshold_A 且 Test_AUC 小于 threshold_B 的行
        data = data[~((data['Mod_Sample_Accuracy'] < threshold_A) & (data['Test_AUC'] < threshold_B))]

    # 存储有效的行
    valid_rows = []

    # 添加Retrainer和SISA模型的数据
    valid_rows.extend(data[data['Model'].isin(['Retrainer', 'SISA'])].to_dict(orient='records'))

    # 定义要处理的模型
    models_to_check = ['First_Order', 'BFRT', 'Fine_Tuning']

    # 遍历每个实验轮次
    for run in data['Run'].unique():
        run_data = data[data['Run'] == run]

        for model in models_to_check:
            model_data = run_data[run_data['Model'] == model]

            for index in range(len(model_data)):
                current_test_auc = model_data.iloc[index]['Test_AUC']
                invalid_data_found = False

                # 检查接下来是否有10行的 Test_AUC 均大于当前 Test_AUC
                if index + 10 < len(model_data):
                    next_10_rows = model_data.iloc[index+1:index+101]['Test_AUC']
                    if all(next_10_rows > current_test_auc):
                        invalid_data_found = True

                valid_rows.append(model_data.iloc[index].to_dict())

                # 如果找到了无效数据，跳出循环，忽略后续行
                if invalid_data_found:
                    break

    # 创建一个新的 DataFrame，包含有效行
    cleaned_data = pd.DataFrame(valid_rows)

    # 生成新的文件名
    new_file_path = file_path.replace('.csv', '_cleaned.csv')

    # 将清理后的数据保存
    cleaned_data.to_csv(new_file_path, index=False)
    print(f"数据清理完成，文件已保存到: {new_file_path}")

# 示例调用
if __name__ == "__main__":
    # 要清理的文件路径
    file_path = '/Users/liuningbo/Desktop/1_under_writing/5-MLforAC/experiment_DNN/second_part/performance_uci_mod_labels_only_DNN_80.csv'
    
    # 定义多个 threshold_A 和 threshold_B 的组合
    threshold_combinations = [
        (0.18, 0.75) # 第一次阈值组合
        #(0.1, 0.8)# 可以继续添加更多组合
    ]

    # 调用函数进行数据清理
    clean_data(file_path, threshold_combinations)
