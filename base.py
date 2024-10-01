import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support, confusion_matrix
from scipy.spatial.distance import jensenshannon
from numpy import vstack
import random
from sklearn.metrics import roc_curve
import torch.optim as optim
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

class CustomDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# 定义逻辑回归模型
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        # 将线性层和 Sigmoid 封装在 nn.Sequential 中
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1),
            #nn.Sigmoid()  # 手动添加 Sigmoid 函数，类似于原始代码
        )

    def forward(self, x):
        return self.model(x)  # 使用 self.model 进行前向传播
    
    def calculate_class_weight(self, Y_train):
        # 确保 Y_train 是在 CPU 上进行 bincount，因为 GPU 不支持 torch.bincount
        class_counts = torch.bincount(Y_train.to(torch.int64).cpu())
        total_count = len(Y_train)
        weights = total_count / class_counts
        class_weight = weights[1] / (weights[0] + weights[1])  # 假设二元分类，标签为 0 和 1
        return class_weight.to(Y_train.device)  # 将权重放回原始设备（GPU 或 CPU）
    
    def train_model(self, X, Y, epochs=5, lr=0.012, batch_size=128, weight_decay=1e-6): #kaggle # S1: lr=0.005; S2: lr=0.012
    #def train_model(self, X, Y, epochs=5, lr=0.009, batch_size=128, weight_decay=1e-7): #uci
        device = X.device  # 获取数据所在的设备
        self.to(device)  # 确保模型也在同一个设备上
        # 计算类别权重
        class_weight = self.calculate_class_weight(Y).to(device)

        # 定义损失函数和优化器
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([class_weight], device=device))
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        
        # 创建 DataLoader
        train_dataset = CustomDataset(X.cpu().numpy(), Y.cpu().numpy())
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            self.train()  # 模型设置为训练模式
            train_loss = 0
            
            for X_batch, Y_batch in train_loader:
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
                optimizer.zero_grad()
                outputs = self(X_batch).squeeze()
                loss = criterion(outputs, Y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            #avg_train_loss = train_loss / len(train_loader)
            #print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_train_loss:.4f}')
        
        return self
    
    def evaluate_model(self, retrainer, X_test, Y_test, X_sample_mod, Y_sample_mod, X_proba):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        retrainer.model.to(device)
        
        self.model.eval()
        retrainer.model.eval()
        
        with torch.no_grad():
            test_outputs = self(X_test).squeeze()
            test_probas = torch.sigmoid(test_outputs)

            # 计算 ROC 曲线以确定最佳阈值
            test_probas_cpu = test_probas.cpu().numpy()
            Y_test_cpu = Y_test.cpu().numpy()
            fpr, tpr, thresholds = roc_curve(Y_test_cpu, test_probas_cpu)
            youden_index = tpr - fpr
            best_threshold_index_youden = np.argmax(youden_index)
            best_threshold_youden = thresholds[best_threshold_index_youden]

            # 使用最佳阈值进行预测
            test_predictions = (test_probas > best_threshold_youden).cpu().numpy().astype(int)
            
            auc = roc_auc_score(Y_test_cpu, test_probas_cpu)
            test_accuracy = accuracy_score(Y_test_cpu, test_predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(Y_test_cpu, test_predictions, average='binary')
            
            tn, fp, fn, tp = confusion_matrix(Y_test_cpu, test_predictions).ravel()
            fpr_final = fp / (fp + tn)
            tpr_final = tp / (tp + fn)
            
            # 计算 (X_sample_mod, Y_sample_mod) 上的 Sample Accuracy
            sample_mod_outputs = self(X_sample_mod).squeeze()
            sample_mod_probas = torch.sigmoid(sample_mod_outputs)
            sample_mod_predictions = (sample_mod_probas > best_threshold_youden).cpu().numpy().astype(int)
            sample_accuracy = accuracy_score(Y_sample_mod.cpu().numpy(), sample_mod_predictions)
            
            # 计算 X_proba 上的概率
            X_proba_tensor = X_proba.to(device)
            proba_outputs = torch.sigmoid(self(X_proba_tensor)).cpu().numpy()
            
            # 计算 JS 散度
            js_div = self.calculate_js_divergence_combined(retrainer, X_test, X_sample_mod)
            
            # 计算 MIA 攻击的 AUC
            mia_accuracy, mia_roc_auc = self.mia_attack(self, test_probas_cpu.flatten(), sample_mod_probas.cpu().numpy().flatten())
            
            return {
                'AUC': auc,
                'Test Accuracy': test_accuracy,
                'FPR': fpr_final,
                'TPR': tpr_final,
                'F1': f1,
                'Sample Accuracy': sample_accuracy,
                'JS Div.': js_div,
                'MIA_AUC': mia_roc_auc
            }
    
    def calculate_js_divergence_combined(self, retrainer, X, X_sample):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        retrainer.model.to(device)
        
        self.model.eval()
        retrainer.model.eval()
        
        X_combined_tensor = torch.cat([X, X_sample], dim=0).to(device)
        
        with torch.no_grad():
            retrainer_probas = torch.sigmoid(retrainer(X_combined_tensor)).cpu().numpy().flatten()
            unlearner_probas = torch.sigmoid(self(X_combined_tensor)).cpu().numpy().flatten()
            
            js_div = jensenshannon(retrainer_probas, unlearner_probas, base=2)
            
        return js_div

    def mia_attack(self, target_model, train_probs, test_probs):
        # 构建攻击数据集
        mia_X = np.concatenate((train_probs, test_probs)).reshape(-1, 1)
        mia_y = np.concatenate((np.ones_like(train_probs), np.zeros_like(test_probs)))

        # 训练攻击模型
        attack_model = SklearnLogisticRegression(solver='lbfgs', class_weight='balanced')
        attack_model.fit(mia_X, mia_y)

        # 评估攻击模型
        mia_preds = attack_model.predict(mia_X)
        accuracy = accuracy_score(mia_y, mia_preds)
        roc_auc = roc_auc_score(mia_y, attack_model.predict_proba(mia_X)[:, 1])

        return accuracy, roc_auc

class DeepNeuralNetwork(nn.Module):
    def __init__(self, input_dim):
        super(DeepNeuralNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        return self.model(x)
    
    def calculate_class_weight(self, Y_train):
        # 确保 Y_train 是在 CPU 上进行 bincount，因为 GPU 不支持 torch.bincount
        class_counts = torch.bincount(Y_train.to(torch.int64).cpu())
        total_count = len(Y_train)
        weights = total_count / class_counts
        class_weight = weights[1] / (weights[0] + weights[1])  # 假设二元分类，标签为 0 和 1
        
        return class_weight.to(Y_train.device)  # 将权重放回原始设备（GPU 或 CPU）
    
    #def train_model(self, X, Y, epochs=5, lr=0.001, batch_size=128, weight_decay=1e-4, patience=10): # kaggle
    def train_model(self, X, Y, epochs=5, lr=0.001, batch_size=128, weight_decay=1e-6, patience=10): # UCI
        device = X.device  # 获取数据所在的设备
        self.model.to(device)  # 确保模型也在同一个设备上
        
        # Calculate class weights
        class_weight = self.calculate_class_weight(Y).to(device)
        
        # Define loss function and optimizer
        criterion = nn.BCEWithLogitsLoss(pos_weight=class_weight)
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Create DataLoader for training data (注意这里 X 和 Y 已经是张量)
        train_dataset = CustomDataset(X.cpu().numpy(), Y.cpu().numpy())  # 使用 CPU 将张量转换回 NumPy
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            
            for X_batch, Y_batch in train_loader:
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)  # Move data to GPU (already on GPU)
                optimizer.zero_grad()
                outputs = self.model(X_batch).squeeze()
                loss = criterion(outputs, Y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            #print(f"Epoch {epoch + 1}, Loss: {avg_train_loss:.4f}")
            
            if avg_train_loss < 0.03:
                break
        
        return self
    
    def evaluate_model(self, retrainer, X_test, Y_test, X_sample_mod, Y_sample_mod, X_proba):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        retrainer.model.to(device)
        
        self.model.eval()
        retrainer.model.eval()
        
        with torch.no_grad():            
            test_outputs = self.model(X_test).squeeze()
            test_probas = torch.sigmoid(test_outputs)
            
            # 计算ROC曲线以确定最佳阈值
            test_probas_cpu = test_probas.cpu().numpy()
            Y_test_cpu = Y_test.cpu().numpy()
            fpr, tpr, thresholds = roc_curve(Y_test_cpu, test_probas_cpu)
            youden_index = tpr - fpr
            best_threshold_index_youden = np.argmax(youden_index)
            best_threshold_youden = thresholds[best_threshold_index_youden]

            # 使用最佳阈值进行预测
            test_predictions = (test_probas > best_threshold_youden).cpu().numpy().astype(int)
            
            auc = roc_auc_score(Y_test_cpu, test_probas_cpu)
            test_accuracy = accuracy_score(Y_test_cpu, test_predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(Y_test_cpu, test_predictions, average='binary')
            
            tn, fp, fn, tp = confusion_matrix(Y_test_cpu, test_predictions).ravel()
            fpr_final = fp / (fp + tn)
            tpr_final = tp / (tp + fn)
            
            # 计算 (X_sample_mod, Y_sample_mod) 上的 Sample Accuracy
            
            sample_mod_outputs = self.model(X_sample_mod).squeeze()
            sample_mod_probas = torch.sigmoid(sample_mod_outputs)
            sample_mod_predictions = (sample_mod_probas > best_threshold_youden).cpu().numpy().astype(int)
            sample_accuracy = accuracy_score(Y_sample_mod.cpu().numpy(), sample_mod_predictions)
            
            # 计算 X_proba 上的概率
            X_proba_tensor = X_proba.to(device)
            proba_outputs = torch.sigmoid(self.model(X_proba_tensor)).cpu().numpy()
            
            # 计算 JS Divergence
            js_div = self.calculate_js_divergence_combined(retrainer, X_test, X_sample_mod)
            
            # 计算 MIA 攻击的 AUC
            #X_mia, _ = self.stratified_sample(X_test, Y_test, X_sample_mod, Y_sample_mod, factor=50)
            #train_probs = torch.sigmoid(self.model(X_mia)).cpu().numpy().flatten()
            mia_accuracy, mia_roc_auc = self.mia_attack(self.model, test_probas_cpu.flatten(), sample_mod_probas.cpu().numpy().flatten())
            
            return {
                'AUC': auc,
                'Test Accuracy': test_accuracy,
                'FPR': fpr_final,
                'TPR': tpr_final,
                'F1': f1,
                'Sample Accuracy': sample_accuracy,
                #'Probability': proba_outputs,
                'JS Div.': js_div,
                'MIA_AUC': mia_roc_auc
            }

    def calculate_js_divergence_combined(self, retrainer, X, X_sample):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        retrainer.model.to(device)
        
        self.model.eval()
        retrainer.model.eval()
        
        X_combined_tensor = torch.cat([X, X_sample], dim=0).to(device)
        
        with torch.no_grad():
            retrainer_probas = torch.sigmoid(retrainer.model(X_combined_tensor)).cpu().numpy().flatten()
            unlearner_probas = torch.sigmoid(self.model(X_combined_tensor)).cpu().numpy().flatten()
            
            js_div = jensenshannon(retrainer_probas, unlearner_probas, base=2)
            
        return js_div

    def mia_attack(self, target_model, train_probs, test_probs):
        # 构建攻击数据集
        mia_X = np.concatenate((train_probs, test_probs)).reshape(-1, 1)
        mia_y = np.concatenate((np.ones_like(train_probs), np.zeros_like(test_probs)))

        # 训练攻击模型
        attack_model = SklearnLogisticRegression(solver='lbfgs', class_weight='balanced')
        attack_model.fit(mia_X, mia_y)

        # 评估攻击模型
        mia_preds = attack_model.predict(mia_X)
        accuracy = accuracy_score(mia_y, mia_preds)
        roc_auc = roc_auc_score(mia_y, attack_model.predict_proba(mia_X)[:, 1])

        return accuracy, roc_auc





def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
