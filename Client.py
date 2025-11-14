import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from utils import VAEDecoupler, FeatureDecoupling

import torch
import torch.nn as nn
import numpy as np


class EEGNet(nn.Module):

    def __init__(self, n_timepoints=384, n_channels=32, n_classes=2,
                 feature_dim=128, dropout_rate=0.5):

        super(EEGNet, self).__init__()

        # --- 固定内部超参数 (Hardcoded Parameters) ---
        # 这些参数定义了EEGNet的经典结构，原代码中未作为输入参数
        F1 = 8  # 初始时间卷积核数量
        D = 2  # 深度乘数 (Depth Multiplier)
        F2 = F1 * D  # 深度卷积输出通道数量 (F1 * D)
        kernel_length = 64  # 初始时间卷积核长度
        P1 = 4  # Depthwise Conv后的池化因子
        P2 = 8  # Separable Conv后的池化因子
        # ------------------------------------------------

        self.n_channels = n_channels
        self.n_timepoints = n_timepoints
        self.dropout_rate = dropout_rate
        self.feature_dim = feature_dim

        # --- Block 1: Temporal Convolution (时间滤波) ---
        # (1, C, T) -> (F1, C, T)
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=F1,
                      kernel_size=(1, kernel_length),
                      padding=(0, kernel_length // 2),
                      bias=False),
            nn.BatchNorm2d(F1)
        )

        # --- Block 2: Depthwise Convolution (空间滤波) ---
        # (F1, C, T) -> (F2, 1, T/P1)
        self.depthwise_conv = nn.Sequential(
            # 空间滤波: 卷积核(C, 1)跨越所有通道，groups=F1 实现深度卷积
            nn.Conv2d(in_channels=F1,
                      out_channels=F2,
                      kernel_size=(n_channels, 1),
                      groups=F1,  # 关键: Depthwise分组
                      bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            # 时间轴平均池化
            nn.AvgPool2d((1, P1)),
            nn.Dropout(dropout_rate)
        )

        # --- Block 3: Separable Convolution (深度可分离卷积) ---
        # (F2, 1, T/P1) -> (F2, 1, T/(P1*P2))
        self.separable_conv = nn.Sequential(
            # 3.1 Depthwise Temporal Conv (时间特征提炼)
            nn.Conv2d(in_channels=F2,
                      out_channels=F2,
                      kernel_size=(1, 16),
                      padding=(0, 8),
                      groups=F2,  # 关键: Depthwise分组
                      bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),

            # 3.2 Pointwise Conv (特征混合)
            nn.Conv2d(in_channels=F2,
                      out_channels=F2,
                      kernel_size=(1, 1),
                      bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),

            # 时间轴平均池化
            nn.AvgPool2d((1, P2)),
            nn.Dropout(dropout_rate)
        )

        # --- Classifier (分类器) ---
        self.flatten_size = self._get_flatten_size()

        # **原代码修复和兼容性处理**
        # 1. 恢复原代码的 feature_dim 中间层，以匹配原__init__中的参数
        # 2. 避免在 self.fc_features 后使用 ReLU，因为原代码在 extract_features 中使用了 ReLU
        self.fc_features = nn.Linear(self.flatten_size, feature_dim)

        # 恢复原代码的分类层
        self.classifier = nn.Linear(feature_dim, n_classes)

    # ------------------- 辅助函数 -------------------
    def _get_flatten_size(self):
        """计算展平后的特征大小"""

        # 固定内部超参数，与 __init__ 中保持一致
        F2 = 16
        P1 = 4
        P2 = 8

        # 假设输入形状: (1, 1, n_channels, n_timepoints)
        x = torch.zeros(1, 1, self.n_channels, self.n_timepoints)

        # Block 1: Temporal Conv (不改变 T 维度)

        # Block 2: Depthwise Conv + Pool (将 C 维从 n_channels 变为 1, T 维 / P1)
        # 最终通道数是 F2 (16), C=1, T 维度降采样
        # T_pool1 = ceil((T + 2*P - K)/S + 1) -> T/P1
        T_pool1 = int(np.floor(self.n_timepoints / P1))

        # Block 3: Separable Conv + Pool (不改变 C 维, T 维 / P2)
        # T_pool2 = T_pool1 / P2
        T_pool2 = int(np.floor(T_pool1 / P2))

        # 最终形状: (1, F2, 1, T_pool2)
        return F2 * 1 * T_pool2

    # ------------------- 前向传播 -------------------
    def extract_features(self, x):
        """
        特征提取部分
        """
        # 维度调整逻辑保持不变
        if x.dim() == 3:
            # (batch, C, T) -> (batch, 1, C, T)
            x = x.unsqueeze(1)
        elif x.dim() == 2:
            # (batch, features) -> reshape
            batch_size = x.size(0)
            x = x.view(batch_size, 1, self.n_channels, self.n_timepoints)

        # Block 1: Temporal Conv
        x = self.temporal_conv(x)

        # Block 2: Depthwise Conv + Pool
        x = self.depthwise_conv(x)

        # Block 3: Separable Conv + Pool
        x = self.separable_conv(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # **恢复原代码的 fc_features 和 ReLU 逻辑**
        # 原代码: features = torch.relu(self.fc_features(x))
        features = torch.relu(self.fc_features(x))
        return features

    def forward(self, x):
        """
        前向传播
        """
        features = self.extract_features(x)
        out = self.classifier(features)

        # 返回分类 logits 和特征
        return out, features

class ACRNN(nn.Module):
    # 保持 __init__ 的参数签名不变
    def __init__(self, n_timepoints=384, n_channels=32, n_classes=2, feature_dim=128):
        super(ACRNN, self).__init__()
        # 保持实例变量不变
        self.feature_dim = feature_dim
        self.n_channels = n_channels  # 32
        self.n_timepoints = n_timepoints  # 384

        # CNN layers 保持不变
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d((2, 2))
        self.dropout = nn.Dropout(0.25)

        # 核心修正: LSTM input size 使用固定值计算
        # n_channels=32, 经过两次 (2, 2) 池化后，高度 H_out = 32 / 4 = 8
        # C_out = 128 (conv2的输出通道数)
        # lstm_input_size = C_out * H_out = 128 * 8 = 1024
        # 虽然 n_channels 是参数，但根据固定参数的要求，这里写死为 1024
        LSTM_INPUT_SIZE = 128 * (32 // 4)  # 1024
        self.lstm = nn.LSTM(LSTM_INPUT_SIZE, 128, batch_first=True, bidirectional=True)

        # Feature extraction & Classifier 保持不变
        self.fc_features = nn.Linear(256, feature_dim)  # 256 = 128 * 2 (Bidirectional)
        self.classifier = nn.Linear(feature_dim, n_classes)

        # 预计算固定值（供 extract_features 使用，避免重复计算）
        # n_timepoints=384, 经过两次 (2, 2) 池化后，时间步 T_out = 384 / 4 = 96
        self.fixed_time_steps = 384 // 4  # 96
        # n_channels=32, 经过两次 (2, 2) 池化后，高度 H_out = 32 // 4 = 8
        self.fixed_height_out = 32 // 4  # 8

        # 验证：确保传入的参数与内部写死的参数兼容
        if n_channels != 32 or n_timepoints != 384:
            import warnings
            warnings.warn(
                "ACRNN is initialized with n_channels={} and n_timepoints={}, but the LSTM configuration is hardcoded based on the default values (32 and 384). The model structure may be incorrect for this input size.".format(
                    n_channels, n_timepoints))

    # 保持 extract_features 的签名不变
    def extract_features(self, x):
        # 1. 预处理输入维度
        if x.dim() == 3:
            # (batch, 32, 384) -> (batch, 1, 32, 384)
            x = x.unsqueeze(1)
        elif x.dim() == 2:
            batch_size = x.size(0)
            x = x.view(batch_size, 1, self.n_channels, self.n_timepoints)

        # 2. CNN 提取特征
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.dropout(x)
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.dropout(x)
        # 此时 x 的形状应为 (B, 128, 8, 96) (如果 n_channels=32, n_timepoints=384)

        # 3. 修正：为 LSTM 进行时序重塑 (使用写死的值)
        batch_size = x.size(0)

        # 检查重塑是否兼容写死的值
        # 如果 self.n_timepoints != 384, 那么 x.size(3) 不等于 self.fixed_time_steps，重塑会失败

        # 将时序维度 (T_out=96) 移动到第二位，作为 Sequence Length
        # (B, 128, 8, 96) -> (B, 96, 128, 8)
        x = x.permute(0, 3, 1, 2).contiguous()

        # 展平特征维度 (128 * 8 = 1024)
        # (B, 96, 128, 8) -> (B, 96, 1024) -> B, Seq_Len, Feature_Dim
        # 注意：这里我们**不使用** x.size(3) 的 runtime 值，而是使用硬编码的 96 来确保逻辑与 __init__ 保持一致，
        # 但在 view 操作中，我们依赖 -1 来自动计算特征维度 1024。
        # 保持代码的健壮性，使用 x.size(1) 获取当前的 Seq_Len
        time_steps = x.size(1)
        x = x.view(batch_size, time_steps, -1)

        # 4. LSTM 处理时序
        x, _ = self.lstm(x)

        # 5. 特征聚合 (取最后一个时间步的输出)
        features_raw = x[:, -1, :]

        # 6. 特征映射
        features = torch.relu(self.fc_features(features_raw))
        return features

    # 保持 forward 的签名和逻辑不变
    def forward(self, x):
        features = self.extract_features(x)
        out = self.classifier(features)
        return out, features

class DeepConvNet(nn.Module):

    def __init__(self, n_timepoints=384, n_channels=32, n_classes=2, feature_dim=128):
        # 严格保留传入参数和父类初始化
        super(DeepConvNet, self).__init__()
        self.feature_dim = feature_dim
        self.n_channels = n_channels
        self.n_timepoints = n_timepoints

        # --- 块 1: 初始卷积 (Temporal + Spatial Filtering) ---
        # 1. Temporal Convolution (k=(1, 25) to capture broad temporal features,
        #    padding adjusted to maintain size, stride=1)
        self.conv1_t = nn.Conv2d(1, 25, kernel_size=(1, 25), padding=(0, 12), bias=False)

        # 2. Spatial Convolution (k=(n_channels, 1) to combine electrodes/channels)
        self.conv2_s = nn.Conv2d(25, 25, kernel_size=(n_channels, 1), bias=False)

        self.bn1 = nn.BatchNorm2d(25)
        self.elu = nn.ELU()  # 替换ReLU，EEG任务常用
        self.pool1 = nn.MaxPool2d((1, 3))
        self.dropout1 = nn.Dropout(0.5)

        # --- 块 2: Time Convolution & Downsampling ---
        self.conv3 = nn.Conv2d(25, 50, kernel_size=(1, 10), padding=(0, 5), bias=False)
        self.bn2 = nn.BatchNorm2d(50)
        self.pool2 = nn.MaxPool2d((1, 3))
        self.dropout2 = nn.Dropout(0.5)

        # --- 块 3: Time Convolution & Downsampling ---
        self.conv4 = nn.Conv2d(50, 100, kernel_size=(1, 10), padding=(0, 5), bias=False)
        self.bn3 = nn.BatchNorm2d(100)
        self.pool3 = nn.MaxPool2d((1, 3))
        self.dropout3 = nn.Dropout(0.5)

        # --- 块 4: Final Time Convolution & Downsampling ---
        self.conv5 = nn.Conv2d(100, 200, kernel_size=(1, 10), padding=(0, 5), bias=False)
        self.bn4 = nn.BatchNorm2d(200)
        self.pool4 = nn.MaxPool2d((1, 3))  # 增加一个池化层以确保时间维度充分压缩
        self.dropout4 = nn.Dropout(0.5)

        # 计算 AdaptiveAvgPool 之前的特征维度
        # 初始 T=384。经过 4 次 MaxPool(k=3)，时间维度近似为 384 / (3^4) = 384 / 81 ≈ 4.7
        # 实际计算: 384 -> 128 -> 42 -> 14 -> 4 (Floor(42/3)=14, Floor(14/3)=4)
        # Global average pooling 后的通道数是 200。

        # Feature extraction layer
        self.fc_features = nn.Linear(200, feature_dim)  # 输入是最后的通道数 200

        # Classifier
        self.classifier = nn.Linear(feature_dim, n_classes)

    def extract_features(self, x):

        if x.dim() == 3:
            # (batch, n_channels, n_timepoints) -> (batch, 1, n_channels, n_timepoints)
            x = x.unsqueeze(1)
        elif x.dim() == 2:
            batch_size = x.size(0)
            x = x.view(batch_size, 1, self.n_channels, self.n_timepoints)

        # 1. Block 1 (Temporal + Spatial)
        x = self.conv1_t(x)
        x = self.conv2_s(x)  # Spatial filtering
        x = self.bn1(x)
        x = self.elu(x)  # 使用ELU
        x = self.pool1(x)
        x = self.dropout1(x)

        # 2. Block 2
        x = self.conv3(x)
        x = self.bn2(x)
        x = self.elu(x)  # 使用ELU
        x = self.pool2(x)
        x = self.dropout2(x)

        # 3. Block 3
        x = self.conv4(x)
        x = self.bn3(x)
        x = self.elu(x)  # 使用ELU
        x = self.pool3(x)
        x = self.dropout3(x)

        # 4. Block 4 (New)
        x = self.conv5(x)
        x = self.bn4(x)
        x = self.elu(x)  # 使用ELU
        x = self.pool4(x)
        x = self.dropout4(x)

        # Global average pooling (H=1, W' 是最后的时间维度)
        # 将 (B, C, 1, W') 转换为 (B, C, 1, 1)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)  # (B, C) -> (B, 200)

        # Feature extraction (ReLU is fine here)
        features = torch.relu(self.fc_features(x))
        return features

    def forward(self, x):
        features = self.extract_features(x)
        out = self.classifier(features)
        # 严格遵守 (out, features) 的输出格式
        return out, features

class ClientAdaptiveLayer(nn.Module):

    def __init__(self, feature_dim, num_classes, rho=0.7):
        super(ClientAdaptiveLayer, self).__init__()
        self.feature_decoupling = FeatureDecoupling(feature_dim, rho=rho)
        self.num_classes = num_classes

    def compute_class_prototypes(self, x_rep, labels):
        y_onehot = F.one_hot(labels, num_classes=self.num_classes).float()
        ytb = torch.mm(y_onehot.t(), y_onehot)
        ytb_inv = torch.inverse(ytb + 1e-5 * torch.eye(self.num_classes).to(ytb.device))
        prototypes = torch.mm(torch.mm(ytb_inv, y_onehot.t()), x_rep)
        return prototypes

    def add_differential_privacy(self, prototypes, sigma=0.15):
        noise = torch.randn_like(prototypes) * sigma
        return prototypes + noise

    def forward(self, features, labels=None, add_dp=True, sigma=0.15):
        x_rep, x_usr, mu, logvar = self.feature_decoupling(features)

        results = {'x_rep': x_rep, 'x_usr': x_usr, 'mu': mu, 'logvar': logvar}

        if labels is not None:
            prototypes = self.compute_class_prototypes(x_rep, labels)
            if add_dp:
                prototypes = self.add_differential_privacy(prototypes, sigma)
            results['prototypes'] = prototypes

        return results



class LocalClient:
    """本地客户端"""

    def __init__(self, client_id, model, feature_dim, num_classes, device,
                 lr=1e-4, rho=0.7, lambda_r=0.1, lambda_ewc=0.01, tau_ewc=10):
        self.client_id = client_id
        self.device = device
        self.num_classes = num_classes

        self.feature_extractor = model.to(device)
        self.adaptive_layer = ClientAdaptiveLayer(feature_dim, num_classes, rho).to(device)

        params = list(self.feature_extractor.parameters()) + \
                 list(self.adaptive_layer.parameters())
        self.optimizer = optim.Adam(params, lr=lr)

        self.lambda_r = lambda_r
        self.lambda_ewc = lambda_ewc
        self.tau_ewc = tau_ewc

        self.fisher_dict = {}
        self.optimal_params = {}
        self.local_prototypes = None
        self.global_prototypes = None

    def compute_fisher_information(self, dataloader, num_samples=200):
        """计算Fisher信息矩阵"""
        # 保持训练模式,因为LSTM的backward需要训练模式
        self.feature_extractor.train()
        self.fisher_dict = {}

        # 初始化Fisher信息
        for name, param in self.feature_extractor.named_parameters():
            if param.requires_grad:
                self.fisher_dict[name] = torch.zeros_like(param)

        # 采样计算Fisher信息
        sample_count = 0
        for data, labels in dataloader:
            if sample_count >= num_samples:
                break

            data, labels = data.to(self.device), labels.to(self.device)
            batch_size = data.size(0)

            # 清空梯度
            self.optimizer.zero_grad()

            # 前向传播
            out, features = self.feature_extractor(data)

            # 使用模型输出计算损失
            loss = F.cross_entropy(out, labels)

            # 反向传播
            loss.backward()

            # 累积梯度平方
            for name, param in self.feature_extractor.named_parameters():
                if param.grad is not None and name in self.fisher_dict:
                    self.fisher_dict[name] += param.grad.data ** 2 * batch_size

            sample_count += batch_size

        # 归一化Fisher信息
        for name in self.fisher_dict:
            self.fisher_dict[name] /= sample_count

        # 保存当前最优参数
        self.optimal_params = {}
        for name, param in self.feature_extractor.named_parameters():
            if name in self.fisher_dict:
                self.optimal_params[name] = param.data.clone()

        # 清空梯度
        self.optimizer.zero_grad()

    def ewc_loss(self):
        """EWC损失 - 防止灾难性遗忘"""
        if not self.fisher_dict:
            return torch.tensor(0.0).to(self.device)

        loss = 0
        for name, param in self.feature_extractor.named_parameters():
            if name in self.fisher_dict:
                # Fisher信息加权的参数偏移惩罚
                loss += (self.fisher_dict[name] *
                         (param - self.optimal_params[name]) ** 2).sum()

        return (self.tau_ewc / 2) * loss

    def prototype_regularization_loss(self):
        """原型正则化损失"""
        if self.local_prototypes is None or self.global_prototypes is None:
            return torch.tensor(0.0).to(self.device)

        loss = torch.sum((self.local_prototypes - self.global_prototypes) ** 2)
        return loss

    def train_local(self, dataloader, global_prototypes=None,
                    alpha_t=0.6, sigma=0.15, epochs=5, show_progress=True,
                    update_fisher=False):
        """
        本地训练

        Args:
            dataloader: 数据加载器
            global_prototypes: 全局原型
            alpha_t: 融合系数
            sigma: 差分隐私噪声标准差
            epochs: 训练轮数
            show_progress: 是否显示进度条
            update_fisher: 是否在训练后更新Fisher信息(用于EWC)
        """
        self.feature_extractor.train()
        self.adaptive_layer.train()
        self.global_prototypes = global_prototypes

        epoch_pbar = tqdm(range(epochs), desc=f'Client {self.client_id}',
                          leave=False, disable=not show_progress)

        for epoch in epoch_pbar:
            total_loss = 0
            num_batches = 0

            batch_pbar = tqdm(dataloader, desc=f'  Epoch {epoch + 1}/{epochs}',
                              leave=False, disable=not show_progress)

            for batch_idx, (data, labels) in enumerate(batch_pbar):
                data, labels = data.to(self.device), labels.to(self.device)

                out, features = self.feature_extractor(data)
                adaptive_results = self.adaptive_layer(features, labels,
                                                       add_dp=False, sigma=sigma)
                x_rep = adaptive_results['x_rep']
                x_usr = adaptive_results['x_usr']
                mu = adaptive_results['mu']
                logvar = adaptive_results['logvar']

                self.local_prototypes = self.adaptive_layer.compute_class_prototypes(
                    x_rep, labels)

                if global_prototypes is not None:
                    fused_prototypes = ((1 - alpha_t) * self.local_prototypes +
                                        alpha_t * global_prototypes)
                else:
                    fused_prototypes = self.local_prototypes

                distances = torch.cdist(x_rep, fused_prototypes)
                pred_logits = -distances

                loss_cls = F.cross_entropy(pred_logits, labels)

                recon_loss = F.mse_loss(x_usr + x_rep, features, reduction='mean')
                kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                vae_loss = recon_loss + 0.1 * kld_loss

                loss_reg = self.prototype_regularization_loss()
                loss_ewc = self.ewc_loss()

                total_loss_batch = (loss_cls + 0.1 * vae_loss +
                                    self.lambda_r * loss_reg + self.lambda_ewc * loss_ewc)

                self.optimizer.zero_grad()
                total_loss_batch.backward()
                self.optimizer.step()

                total_loss += total_loss_batch.item()
                num_batches += 1

                batch_pbar.set_postfix({'loss': f'{total_loss_batch.item():.4f}'})

            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            epoch_pbar.set_postfix({'avg_loss': f'{avg_loss:.4f}'})

        # 训练完成后更新Fisher信息(如果需要)
        if update_fisher:
            self.compute_fisher_information(dataloader)

    def get_prototypes(self, dataloader, sigma=0.15):
        """获取原型"""
        self.feature_extractor.eval()
        self.adaptive_layer.eval()

        all_features = []
        all_labels = []

        with torch.no_grad():
            for data, labels in dataloader:
                data = data.to(self.device)
                _, features = self.feature_extractor(data)
                adaptive_results = self.adaptive_layer(features, add_dp=False)
                all_features.append(adaptive_results['x_rep'])
                all_labels.append(labels)

        all_features = torch.cat(all_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0).to(self.device)

        prototypes = self.adaptive_layer.compute_class_prototypes(
            all_features, all_labels)
        prototypes = self.adaptive_layer.add_differential_privacy(prototypes, sigma)

        return prototypes.cpu().numpy()

    def evaluate(self, dataloader):
        """评估模型"""
        self.feature_extractor.eval()
        self.adaptive_layer.eval()

        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for data, labels in dataloader:
                data, labels = data.to(self.device), labels.to(self.device)
                _, features = self.feature_extractor(data)

                adaptive_results = self.adaptive_layer(features, add_dp=False)
                x_rep = adaptive_results['x_rep']

                if self.global_prototypes is not None:
                    prototypes = self.global_prototypes
                else:
                    prototypes = (self.local_prototypes if self.local_prototypes is not None
                                  else torch.zeros(self.num_classes, x_rep.size(1)).to(self.device))

                distances = torch.cdist(x_rep, prototypes)
                pred = torch.argmin(distances, dim=1)

                correct += (pred == labels).sum().item()
                total += labels.size(0)

                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = correct / total if total > 0 else 0
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        return accuracy, f1