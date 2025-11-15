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

        F1 = 8  # 初始时间卷积核数量
        D = 2  # 深度乘数 (Depth Multiplier)
        F2 = F1 * D  # 深度卷积输出通道数量 (F1 * D)
        kernel_length = 64  # 初始时间卷积核长度
        P1 = 4  # Depthwise Conv后的池化因子
        P2 = 8  # Separable Conv后的池化因子


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
                      groups=F1,
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
            # Depthwise Temporal Conv (时间特征提炼)
            nn.Conv2d(in_channels=F2,
                      out_channels=F2,
                      kernel_size=(1, 16),
                      padding=(0, 8),
                      groups=F2,
                      bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            # Pointwise Conv (特征混合)
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

        self.fc_features = nn.Linear(self.flatten_size, feature_dim)

        self.classifier = nn.Linear(feature_dim, n_classes)

    def _get_flatten_size(self):
        """计算展平后的特征大小"""

        F2 = 16
        P1 = 4
        P2 = 8

        x = torch.zeros(1, 1, self.n_channels, self.n_timepoints)

        T_pool1 = int(np.floor(self.n_timepoints / P1))

        T_pool2 = int(np.floor(T_pool1 / P2))

        return F2 * 1 * T_pool2

    def extract_features(self, x):
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

        features = torch.relu(self.fc_features(x))
        return features

    def forward(self, x):
        features = self.extract_features(x)
        out = self.classifier(features)
        return out, features

class ACRNN(nn.Module):
    def __init__(self, n_timepoints=384, n_channels=32, n_classes=2, feature_dim=128):
        super(ACRNN, self).__init__()
        self.feature_dim = feature_dim
        self.n_channels = n_channels
        self.n_timepoints = n_timepoints  # 384

        # CNN layers
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d((2, 2))
        self.dropout = nn.Dropout(0.25)


        LSTM_INPUT_SIZE = 128 * (32 // 4)
        self.lstm = nn.LSTM(LSTM_INPUT_SIZE, 128, batch_first=True, bidirectional=True)

        # Feature extraction & Classifier
        self.fc_features = nn.Linear(256, feature_dim)  # 256 = 128 * 2 (Bidirectional)
        self.classifier = nn.Linear(feature_dim, n_classes)

        self.fixed_time_steps = 384 // 4  # 96
        self.fixed_height_out = 32 // 4  # 8


    def extract_features(self, x):
        if x.dim() == 3:
            # (batch, 32, 384) -> (batch, 1, 32, 384)
            x = x.unsqueeze(1)
        elif x.dim() == 2:
            batch_size = x.size(0)
            x = x.view(batch_size, 1, self.n_channels, self.n_timepoints)

        # CNN 提取特征
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.dropout(x)
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.dropout(x)

        batch_size = x.size(0)

        x = x.permute(0, 3, 1, 2).contiguous()

        time_steps = x.size(1)
        x = x.view(batch_size, time_steps, -1)

        # LSTM 处理时序
        x, _ = self.lstm(x)

        # 特征聚合 (取最后一个时间步的输出)
        features_raw = x[:, -1, :]

        # 特征映射
        features = torch.relu(self.fc_features(features_raw))
        return features

    def forward(self, x):
        features = self.extract_features(x)
        out = self.classifier(features)
        return out, features

class DeepConvNet(nn.Module):

    def __init__(self, n_timepoints=384, n_channels=32, n_classes=2, feature_dim=128):
        super(DeepConvNet, self).__init__()
        self.feature_dim = feature_dim
        self.n_channels = n_channels
        self.n_timepoints = n_timepoints

        self.conv1_t = nn.Conv2d(1, 25, kernel_size=(1, 25), padding=(0, 12), bias=False)

        self.conv2_s = nn.Conv2d(25, 25, kernel_size=(n_channels, 1), bias=False)

        self.bn1 = nn.BatchNorm2d(25)
        self.elu = nn.ELU()
        self.pool1 = nn.MaxPool2d((1, 3))
        self.dropout1 = nn.Dropout(0.5)

        self.conv3 = nn.Conv2d(25, 50, kernel_size=(1, 10), padding=(0, 5), bias=False)
        self.bn2 = nn.BatchNorm2d(50)
        self.pool2 = nn.MaxPool2d((1, 3))
        self.dropout2 = nn.Dropout(0.5)

        self.conv4 = nn.Conv2d(50, 100, kernel_size=(1, 10), padding=(0, 5), bias=False)
        self.bn3 = nn.BatchNorm2d(100)
        self.pool3 = nn.MaxPool2d((1, 3))
        self.dropout3 = nn.Dropout(0.5)

        self.conv5 = nn.Conv2d(100, 200, kernel_size=(1, 10), padding=(0, 5), bias=False)
        self.bn4 = nn.BatchNorm2d(200)
        self.pool4 = nn.MaxPool2d((1, 3))
        self.dropout4 = nn.Dropout(0.5)


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

        # Block 1 (Temporal + Spatial)
        x = self.conv1_t(x)
        x = self.conv2_s(x)  # Spatial filtering
        x = self.bn1(x)
        x = self.elu(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        # Block 2
        x = self.conv3(x)
        x = self.bn2(x)
        x = self.elu(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        # Block 3
        x = self.conv4(x)
        x = self.bn3(x)
        x = self.elu(x)
        x = self.pool3(x)
        x = self.dropout3(x)

        # Block 4
        x = self.conv5(x)
        x = self.bn4(x)
        x = self.elu(x)
        x = self.pool4(x)
        x = self.dropout4(x)


        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)  # (B, C) -> (B, 200)


        features = torch.relu(self.fc_features(x))
        return features

    def forward(self, x):
        features = self.extract_features(x)
        out = self.classifier(features)
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
        self.feature_extractor.train()
        self.fisher_dict = {}


        for name, param in self.feature_extractor.named_parameters():
            if param.requires_grad:
                self.fisher_dict[name] = torch.zeros_like(param)


        sample_count = 0
        for data, labels in dataloader:
            if sample_count >= num_samples:
                break

            data, labels = data.to(self.device), labels.to(self.device)
            batch_size = data.size(0)


            self.optimizer.zero_grad()


            out, features = self.feature_extractor(data)


            loss = F.cross_entropy(out, labels)
            loss.backward()


            for name, param in self.feature_extractor.named_parameters():
                if param.grad is not None and name in self.fisher_dict:
                    self.fisher_dict[name] += param.grad.data ** 2 * batch_size

            sample_count += batch_size


        for name in self.fisher_dict:
            self.fisher_dict[name] /= sample_count


        self.optimal_params = {}
        for name, param in self.feature_extractor.named_parameters():
            if name in self.fisher_dict:
                self.optimal_params[name] = param.data.clone()


        self.optimizer.zero_grad()

    def ewc_loss(self):
        """EWC loss"""
        if not self.fisher_dict:
            return torch.tensor(0.0).to(self.device)

        loss = 0
        for name, param in self.feature_extractor.named_parameters():
            if name in self.fisher_dict:
                # Fisher
                loss += (self.fisher_dict[name] *
                         (param - self.optimal_params[name]) ** 2).sum()

        return (self.tau_ewc / 2) * loss

    def prototype_regularization_loss(self):
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