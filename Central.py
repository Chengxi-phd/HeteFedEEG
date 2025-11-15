import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
from typing import Dict, List, Tuple



class CentralServer:

    def __init__(self, num_classes, feature_dim):
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.global_prototypes = None
        self.prev_global_prototypes = None

    def compute_residual_distance(self, proto1, proto2):
        proto1_norm = proto1 / (np.linalg.norm(proto1, axis=1, keepdims=True) + 1e-8)
        proto2_norm = proto2 / (np.linalg.norm(proto2, axis=1, keepdims=True) + 1e-8)
        cos_sim = np.sum(proto1_norm * proto2_norm, axis=1)
        distance = 1 - cos_sim
        return distance.mean()

    def aggregate_prototypes(self, client_prototypes: List[np.ndarray]):
        if self.prev_global_prototypes is None:
            self.global_prototypes = np.mean(client_prototypes, axis=0)
        else:
            weights = []
            for proto in client_prototypes:
                distance = self.compute_residual_distance(proto, self.prev_global_prototypes)
                weight = 1.0 / (distance + 1e-5)
                weights.append(weight)

            weights = np.array(weights)
            weights = weights / weights.sum()

            self.global_prototypes = np.zeros_like(client_prototypes[0])
            for i, proto in enumerate(client_prototypes):
                self.global_prototypes += weights[i] * proto

        self.prev_global_prototypes = self.global_prototypes.copy()
        return self.global_prototypes
