#!/usr/bin/env python3

import numpy as np
import torch.nn as nn

import models


class ConsecutiveMatch(nn.Module):
    def __init__(self):
        super().__init__()
        self.cosine = models.PairwiseCosine()

    def forward(self, desc_src, desc_dst, points_dst):
        confidence, idx = self.cosine(desc_src, desc_dst).max(dim=2)
        matched = points_dst.gather(1, idx.unsqueeze(2).expand(-1, -1, 2))

        return matched, confidence

