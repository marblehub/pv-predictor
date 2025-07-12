# -*- coding: utf-8 -*-
"""
Created on Sat Jul 12 16:01:19 2025

@author: g.whyte
"""
# model.py
import torch.nn as nn
import torch

class PVNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(2, 6)
        self.output = nn.Linear(6, 1)

    def forward(self, x):
        x = torch.relu(self.hidden(x))
        return self.output(x)

