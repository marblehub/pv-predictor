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
        self.hidden1 = nn.Linear(2, 10)
        self.hidden2 = nn.Linear(10, 10)
        self.output = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        return self.output(x)
