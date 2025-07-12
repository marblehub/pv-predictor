# -*- coding: utf-8 -*-
"""
Created on Sat Jul 12 16:01:19 2025

@author: g.whyte
"""

# train.py
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
from model import PVNet

# 1. Generate synthetic data
np.random.seed(0)
N = 500
temperature = np.random.uniform(15, 45, N)
irradiance = np.random.uniform(200, 1000, N)
power = 0.05 * irradiance - 0.1 * temperature + np.random.normal(0, 10, N)  # linear-ish

X = np.vstack((temperature, irradiance)).T
y = power.reshape(-1, 1)

# 2. Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Prepare PyTorch datasets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

# 4. Load small model from model.py
model = PVNet()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 5. Train
for epoch in range(300):
    model.train()
    optimizer.zero_grad()
    output = model(X_train_tensor)
    loss = criterion(output, y_train_tensor)
    loss.backward()
    optimizer.step()

print("Training complete. Final loss:", loss.item())

# 6. Save model + scaler
torch.save(model.state_dict(), "pv_model.pth")
joblib.dump(scaler, "scaler.pkl")

