# -*- coding: utf-8 -*-
"""
Created on Sat Jul 12 16:01:19 2025

@author: g.whyte
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from model import PVNet  # import your network architecture

# 1. Generate synthetic PV data
np.random.seed(42)
torch.manual_seed(42)
n_samples = 1000

temperature = np.random.uniform(15, 45, n_samples)
irradiance = np.random.uniform(200, 1000, n_samples)

# Simulated power output (nonlinear relation + noise)
power_output = 0.9 * irradiance * (1 - 0.004 * (temperature - 25)) / 1000
power_output += np.random.normal(0, 0.05, n_samples)  # add noise

X = np.column_stack((temperature, irradiance))
y = power_output

# 2. Train-test split
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Standardize input features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Save the scaler
joblib.dump(scaler, "scaler.pkl")

# 4. Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

# 5. Initialize model, loss, optimizer
model = PVNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 6. Train the ANN
epochs = 500
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train_tensor)
    loss = criterion(output, y_train_tensor)
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# 7. Save the trained model
torch.save(model.state_dict(), "pv_model.pth")
print("✅ Training complete. Model and scaler saved.")
