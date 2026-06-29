import copy
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  # КРИТИЧЕСКИ ВАЖНО ДЛЯ NN
from torch.utils.data import DataLoader, TensorDataset
import tqdm

# 1. Загрузка и подготовка данных
data = fetch_california_housing()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.7, shuffle=True, random_state=42
)

# Нормализация признаков
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Конвертация в тензоры PyTorch
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

# Использование DataLoader вместо ручных срезов
batch_size = 64  # Увеличим размер батча для стабильности градиента
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)  # Перемешивание на каждой эпохе

# 2. Архитектура модели
model = nn.Sequential(
    nn.Linear(8, 24),
    nn.ReLU(),
    nn.Linear(24, 12),
    nn.ReLU(),
    nn.Linear(12, 6),
    nn.ReLU(),
    nn.Linear(6, 1),
)

loss_fn = nn.MSELoss()
optimizer = optim.Adam(
    model.parameters(), lr=0.005
)  # Поднят LR, так как данные нормализованы

n_epochs = 50
best_mse = np.inf
best_weights = None
history = []

# 3. Цикл обучения
for epoch in range(n_epochs):
    model.train()
    # Включаем прогресс-бар, убрав disable=True
    with tqdm.tqdm(
        train_loader, unit="batch", desc=f"Epoch {epoch+1}/{n_epochs}"
    ) as pbar:
        for X_batch, y_batch in pbar:
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=float(loss))

    # Валидация в конце эпохи
    model.eval()
    with torch.no_grad():  # Отключаем расчет градиентов для экономии памяти
        y_pred = model(X_test)
        mse = float(loss_fn(y_pred, y_test))

    history.append(mse)

    if mse < best_mse:
        best_mse = mse
        best_weights = copy.deepcopy(model.state_dict())

# 4. Восстановление лучшей модели и вывод результатов
model.load_state_dict(best_weights)
print("MSE: %.4f" % best_mse)
print("RMSE: %.4f" % np.sqrt(best_mse))

# Визуализация обучения
plt.figure(figsize=(8, 4))
plt.plot(history, label="Validation MSE")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("Процесс обучения нейросети")
plt.legend()
plt.grid(True)
plt.show()
