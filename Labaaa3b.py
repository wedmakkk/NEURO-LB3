import torch
import torch.nn as nn
import pandas as pd
import numpy as np

# гружу данные из файла
df = pd.read_csv('data.csv')

print(df.head())

# Разделение данных на признаки (X) и метки (y)
# столбцы  0, 1, 2, 3 содержат параметры, а 4й — вид ириса
X = df.iloc[:, :4].values  # Первые 4 столбца объединяем в массив х
y = df.iloc[:, 4].values   # Последний столбец (вид растения) – в у

# Преобразуем метки в числовой формат
label_map = {"Iris-setosa": 0, "Iris-versicolor": 1}
y = np.array([label_map[label] for label in y])

# преобразую данные в тензоры
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

# Создание полносвязного слоя 
# 4 параметра и 2 класса
model = nn.Linear(4, 2)

# Определение функции потерь и оптимизатора
loss_fn = nn.CrossEntropyLoss()  # Подходит для задач классификации
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


num_epochs = 100
for epoch in range(num_epochs):
    # Прямой проход (предсказание)
    outputs = model(X_tensor)
    
    # Вычисление ошибки
    loss = loss_fn(outputs, y_tensor)
    
    # Обратный проход и оптимизация
    optimizer.zero_grad()  # Обнуление градиентов
    loss.backward()        # Вычисление градиентов
    optimizer.step()       # Обновление весов
    
    # Вывод ошибки каждые 10 эпох
    if (epoch + 1) % 10 == 0:
        print(f'Эпоха [{epoch+1}/{num_epochs}], Ошибка: {loss.item():.4f}')

# Тестирование модели
with torch.no_grad():  # Отключаем вычисление градиентов
    predictions = model(X_tensor)
    _, predicted_classes = torch.max(predictions, 1)  # Получаем предсказанные классы

t = torch.tensor(y, dtype=torch.long)  # Эталонный тензор


# Вывод результатов
print("Предсказанные классы:", predicted_classes)
print("Эталонные метки:", t)