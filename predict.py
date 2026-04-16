import torch
from sasrec.model import SASRec

# 1. Задаем те же параметры, что были при обучении
# ВАЖНО: число товаров должно точно совпадать!
num_items = 26744  # Пример для ML-20M
max_len = 200
hidden_units = 256

# 2. Создаем "пустую" архитектуру
model = SASRec(item_num=num_items, maxlen=max_len, hidden_units=hidden_units)

# 3. Загружаем веса из твоего файла
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load('checkpoints/best_model.pt', map_location=device))
model.to(device)
model.eval()  # Переключаем в режим предсказания

# 4. Пример: история пользователя (ID товаров)
# Допустим, он посмотрел фильмы с ID [12, 45, 120]
user_history = [12, 45, 120]

# Подготавливаем тензор (добавляем паддинг слева и батч-размерность)
input_ids = torch.zeros((1, max_len), dtype=torch.long)

print(input_ids)

input_ids[0, -len(user_history):] = torch.tensor(user_history)

print(input_ids)

input_ids = input_ids.to(device)

# 5. Получаем предсказание
with torch.no_grad():
    hidden = model(input_ids)  # Прогоняем через Трансформер

    print(hidden.shape)

    last_hidden = hidden[0, -1, :]  # Берем вектор последнего состояния

    # Идеальный эмбеддинг для следующего товара.
    print(last_hidden)

    # Считаем похожесть на все товары через скалярное произведение
    scores = torch.matmul(last_hidden, model.item_emb.weight.T)  #

    # Берем ТОП-10 рекомендаций
    top_scores, top_indices = torch.topk(scores, k=11)  # +1 на случай, если выпадет уже виденный товар
    print("Рекомендованные ID товаров:", top_indices.cpu().numpy())