import torch
import argparse
import os


def stack_model_weights(old_path, new_path, method='adjacent'):
    print(f"Загрузка старых весов из {old_path}...")
    old_state_dict = torch.load(old_path, map_location='cpu')
    new_state_dict = {}

    old_indices = set()
    for key in old_state_dict.keys():
        if 'attention_layers.' in key:
            old_indices.add(int(key.split('.')[1]))
    L = len(old_indices)
    print(f"Обнаружено блоков: {L}. Создаем модель на {2 * L} блоков (Метод: {method}).")

    prefixes_to_stack = ['attention_layernorms', 'attention_layers', 'forward_layernorms', 'forward_layers', 'alphas']

    for key, value in old_state_dict.items():
        is_stacked_layer = any(key.startswith(prefix + '.') for prefix in prefixes_to_stack)

        if is_stacked_layer:
            parts = key.split('.')
            prefix = parts[0]
            old_idx = int(parts[1])
            remaining_key = ".".join(parts[2:])

            new_idx1, new_idx2 = None, None

            if method == 'adjacent':
                new_idx1 = 2 * old_idx
                new_idx2 = 2 * old_idx + 1
            elif method == 'cross':
                new_idx1 = old_idx
                new_idx2 = old_idx + L

            key1 = f"{prefix}.{new_idx1}.{remaining_key}" if remaining_key else f"{prefix}.{new_idx1}"
            key2 = f"{prefix}.{new_idx2}.{remaining_key}" if remaining_key else f"{prefix}.{new_idx2}"

            new_state_dict[key1] = value.clone()
            if prefix == 'alphas':
                new_state_dict[key2] = torch.zeros_like(value)
            else:
                new_state_dict[key2] = value.clone()
        else:
            # Эмбеддинги и last_layernorm переносятся без изменений
            new_state_dict[key] = value.clone()

    torch.save(new_state_dict, new_path)
    print(f"Успешно сохранено в {new_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Путь к частично обученной модели')
    parser.add_argument('--output', type=str, required=True, help='Путь для расширенной модели, которую планируется обучать далее')
    parser.add_argument('--method', type=str, default='adjacent', choices=['adjacent', 'cross'])
    args = parser.parse_args()

    stack_model_weights(args.input, args.output, args.method)
