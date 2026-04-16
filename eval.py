import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from sasrec.model import SASRec
from sasrec.data import load_data, split_leave_one_out
from sasrec.evaluate import evaluate


def evaluate_robust(args):
    user_sequences, num_items = load_data(args.data_path)
    _, _, _, test_sequences, test_targets = split_leave_one_out(user_sequences)

    model = SASRec(
        item_num=num_items,
        maxlen=args.maxlen,
        hidden_units=args.hidden_units,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        dropout_rate=args.dropout_rate
    ).to(args.device)

    print(f"Загрузка весов из: {args.model_path}")
    state_dict = torch.load(args.model_path, map_location=args.device)

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    with torch.no_grad():
        for i in range(model.num_blocks):
            alpha_key = f'alphas.{i}'
            if alpha_key in missing_keys:
                model.alphas[i].copy_(torch.tensor([1.0], device=args.device))
                print(f"[FIX] Параметр {alpha_key} не найден. Установлен в 1.0 (режим Legacy).")

    print("\nМодель успешно собрана. Начинаем расчет метрик на тестовой выборке...\n")

    model.eval()
    t_test = evaluate(
        model=model,
        user_sequences=test_sequences,
        user_targets=test_targets,
        num_items=num_items,
        max_length=args.maxlen,
        device=args.device,
        k=10,
        batch_size=256,
        filter_seen=True
    )

    print(f"\n{'=' * 40}")
    print(f"РЕЗУЛЬТАТЫ ДЛЯ: {args.model_path}")
    print(f"{'=' * 40}")
    for name, value in t_test.items():
        print(f"  {name}: {value:.4f}")
    print(f"{'=' * 40}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True, help='Путь к датасету (например: data/ml-1m.txt)')
    parser.add_argument('--model_path', required=True, help='Путь к файлу .pt')

    parser.add_argument('--maxlen', type=int, default=200)
    parser.add_argument('--hidden_units', type=int, default=256)
    parser.add_argument('--num_blocks', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=1)
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()

    evaluate_robust(args)