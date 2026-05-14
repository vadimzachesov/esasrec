import os
import subprocess
import re
import csv
from datetime import datetime

DATASET_PATH = "../../data/ml-20m.txt"
BASE_DIR = "benchmark_results"
SUMMARY_FILE = os.path.join(BASE_DIR, "summary_metrics.csv")

# Базовые параметры
BASE_EPOCHS = "100"
BASE_PATIENCE = "5"
BATCH_SIZE = "256"

# План исследований
EXPERIMENTS = {
    "01_depth": {
        "param_name": "--num_blocks",
        "values": [2, 4, 8, 16, 24],
        "constants": {"--hidden_units": "64", "--max_length": "50"}
    },
    "02_embeds": {
        "param_name": "--hidden_units",
        "values": [64, 128, 256, 512, 1024, 4096],
        "constants": {"--num_blocks": "2", "--max_length": "50"}
    },
    "03_context": {
        "param_name": "--max_length",
        "values": [25, 50, 100, 200, 300, 400],
        "constants": {"--num_blocks": "2", "--hidden_units": "64"}
    }
}


def parse_and_extract_history(log_path, history_csv_path):
    """Вытаскивает финальные метрики и создает CSV с историей обучения для графиков"""
    hr, ndcg = None, None
    history = []

    if not os.path.exists(log_path):
        return hr, ndcg

    epoch_pattern = re.compile(
        r"Epoch\s+(\d+)/\d+\s+\|\s+Loss:\s+([0-9.]+)\s+\|\s+Val HR@10:\s+([0-9.]+)\s+\|\s+Val NDCG@10:\s+([0-9.]+)")

    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()

        # 1. Собираем историю по эпохам
        for match in epoch_pattern.finditer(content):
            epoch, loss, val_hr, val_ndcg = match.groups()
            history.append([epoch, loss, val_hr, val_ndcg])

        # 2. Ищем финальные метрики на тесте
        if "--- Test metrics @10 ---" in content:
            test_block = content.split("--- Test metrics @10 ---")[1]
            hr_match = re.search(r"HR@10:\s+([0-9.]+)", test_block)
            ndcg_match = re.search(r"NDCG@10:\s+([0-9.]+)", test_block)
            if hr_match: hr = float(hr_match.group(1))
            if ndcg_match: ndcg = float(ndcg_match.group(1))

    # Сохраняем историю в CSV для графиков
    if history:
        with open(history_csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "Loss", "Val_HR_10", "Val_NDCG_10"])
            writer.writerows(history)

    return hr, ndcg


def main():
    os.makedirs(BASE_DIR, exist_ok=True)

    with open(SUMMARY_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Thread", "Parameter", "Value", "Test HR@10", "Test NDCG@10", "Log File"])

    for thread_name, config in EXPERIMENTS.items():
        thread_dir = os.path.join(BASE_DIR, thread_name)
        os.makedirs(thread_dir, exist_ok=True)

        param_flag = config["param_name"]
        print(f"\n{'=' * 50}\n🚀 ЗАПУСК ТРЕДА: {thread_name}\n{'=' * 50}")

        for val in config["values"]:
            val_str = str(val)
            run_name = f"{param_flag.strip('--')}_{val_str}"
            run_dir = os.path.join(thread_dir, run_name)

            log_file = os.path.join(thread_dir, f"log_{run_name}.txt")
            history_file = os.path.join(thread_dir, f"learning_curve_{run_name}.csv")

            print(f"\n⏳ Тестируем {param_flag} = {val_str}...")

            cmd = [
                "python", "-m", "sasrec.train",
                "--data_path", DATASET_PATH,
                "--max_epochs", BASE_EPOCHS,
                "--patience", BASE_PATIENCE,
                "--batch_size", BATCH_SIZE,
                "--save_dir", run_dir,
                param_flag, val_str
            ]

            for k, v in config["constants"].items():
                cmd.extend([k, v])

            with open(log_file, "w", encoding="utf-8") as f:
                process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
                process.wait()

            if process.returncode != 0:
                print(f"❌ Ошибка/OOM при запуске {run_name}.")
                hr, ndcg = "OOM/Error", "OOM/Error"
            else:
                hr, ndcg = parse_and_extract_history(log_file, history_file)
                print(f"✅ Готово! HR@10: {hr}, NDCG@10: {ndcg}")

            with open(SUMMARY_FILE, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([thread_name, param_flag, val_str, hr, ndcg, log_file])

    print(f"\n🎉 Все данные (включая CSV для графиков) собраны в: {BASE_DIR}")


if __name__ == "__main__":
    main()
