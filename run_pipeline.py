import os
import subprocess
import math
import sys
from datetime import datetime

# ==========================================
# НАСТРОЙКИ ПО СТАТЬЕ (СЦЕНАРИЙ TS)
# ==========================================
DATASET_PATH = "../../data/ml-20m.txt"

# 1. Сначала честно обучаем (или вспоминаем) за сколько сходится модель
KNOWN_CONVERGENCE_EPOCHS = 68  # ВПИШИ СЮДА то, за сколько у тебя сошелся бейзлайн
FRACTION = 0.3  # Доля для прогрева (от 1/8 до 1/3 по статье)

# 2. Лимит для самой последней, глубокой модели (даем ей сойтись до конца)
MAX_FINAL_EPOCHS = 100

# Оптимизации для A100 (Оставлены для скорости)
HIDDEN_UNITS = 256
MAX_LENGTH = 100
BATCH_SIZE = "256"

DEPTHS = [2, 4, 8, 16]

BASE_AUTOPILOT_DIR = "autopilot_experiments"
TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
RUN_DIR = os.path.join(BASE_AUTOPILOT_DIR, TIMESTAMP)


# ==========================================

def run_cmd(cmd, log_file=None):
    command_str = ' '.join(cmd)
    print(f"\n🚀 [ЗАПУСК]: {command_str}")

    if log_file:
        with open(log_file, "w", encoding="utf-8") as f:
            # Используем Popen, чтобы читать поток по мере его появления
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            # Пишем и в терминал (sys.stdout), и в файл (f)
            for line in process.stdout:
                sys.stdout.write(line)
                f.write(line)
                f.flush()

            process.wait()
            if process.returncode != 0:
                print(f"\n❌ ОШИБКА: Команда завершилась с кодом {process.returncode}")
                sys.exit(1)
    else:
        subprocess.run(cmd, check=True)


def main():
    os.makedirs(RUN_DIR, exist_ok=True)
    print(f"📁 Директория эксперимента: {RUN_DIR}")

    # Считаем ту самую "правильную долю" по статье
    partial_epochs = math.ceil(KNOWN_CONVERGENCE_EPOCHS * FRACTION)
    print(f"🎯 Математика StackRec: Сходимость = {KNOWN_CONVERGENCE_EPOCHS}, Доля = {FRACTION}")
    print(f"⏳ Прогревочных эпох на каждый промежуточный шаг: {partial_epochs}")

    current_pretrain = None

    for i, depth in enumerate(DEPTHS):
        is_final = (i == len(DEPTHS) - 1)
        epochs = MAX_FINAL_EPOCHS if is_final else partial_epochs

        save_dir = os.path.join(RUN_DIR, f"checkpoints_{depth}b")
        log_path = os.path.join(RUN_DIR, f"train_log_{depth}b.txt")

        print(f"\n{'=' * 60}")
        print(f"🔥 ЭТАП {i + 1}/{len(DEPTHS)}: Модель {depth} блоков | Эпох: {epochs}")
        print(f"{'=' * 60}")

        train_cmd = [
            "python", "-m", "sasrec.train",
            "--data_path", DATASET_PATH,
            "--num_blocks", str(depth),
            "--hidden_units", str(HIDDEN_UNITS),
            "--max_length", str(MAX_LENGTH),
            "--max_epochs", str(epochs),
            "--batch_size", BATCH_SIZE,
            "--lr", "0.0001",
            "--patience", "10",
            "--save_dir", save_dir
        ]

        if current_pretrain:
            train_cmd.extend(["--pretrain_path", current_pretrain])

        run_cmd(train_cmd, log_file=log_path)

        eval_log_path = os.path.join(RUN_DIR, f"eval_metrics_{depth}b.txt")
        eval_cmd = [
            "python", "eval.py",
            "--data_path", DATASET_PATH,
            "--model_path", os.path.join(save_dir, "best_model.pt"),
            "--num_blocks", str(depth),
            "--hidden_units", str(HIDDEN_UNITS),
            "--maxlen", str(MAX_LENGTH),
        ]

        # Для eval логов тоже дублируем вывод на экран и в файл
        run_cmd(eval_cmd, log_file=eval_log_path)

        if not is_final:
            next_depth = DEPTHS[i + 1]
            stacked_model_path = os.path.join(RUN_DIR, f"stacked_{next_depth}b.pt")

            print(f"\n🧬 [STACKING] Размножаем веса: {depth} -> {next_depth} блоков")
            stack_cmd = [
                "python", "./sasrec/stacking_utils.py",
                "--input", os.path.join(save_dir, "best_model.pt"),
                "--output", stacked_model_path,
                "--method", "adjacent"
            ]
            run_cmd(stack_cmd)
            current_pretrain = stacked_model_path

    print(f"\n🎉 ВСЕ ЭТАПЫ УСПЕШНО ЗАВЕРШЕНЫ! Итоговая модель на {DEPTHS[-1]} блоков ждет в {RUN_DIR}")

if __name__ == "__main__":
    main()