import os
import re
import pandas as pd
import matplotlib.pyplot as plt

ROOT_DIR = "./Grand_Experiment"
SUMMARY_FILE = os.path.join(ROOT_DIR, "2_reports", "summary_metrics.csv")

summary_data = []

print("📊 Начинаем парсинг логов и генерацию графиков...")

# Проходим по всем папкам эксперимента
for root, dirs, files in os.walk(ROOT_DIR):
    for file in files:
        if file.startswith("log_") and file.endswith(".txt"):
            log_path = os.path.join(root, file)

            # Имена для новых файлов
            csv_name = file.replace("log_", "learning_curve_").replace(".txt", ".csv")
            png_name = file.replace("log_", "plot_").replace(".txt", ".png")

            csv_path = os.path.join(root, csv_name)
            png_path = os.path.join(root, png_name)

            epochs, losses, hrs, ndcgs = [], [], [], []
            test_hr, test_ndcg = "N/A", "N/A"

            # Парсинг
            with open(log_path, 'r', encoding='utf-8') as f:
                content = f.read()

                pattern = r'Epoch\s+(\d+)/\d+\s+\|\s+Loss:\s+([\d\.]+)\s+\|\s+Val HR@10:\s+([\d\.]+)\s+\|\s+Val NDCG@10:\s+([\d\.]+)'
                for match in re.finditer(pattern, content):
                    epochs.append(int(match.group(1)))
                    losses.append(float(match.group(2)))
                    hrs.append(float(match.group(3)))
                    ndcgs.append(float(match.group(4)))

                test_sec = content.split('--- Test metrics @10 ---')
                if len(test_sec) > 1:
                    ndcg_m = re.search(r'NDCG@10:\s+([\d\.]+)', test_sec[1])
                    hr_m = re.search(r'HR@10:\s+([\d\.]+)', test_sec[1])
                    if hr_m: test_hr = float(hr_m.group(1))
                    if ndcg_m: test_ndcg = float(ndcg_m.group(1))

            if epochs:
                # 1. Сохраняем learning_curve.csv
                df = pd.DataFrame({
                    'Epoch': epochs, 'Loss': losses, 'Val_HR_10': hrs, 'Val_NDCG_10': ndcgs
                })
                df.to_csv(csv_path, index=False)

                # Добавляем данные для итогового CSV
                summary_data.append([root.split('/')[-1], file, test_hr, test_ndcg])

                # 2. Графики
                plt.figure(figsize=(12, 5))

                # График Loss
                plt.subplot(1, 2, 1)
                plt.plot(epochs, losses, label='Train Loss', color='#FF5733', linewidth=2)
                plt.xlabel('Эпохи')
                plt.ylabel('Loss')
                plt.title(f'Падение ошибки ({file})')
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend()

                # График Val NDCG
                plt.subplot(1, 2, 2)
                plt.plot(epochs, ndcgs, label='Val NDCG@10', color='#3383FF', linewidth=2)
                plt.xlabel('Эпохи')
                plt.ylabel('NDCG@10')
                plt.title(f'Рост качества ({file})')
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend()

                plt.tight_layout()
                plt.savefig(png_path, dpi=300)
                plt.close()

                print(f"✅ Обработан: {file} -> Созданы .csv и .png")

# 3. Сохраняем сводный файл summary_metrics.csv
if summary_data:
    summary_df = pd.DataFrame(summary_data, columns=["Этап", "Log File", "Test HR@10", "Test NDCG@10"])

    summary_df.to_csv(SUMMARY_FILE, index=False)
    print(f"\n🏆 Сводный файл сохранен в: {SUMMARY_FILE}")
else:
    print("\n⚠️ Логов с эпохами не найдено. Обучение еще не началось?")