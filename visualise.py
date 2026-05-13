import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob

# Настройки стиля
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'figure.figsize': (10, 6)})

BASE_DIR = "benchmark_results"
OUTPUT_DIR = "plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_summary_metrics(summary_csv):
    if not os.path.exists(summary_csv):
        print(f"File {summary_csv} not found.")
        return

    df = pd.read_csv(summary_csv)
    # Очистка данных от ошибок/NaN
    df = df[~df['Test NDCG@10'].astype(str).str.contains('OOM|Error|nan', na=False)]
    df['Test NDCG@10'] = pd.to_numeric(df['Test NDCG@10'])
    df['Test HR@10'] = pd.to_numeric(df['Test HR@10'])
    df['Value'] = pd.to_numeric(df['Value'])

    threads = df['Thread'].unique()

    for thread in threads:
        thread_df = df[df['Thread'] == thread].sort_values('Value')

        # График NDCG
        plt.figure()
        # Исправляем FutureWarning: добавляем hue='Value' и legend=False
        sns.barplot(data=thread_df, x='Value', y='Test NDCG@10', hue='Value', palette='viridis', legend=False)

        title_param = thread.split('_')[1] if '_' in thread else thread
        plt.title(f'Summary: NDCG@10 vs {title_param}')
        plt.xlabel(title_param.capitalize())
        plt.ylabel('NDCG@10')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"summary_{thread}_ndcg.png"))
        plt.close()


def plot_learning_curves():
    if not os.path.exists(BASE_DIR):
        return

    threads = [d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))]

    for thread in threads:
        thread_path = os.path.join(BASE_DIR, thread)
        csv_files = glob.glob(os.path.join(thread_path, "learning_curve_*.csv"))

        if not csv_files:
            continue

        # Сортируем файлы, чтобы легенда была по порядку (например, 2, 4, 8...)
        csv_files.sort()

        plt.figure()
        for file in csv_files:
            # Извлекаем значение параметра из имени файла
            label = os.path.basename(file).replace("learning_curve_", "").replace(".csv", "")
            df = pd.read_csv(file)
            plt.plot(df['Epoch'], df['Val_NDCG_10'], label=label, marker='o', markersize=4)

        plt.title(f'Learning Curves: {thread}')
        plt.xlabel('Epoch')
        plt.ylabel('Validation NDCG@10')
        plt.legend(title="Parameter Value", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"curves_{thread}_ndcg.png"))
        plt.close()

        # График Loss
        plt.figure()
        for file in csv_files:
            label = os.path.basename(file).replace("learning_curve_", "").replace(".csv", "")
            df = pd.read_csv(file)
            df = df.dropna(subset=['Loss'])
            plt.plot(df['Epoch'], df['Loss'], label=label)

        plt.title(f'Loss Curves: {thread}')
        plt.xlabel('Epoch')
        plt.ylabel('Training Loss')
        plt.legend(title="Parameter Value", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"curves_{thread}_loss.png"))
        plt.close()


if __name__ == "__main__":
    print("Generating plots (clean version)...")
    summary_path = os.path.join(BASE_DIR, "summary_metrics.csv")
    plot_summary_metrics(summary_path)
    plot_learning_curves()
    print(f"Success! Graphs saved to '{OUTPUT_DIR}/'. Warnings suppressed.")