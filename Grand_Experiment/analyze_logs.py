import os, re, glob
import pandas as pd

def to_md(df):
    if df.empty: return "Нет данных"
    cols = df.columns.tolist()
    res = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"]*len(cols)) + "|"]
    for _, row in df.iterrows():
        res.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(res)

def parse_log(filepath):
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            
            # ЖЕСТКАЯ ПРОВЕРКА НА NaN (регистронезависимая)
            if re.search(r'Loss:\s*nan', content, re.IGNORECASE) or "CUDA out of memory" in content:
                return {'val_ndcg': 0.0, 'test_ndcg': 0.0, 'test_hr': 0.0, 'status': 'Ошибка (OOM/NaN)'}
            
            val_ndcg_m = re.search(r'Best validation NDCG@10:\s+([\d\.]+)', content)
            test_sec = content.split('--- Test metrics @10 ---')
            test_ndcg, test_hr = 0.0, 0.0
            if len(test_sec) > 1:
                ndcg_m = re.search(r'NDCG@10:\s+([\d\.]+)', test_sec[1])
                hr_m = re.search(r'HR@10:\s+([\d\.]+)', test_sec[1])
                if ndcg_m: test_ndcg = float(ndcg_m.group(1))
                if hr_m: test_hr = float(hr_m.group(1))
            
            # ЗАЩИТА ОТ ИДЕАЛЬНЫХ ЕДИНИЦ (признак NaN-бага на валидации)
            if test_ndcg == 1.0 or val_ndcg_m and float(val_ndcg_m.group(1)) == 1.0:
                return {'val_ndcg': 0.0, 'test_ndcg': 0.0, 'test_hr': 0.0, 'status': 'Ошибка (1.000 Bug)'}

            return {
                'val_ndcg': float(val_ndcg_m.group(1)) if val_ndcg_m else 0,
                'test_ndcg': test_ndcg,
                'test_hr': test_hr,
                'status': 'Доучилась' if test_ndcg > 0 else 'Упала'
            }
    except Exception:
        return None

def get_optimal_param(df, param_col, threshold=0.015, default_val=None):
    if df.empty: return default_val
    df = df.sort_values(param_col).reset_index(drop=True)
    best_val = int(df[param_col].iloc[0])
    best_score = df['test_ndcg'].iloc[0]
    
    for i in range(1, len(df)):
        curr_val = int(df[param_col].iloc[i])
        curr_score = df['test_ndcg'].iloc[i]
        
        if best_score > 0:
            gain = (curr_score - best_score) / best_score
            if gain >= threshold:
                best_val = curr_val
                best_score = curr_score
        elif curr_score > 0: 
            best_val = curr_val
            best_score = curr_score
    return best_val

# --- АНАЛИЗ ГЛУБИНЫ ---
depth_files = glob.glob('./Grand_Experiment/1_ablation/depth/log_*b.txt')
d_res = []
for f in depth_files:
    d = int(re.search(r'log_(\d+)b', f).group(1))
    stats = parse_log(f)
    if stats: d_res.append({'Blocks': d, **stats})
df_d = pd.DataFrame(d_res)
best_blocks = get_optimal_param(df_d, 'Blocks', default_val=16)

with open('./Grand_Experiment/2_reports/0_Depth_Analysis.md', 'w', encoding='utf-8') as f:
    f.write(f"### Анализ Глубины\nОптимальная глубина: **{best_blocks}**\n\n")
    if not df_d.empty: f.write(to_md(df_d.sort_values('Blocks')))

# --- АНАЛИЗ ДЛИНЫ ---
len_files = glob.glob('./Grand_Experiment/1_ablation/length/log_*l.txt')
l_res = []
for f in len_files:
    l = int(re.search(r'log_(\d+)l', f).group(1))
    stats = parse_log(f)
    if stats: l_res.append({'Length': l, **stats})
df_l = pd.DataFrame(l_res)
best_length = get_optimal_param(df_l, 'Length', default_val=100)

with open('./Grand_Experiment/2_reports/1_Length_Analysis.md', 'w', encoding='utf-8') as f:
    f.write(f"### Анализ Контекста\nОптимальная длина: **{best_length}**\n\n")
    if not df_l.empty: f.write(to_md(df_l.sort_values('Length')))

# --- АНАЛИЗ ЭМБЕДДИНГОВ ---
emb_files = glob.glob('./Grand_Experiment/1_ablation/embeds/log_*e.txt')
e_res = []
for f in emb_files:
    e = int(re.search(r'log_(\d+)e', f).group(1))
    stats = parse_log(f)
    if stats: e_res.append({'Units': e, **stats})
df_e = pd.DataFrame(e_res)
best_embeds = get_optimal_param(df_e, 'Units', default_val=256)

with open('./Grand_Experiment/2_reports/2_Embed_Analysis.md', 'w', encoding='utf-8') as f:
    f.write(f"### Анализ Размерности\nОптимальная ширина: **{best_embeds}**\n\n")
    if not df_e.empty: f.write(to_md(df_e.sort_values('Units')))

# Экспорт лучших параметров
with open('./Grand_Experiment/best_params.env', 'w') as f:
    f.write(f"BEST_BLOCKS={best_blocks}\n")
    f.write(f"BEST_LENGTH={best_length}\n")
    f.write(f"BEST_EMBEDS={best_embeds}\n")
