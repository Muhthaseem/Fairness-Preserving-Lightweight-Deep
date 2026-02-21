import pandas as pd
import os
from src.config import IDX_TO_GROUP

def analyze_csv(path):
    print(f"\nAnalyzing: {path}")
    df = pd.read_csv(path)
    total = len(df)
    real = len(df[df['label'] == 'REAL'])
    fake = len(df[df['label'] == 'FAKE'])
    print(f"Total: {total}, REAL: {real}, FAKE: {fake}")
    
    group_stats = []
    for g_id, g_name in IDX_TO_GROUP.items():
        g_df = df[df['group_id'] == g_id]
        g_total = len(g_df)
        g_real = len(g_df[g_df['label'] == 'REAL'])
        g_fake = len(g_df[g_df['label'] == 'FAKE'])
        
        group_stats.append({
            "Group": g_name,
            "Total": g_total,
            "REAL": g_real,
            "FAKE": g_fake,
            "RealRatio": g_real / g_total if g_total > 0 else 0
        })

    
    print(pd.DataFrame(group_stats).to_string(index=False))

analyze_csv("outputs/splits/val.csv")
analyze_csv("outputs/splits/test.csv")
