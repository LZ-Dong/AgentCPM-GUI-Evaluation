"""
1. 输入为 /data1/home/donglingzhong/codespace/AgentCPM-GUI-Evaluation/cot_eval/data/AgentCPM-GUI/aitz_test/annotations.jsonl
条目举例{"id": "8353462066287083850:0", "gta": "1", "caa": "1", "error_code": "", "em": 1, "q1": 1, "q2": 0, "q3": 0, "q4": 0, "gta_sys": 1}
2. 取gta不为NA的所有条目，需要gta字段和gta_sys字段
3. gta为实际值，gta_sys为模型预测值
4. 计算gta_sys的混淆矩阵
"""

import argparse
import json
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, matthews_corrcoef
import numpy as np

def load_and_process_data(file_path):
    """
    加载JSONL文件并处理数据
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    
    # 转换为DataFrame
    df = pd.DataFrame(data)
    
    # 过滤gta不为NA的条目
    df_filtered = df[df['gta'] != 'NA'].copy()
    
    print(f"总条目数: {len(df)}")
    print(f"gta不为NA的条目数: {len(df_filtered)}")
    
    return df_filtered

def calculate_binary_classification_metrics(df):
    """
    计算二分类评估指标
    """
    # 提取真实值和预测值，转换为整数
    y_true = df['gta'].astype(int)
    y_pred = df['gta_sys'].astype(int)
    
    print("=== 二分类评估指标 ===\n")
    
    # 1. Accuracy：系统与人工判定完全一致的比例
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # 2. Precision / Recall / F1：把人工标注当作 gold，系统判定为预测
    precision = precision_score(y_true, y_pred, pos_label=1)
    recall = recall_score(y_true, y_pred, pos_label=1)
    f1 = f1_score(y_true, y_pred, pos_label=1)
    
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # 3. Confusion Matrix：四格统计
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    print(f"\nConfusion Matrix:")
    print("实际\\预测\t0\t1")
    print(f"0\t\t{cm[0][0]}\t{cm[0][1]}")
    print(f"1\t\t{cm[1][0]}\t{cm[1][1]}")
    
    # 详细解释混淆矩阵
    tn, fp, fn, tp = cm.ravel()
    print(f"\n混淆矩阵详解:")
    print(f"True Negatives (TN): {tn}  # 正确预测为0")
    print(f"False Positives (FP): {fp}  # 错误预测为1")
    print(f"False Negatives (FN): {fn}  # 错误预测为0")
    print(f"True Positives (TP): {tp}  # 正确预测为1")
    
    # 4. Cohen's κ (Kappa)：考虑随机一致性的修正
    kappa = cohen_kappa_score(y_true, y_pred)
    print(f"\nCohen's Kappa: {kappa:.4f}")
    if kappa >= 0.8:
        kappa_level = "almost perfect"
    elif kappa >= 0.6:
        kappa_level = "substantial"
    elif kappa >= 0.4:
        kappa_level = "moderate"
    elif kappa >= 0.2:
        kappa_level = "fair"
    else:
        kappa_level = "poor"
    print(f"Kappa解释: {kappa_level}")
    
    # 5. MCC (Matthews Correlation Coefficient)：在类别不平衡时更稳健
    mcc = matthews_corrcoef(y_true, y_pred)
    print(f"Matthews Correlation Coefficient (MCC): {mcc:.4f}")
    
    # 类别分布
    print(f"\n=== 数据分布 ===")
    print(f"真实标签分布:")
    print(f"  类别0: {sum(y_true == 0)} ({sum(y_true == 0)/len(y_true)*100:.1f}%)")
    print(f"  类别1: {sum(y_true == 1)} ({sum(y_true == 1)/len(y_true)*100:.1f}%)")
    
    print(f"预测标签分布:")
    print(f"  类别0: {sum(y_pred == 0)} ({sum(y_pred == 0)/len(y_pred)*100:.1f}%)")
    print(f"  类别1: {sum(y_pred == 1)} ({sum(y_pred == 1)/len(y_pred)*100:.1f}%)")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'kappa': kappa,
        'mcc': mcc,
        'confusion_matrix': cm
    }

def main():
    parser = argparse.ArgumentParser(description="计算二分类评估指标")
    parser.add_argument('--file_path', type=str, required=True, help='输入的JSONL文件路径')
    args = parser.parse_args()
    
    # 加载数据
    df = load_and_process_data(args.file_path)
    
    # 计算二分类评估指标
    metrics = calculate_binary_classification_metrics(df)

if __name__ == "__main__":
    main()

"""
python rq1.py --file_path /data1/home/donglingzhong/codespace/AgentCPM-GUI-Evaluation/cot_eval/data/AgentCPM-GUI/aitz_test/gta_strict_clean.jsonl
"""