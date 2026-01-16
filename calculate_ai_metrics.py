"""
Script to calculate prediction scores and performance metrics for AI evaluation results.
"""

import os
import pandas as pd
from pathlib import Path


def normalize_boolean(value):
    if pd.isna(value):
        return None
    value_str = str(value).upper().strip()
    if value_str in ['TRUE', 'YES', '1', 'T', 'Y']:
        return True
    elif value_str in ['FALSE', 'NO', '0', 'F', 'N']:
        return False
    return None


def calculate_metrics(df):
    df = df.copy()
    df['is_actually_fake'] = df['is_actually_fake'].apply(normalize_boolean)
    df['is_fake'] = df['is_fake'].apply(normalize_boolean)

    df = df.dropna(subset=['is_actually_fake', 'is_fake'])
    
    if len(df) == 0:
        return {
            'correct': 0,
            'incorrect': 0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0
        }

    tp = len(df[(df['is_actually_fake'] == True) & (df['is_fake'] == True)])
    tn = len(df[(df['is_actually_fake'] == False) & (df['is_fake'] == False)])
    fp = len(df[(df['is_actually_fake'] == False) & (df['is_fake'] == True)])
    fn = len(df[(df['is_actually_fake'] == True) & (df['is_fake'] == False)])

    correct = tp + tn
    incorrect = fp + fn

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'correct': correct,
        'incorrect': incorrect,
        'precision': round(precision, 3),
        'recall': round(recall, 3),
        'f1': round(f1, 3)
    }


def process_evaluation_files(evaluation_dir='ai-evaluation'):
    evaluation_path = Path(evaluation_dir)
    
    if not evaluation_path.exists():
        print(f"Error: Directory {evaluation_dir} does not exist")
        return None

    all_baseline = []
    all_hlq = []
    
    for csv_file in evaluation_path.glob('*_evaluation.csv'):
        print(f"Processing: {csv_file.name}")
        df = pd.read_csv(csv_file, encoding='utf-8-sig')

        baseline_df = df[df['iteration'] == 1].copy()
        hlq_df = df[df['iteration'] == 2].copy()
        
        all_baseline.append(baseline_df)
        all_hlq.append(hlq_df)

    if not all_baseline and not all_hlq:
        print("No evaluation data found")
        return None
    
    baseline_combined = pd.concat(all_baseline, ignore_index=True) if all_baseline else pd.DataFrame()
    hlq_combined = pd.concat(all_hlq, ignore_index=True) if all_hlq else pd.DataFrame()

    baseline_metrics = calculate_metrics(baseline_combined) if len(baseline_combined) > 0 else {
        'correct': 0, 'incorrect': 0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0
    }
    hlq_metrics = calculate_metrics(hlq_combined) if len(hlq_combined) > 0 else {
        'correct': 0, 'incorrect': 0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0
    }
    
    return {
        'baseline': baseline_metrics,
        'hlq': hlq_metrics
    }


def print_tables(metrics):
    if not metrics:
        print("No metrics to display")
        return

    print("\n" + "="*60)
    print("Prediction scores")
    print("="*60)
    print(f"{'Approach':<15} {'Correct':<10} {'Incorrect':<10}")
    print("-"*60)
    print(f"{'Baseline':<15} {metrics['baseline']['correct']:<10} {metrics['baseline']['incorrect']:<10}")
    print(f"{'HLQ':<15} {metrics['hlq']['correct']:<10} {metrics['hlq']['incorrect']:<10}")

    print("\n" + "="*60)
    print("Performance metrics")
    print("="*60)
    print(f"{'Approach':<15} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-"*60)
    print(f"{'Baseline':<15} {metrics['baseline']['precision']:<12} {metrics['baseline']['recall']:<12} {metrics['baseline']['f1']:<12}")
    print(f"{'HLQ':<15} {metrics['hlq']['precision']:<12} {metrics['hlq']['recall']:<12} {metrics['hlq']['f1']:<12}")
    print("="*60 + "\n")


def save_tables_to_csv(metrics, output_dir='analysis'):
    os.makedirs(output_dir, exist_ok=True)

    prediction_data = {
        'Approach': ['Baseline', 'HLQ'],
        'Correct': [metrics['baseline']['correct'], metrics['hlq']['correct']],
        'Incorrect': [metrics['baseline']['incorrect'], metrics['hlq']['incorrect']]
    }
    df_predictions = pd.DataFrame(prediction_data)
    prediction_path = os.path.join(output_dir, 'ai_prediction_scores.csv')
    df_predictions.to_csv(prediction_path, index=False, encoding='utf-8-sig')
    print(f"Saved prediction scores to: {prediction_path}")

    performance_data = {
        'Approach': ['Baseline', 'HLQ'],
        'Precision': [metrics['baseline']['precision'], metrics['hlq']['precision']],
        'Recall': [metrics['baseline']['recall'], metrics['hlq']['recall']],
        'F1': [metrics['baseline']['f1'], metrics['hlq']['f1']]
    }
    df_performance = pd.DataFrame(performance_data)
    performance_path = os.path.join(output_dir, 'ai_performance_metrics.csv')
    df_performance.to_csv(performance_path, index=False, encoding='utf-8-sig')
    print(f"Saved performance metrics to: {performance_path}")


def main():
    print("Calculating evaluation metrics...")
    print("Reading evaluation files from: ai-evaluation/")
    
    metrics = process_evaluation_files('ai-evaluation')
    
    if metrics:
        print_tables(metrics)
        save_tables_to_csv(metrics, 'analysis')

        print(f"\nSummary:")
        print(f"  Baseline: {metrics['baseline']['correct']} correct, {metrics['baseline']['incorrect']} incorrect")
        print(f"  HLQ: {metrics['hlq']['correct']} correct, {metrics['hlq']['incorrect']} incorrect")
        print(f"  Total evaluations: {metrics['baseline']['correct'] + metrics['baseline']['incorrect'] + metrics['hlq']['correct'] + metrics['hlq']['incorrect']}")
    else:
        print("Failed to calculate metrics")


if __name__ == "__main__":
    main()
