import json
from collections import defaultdict
import pandas as pd
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="dataset like weibo",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        help="model_size like 7b",
    )
    parser.add_argument(
        "--identifier",
        type=str,
        help="identifier like bs128_e4_v1.03",
    )

    args = parser.parse_args()

    dataset = args.dataset
    model_size = args.model_size
    identifier = args.identifier

    print('==========')
    print(f"start evaluating {dataset}_{model_size}_{identifier}")

    with open(f'/mnt/bn/lujinghui-nas-lq/NER_dsot_inference/run_cn_scripts/{dataset}/{dataset}_{model_size}_{identifier}.json', 'r') as fin:
        test_data = json.load(fin)

    # Initialize counters
    TP, FP, FN = 0, 0, 0
    per_example_scores = []

    # Function to calculate TP, FP, FN for each class
    def calculate_tpfptn(pred, gt):
        tp = sum([1 for p in pred if p in gt])
        fp = sum([1 for p in pred if p not in gt])
        fn = sum([1 for g in gt if g not in pred])
        return tp, fp, fn

    # Calculating TP, FP, FN for each example
    for example in test_data:
        example_tp, example_fp, example_fn = 0, 0, 0

        for class_key in set(example['pred'].keys()).union(example['gt'].keys()):
            gt_set = set(example["gt"].get(class_key, []))
            pred_set = example["pred"].get(class_key, [])
            pred_set = set([item[0] for item in pred_set])
   

            tp, fp, fn = calculate_tpfptn(pred_set, gt_set)

            example_tp += tp
            example_fp += fp
            example_fn += fn
            
            TP += tp
            FP += fp
            FN += fn

        # Calculate F-score for each example
        precision = example_tp / (example_tp + example_fp) if example_tp + example_fp > 0 else 0
        recall = example_tp / (example_tp + example_fn) if example_tp + example_fn > 0 else 0
        f_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        per_example_scores.append({"p":precision,"r":recall,"f":f_score})



    # Micro F-score
    micro_precision = TP / (TP + FP) if TP + FP > 0 else 0
    micro_recall = TP / (TP + FN) if TP + FN > 0 else 0
    micro_f_score = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if micro_precision + micro_recall > 0 else 0

    for line, score in zip(test_data, per_example_scores):
        line['score'] = score

    json.dump(test_data, open(f'/mnt/bn/lujinghui-nas-lq/NER_dsot_inference/run_cn_scripts/{dataset}/{dataset}_{model_size}_{identifier}_wscore.json', "w"), indent=4, ensure_ascii=False)


    # Reinitializing counts for each class
    class_counts = defaultdict(lambda: {'TP': 0, 'FP': 0, 'FN': 0})

    # Calculating TP, FP, FN for each class
    for example in test_data:
        for class_key in set(example['pred'].keys()).union(example['gt'].keys()):
            gt_set = set(example['gt'].get(class_key, []))
            pred_set = example["pred"].get(class_key, [])
            pred_set = set([item[0] for item in pred_set])
            
            tp, fp, fn = calculate_tpfptn(pred_set, gt_set)
            class_counts[class_key]['TP'] += tp
            class_counts[class_key]['FP'] += fp
            class_counts[class_key]['FN'] += fn

    print(class_counts)

    # Calculating F-score for each class
    class_f_scores = {}
    class_p_scores = {}
    class_r_scores = {}
    for class_key, counts in class_counts.items():
        precision = counts['TP'] / (counts['TP'] + counts['FP']) if counts['TP'] + counts['FP'] > 0 else 0
        recall = counts['TP'] / (counts['TP'] + counts['FN']) if counts['TP'] + counts['FN'] > 0 else 0
        f_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        class_p_scores[class_key] = precision
        class_r_scores[class_key] = recall
        class_f_scores[class_key] = f_score

    # Macro F-score (average of F-scores across all classes)
    macro_f_score = sum(class_f_scores.values()) / len(class_f_scores) if class_f_scores else 0
    macro_p = sum(class_p_scores.values()) / len(class_p_scores) if class_p_scores else 0
    macro_r = sum(class_r_scores.values()) / len(class_r_scores) if class_r_scores else 0

    class_f_scores['micro_f1'] = micro_f_score
    class_f_scores['macro_f1'] = macro_f_score
    class_f_scores['macro_p'] = macro_p
    class_f_scores['macro_r'] = macro_r

    for k, v in class_f_scores.items():
        class_f_scores[k] = [v]

    df = pd.DataFrame(class_f_scores).T
    df.to_csv(f"./{dataset}/{dataset}_{model_size}_{identifier}.csv")