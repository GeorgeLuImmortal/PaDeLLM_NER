import json
from collections import defaultdict
import pandas as pd
import argparse

# Function to remove the entry with the lower score
def remove_lower_score_entry(pred, value_text):
    scores = [(key, item[0][1]) for key, item in pred.items() if item and item[0][0] == value_text]
    scores = []

    for key, items in pred.items():
        for item in items:
            mention = item[0]
            if mention == value_text:
                scores.append((key, item[1]))

    print(value_text, scores)
    if scores:
        max_score_key = max(scores, key=lambda x: x[1])[0]

        for key, values in pred.items():
            if key != max_score_key:
                new_values = []
                for value in values:
                    if value[0] == value_text:
                        print("value to be removed")
                        print(value)
                        pass
                    else:
                        new_values.append(value)
                pred[key] = new_values
   
    print('== pred after remove')
    print(pred)
    return pred


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

    with open(f'./results/{dataset}_{model_size}_{identifier}.json', 'r') as fin:
        test_data = json.load(fin)

    output_path = f'./results/{dataset}_{model_size}_{identifier}_dedu.json'
    
    for example in test_data:
        value_key_dict = {}
        pred = example["pred"]
        dedu_values = []

        for key, values in pred.items():
            for value in values:
                if value[0] not in value_key_dict:
                    value_key_dict[value[0]] = [key]
                else:
                    value_key_dict[value[0]].append(key)

        for value, keys in value_key_dict.items():
            value_key_dict[value] = list(set(keys))

        for value, keys in value_key_dict.items():
            if len(keys) > 1:
                dedu_values.append(value)

        
        if len(dedu_values) > 0:

            print('===========')
            print(example["pred"])

            print(dedu_values)
            for value in dedu_values:
                example["pred"] = remove_lower_score_entry(example["pred"], value)


    json.dump(test_data, open(output_path, "w", encoding="utf-8"), indent=4, ensure_ascii=False)
