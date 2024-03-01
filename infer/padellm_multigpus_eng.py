from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
set_seed(42)
import torch
import json
from tqdm import tqdm
import os
import argparse
import time
import numpy as np
from utils import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default='conll03',
        help="dataset like weibo",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="7b",
        help="model_size like 7b",
    )
    parser.add_argument(
        "--identifier",
        default="dsot_bs128_e1_v1.01",
        type=str,
        help="identifier like bs128_e4_v1.03",
    )
    parser.add_argument(
        "--run",
        default="1",
        type=str,
        help="index of repetition",
    )
    parser.add_argument(
        "--cuda_num",
        default="1",
        type=str,
        help="index of gpu used",
    )
    parser.add_argument(
        "--sample_method",
        default="greedy",
        type=str,
        help="sample method",
    )

    args = parser.parse_args()

    dataset = args.dataset
    model_size = args.model_size
    identifier = args.identifier
    run = args.run
    cuda_num = args.cuda_num
    sample_method = args.sample_method

    device = f"cuda:{cuda_num}" if torch.cuda.is_available() else "cpu"
    model_path = f"JinghuiLuAstronaut/PaDeLLM_llama2_{model_size}_{dataset}"
    test_path = f"./ner_data/{dataset}/{dataset}_test.json"
    output_path = f'./results/{dataset}_{model_size}_{identifier}_{sample_method}_theory_run{run}.json'

    test_data = []
    with open(test_path, "r") as fin:
        for line in fin.readlines():
            test_data.append(json.loads(line))

    print('====')
    print(model_path)
    print(test_data[0])
    print('====')

    if dataset == "conll03":
        keys = ['ORG', 'MISC', 'PER', 'LOC']
    elif dataset == "genia":
        keys = ['DNA', 'RNA', 'cell_type', 'protein', 'cell_line']
    elif dataset == "ace05":
        keys = ['GPE','LOC','VEH','WEA','PER','ORG','FAC']

    print(keys)
    print('====')

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        trust_remote_code = True, 
        padding_side = 'left'
        )

    tokenizer.pad_token_id = 0
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        trust_remote_code = True
        ).to(device)
    model.eval()


    if sample_method == 'greedy':
        default_generate_params = greedy_sample
    else:
        default_generate_params = nucleus_sample
    

    input_template = "text:\n{texts}\nentity type:\n{trg_key}\n\n<num>"
    

    test_preds = []

    with torch.no_grad():

        for example_id, test_example in enumerate(tqdm(test_data[:])):


            texts = test_example[0]
            inputs = [] 
            key_latency = {}
            largest_latency = 0.0

            for trg_key in keys:
                input_data = {"texts":texts, "trg_key":trg_key}
                model_input = input_template.format(**input_data)
                inputs.append(model_input)

            ## sequence inference for num_keys
            num_mention_pred = {}
            
            sep_id = 13 # "\n id for llama2 tokenizer
            eos_id = tokenizer.eos_token_id
            for idx, input_ in enumerate(inputs):
                output_idx = []
                prefix_input_ids = tokenizer(
                    input_, 
                    return_tensors='pt', 
                    padding=True
                    )['input_ids'].to(model.device)


        
                for step_idx in range(10):

                    if step_idx == 0:

                        torch.cuda.synchronize()
                        start_time = time.time()

                        output_num_mentions = model(
                            input_ids = prefix_input_ids, 
                            use_cache=True
                            )

                        torch.cuda.synchronize()
                        latency = time.time() - start_time
                        if keys[idx] not in key_latency:
                            key_latency[keys[idx]] = latency
                        else:
                            key_latency[keys[idx]] += latency

                    else:
                        torch.cuda.synchronize()
                        start_time = time.time()

                        output_num_mentions = model(
                            input_ids = prefix_input_ids, 
                            use_cache=True, 
                            past_key_values = kv_caches
                            )

                        torch.cuda.synchronize()
                        latency = time.time() - start_time
                        if keys[idx] not in key_latency:
                            key_latency[keys[idx]] = latency
                        else:
                            key_latency[keys[idx]] += latency

                    current_idx = output_num_mentions.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)[0]
                    kv_caches = output_num_mentions.past_key_values
                    current_idx = current_idx.cpu().item()
           
                    if current_idx == sep_id or current_idx == eos_id:
                        break
                    else:
                        prefix_input_ids = torch.cat(
                            (prefix_input_ids, 
                            torch.tensor([[current_idx]]
                            ).to(model.device)), dim=1)

                        output_idx.append(current_idx)

                num_mentions = tokenizer.decode(output_idx, skip_special_tokens=True)
                num_mention_pred[keys[idx]] = num_mentions
                
            print(num_mention_pred)
            ## remove the key whose num is zero
            final_inputs = []
            for trg_key, num_mentions in num_mention_pred.items():
                if num_mentions != "0": 
                    try:
                        ## sanity check
                        num_mentions = str(int(num_mentions))
                    except Exception as e:
                        num_mentions = str(int(num_mentions[0]))

                    if int(num_mentions) > 100:
                        num_mentions = 10
                    for order in range(int(num_mentions)):
                        input_data = {"texts":texts, "trg_key":trg_key}
                        model_input = input_template.format(**input_data)+num_mentions+f"\n<mention {order+1}>"
                        final_inputs.append(model_input)

            example_pred = {}

            if len(final_inputs) == 0:
                ## no entity exists
                largest_latency = max(key_latency.values())
                pass
            else:
                for final_input in final_inputs:
                    final_input_ids = tokenizer(final_input, return_tensors='pt', padding=True)['input_ids'].to(model.device)
                    torch.cuda.synchronize()
                    start_time = time.time()
                    final_preds = model.generate(
                        final_input_ids, 
                        return_dict_in_generate=True, 
                        output_scores=True, 
                        **default_generate_params
                        )

                    transition_scores = model.compute_transition_scores(
                    final_preds.sequences, 
                    final_preds.scores, 
                    normalize_logits=True
                    )
                    
                    ## compute scores of new generated tokens by summing all logit scores except for the eos token
                    new_token_scores = torch.sum(transition_scores[0][:-1]).item()
                    new_token_prob = np.exp(new_token_scores)

                    trg_key = final_input.split("entity type:\n")[1].split("\n\n", 1)[0]
                    torch.cuda.synchronize()
                    latency = time.time() - start_time + key_latency[trg_key]
                    if latency > largest_latency:
                        largest_latency = latency
                    
                    input_length = final_input_ids.shape[1] ## input length
                    generated_tokens_id = final_preds.sequences[:, input_length:] ## new generated token id
                    pred = tokenizer.decode(generated_tokens_id.cpu()[0], skip_special_tokens=True)
                    
                    # trg_key = pred.split("指定NER标签:\n")[1].split("\n\n", 1)[0]
                    # mention = pred.split("entity type:\n")[1].split("\n\n", 1)[1].split(">",2)[-1]
                    mention = pred
                    if trg_key in example_pred:
                        example_pred[trg_key].append((mention, new_token_prob))
                    else:
                        example_pred[trg_key] = [(mention, new_token_prob)]

            gt = {}
            if len(test_example) > 1:

                for item in test_example[1:]:
                    output_k, output_v = item[3], item[2]
                    if output_k not in gt:
                        gt[output_k] = [output_v]
                    else:
                        gt[output_k].append(output_v)
            else:
                # gt no entity exists
                pass


            if len(final_inputs) == 0:
                del prefix_input_ids, output_num_mentions
            else:
                del prefix_input_ids, output_num_mentions, final_input_ids, final_preds
            torch.cuda.empty_cache()


            test_preds.append({"id":example_id, "texts": texts, "pred": example_pred, "gt": gt, "latency":largest_latency})
            json.dump(test_preds, open(output_path, "w", encoding="utf-8"), indent=4, ensure_ascii=False)
