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
        default='weibo',
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
        default="dsot_bs128_e4_v1.011",
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
        default="0",
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
    model_path = f"JinghuiLuAstronaut/PaDeLLM_baichuan2_{model_size}_{dataset}"
    test_path = f"./ner_data/{dataset}/{dataset}_test.json"
    output_path = f'./results/{dataset}_{model_size}_{identifier}_{sample_method}_single_run{run}.json'

    test_data = []
    with open(test_path, "r") as fin:
        for line in fin.readlines():
            test_data.append(json.loads(line))

    print('====')
    print(model_path)
    print(test_data[0])
    print('====')

    if dataset == "ecommerce":
        mapping = {'HC':'商品','HP':'品牌'}
    elif dataset == "youku":
        mapping = {'TELEVISION':'电视剧','PER':'明星','MISC':'其他'}
    elif dataset == "resume":
        mapping = {'NAME':"名称", 'CONT':"国籍", 'RACE':"民族", 'TITLE':"职位", 'EDU':"学历", 'ORG':"公司", 'PRO':"专业", 'LOC':"籍贯"}
    elif dataset == "weibo":
        mapping = {'PER.NAM':'名称特指','PER.NOM':'名称代称','GPE.NAM':'行政区特指','GPE.NOM':'行政区代称','LOC.NOM':'地点代称','LOC.NAM':'地点特指','ORG.NOM':'组织代称','ORG.NAM':'组织特指'} # v1.011
        # mapping = {'PER.NAM':'[目标字段1]','PER.NOM':'[目标字段2]','GPE.NAM':'[目标字段3]','GPE.NOM':'[目标字段4]','LOC.NOM':'[目标字段5]','LOC.NAM':'[目标字段6]','ORG.NOM':'[目标字段7]','ORG.NAM':'[目标字段8]'}
        # mapping = {'PER.NAM':'名称详指','PER.NOM':'名称泛指','GPE.NAM':'地缘详指','GPE.NOM':'地缘泛指','LOC.NOM':'地点泛指','LOC.NAM':'地点详指','ORG.NOM':'组织泛指','ORG.NAM':'组织详指'}
    elif dataset == "ontonotes":
        mapping = {'PER':"名称", 'LOC':"地点", 'ORG': "组织", 'GPE': "地缘"}
        # mapping = {'PER':"名称", 'LOC':"地点", 'ORG': "组织", 'GPE': "行政区"} # v1.011
    elif dataset == "msra":
        mapping = {'PER':"名称", 'LOC':"地点", 'ORG':"组织"}

    keys = list(mapping.values())
    print(mapping)
    print('====')

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        trust_remote_code = True, 
        padding_side = 'left'
        )
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        trust_remote_code = True
        ).to(device)


    if sample_method == 'greedy':
        default_generate_params = greedy_sample
    else:
        default_generate_params = nucleus_sample

    
    input_template = "文本:\n{texts}\n指定NER标签:\n{trg_key}\n\n<数量>"
    # input_template = "任务:\n1.统计文本中指定NER标签的文段数量。\n2.预测第k个属于指定NER标签的文段。\n\n特殊符号定义:\n<数量>:表示文段数量的预测开始。\n<第k文段>:表示第k个文段的预测开始。\n\n文本:\n{texts}\n指定NER标签:\n{trg_key}\n\n<数量>"
    key_template = "{trg_key}\n\n<数量>"
  
    
    test_preds = []
    sep_id = 5 # "\n" id for baichuan2 tokenizer
    eos_id = tokenizer.eos_token_id

    with torch.no_grad():
        
        output_idx = []
        for example_id, test_example in enumerate(tqdm(test_data)):

            example_latency = 0.0
            texts = test_example[0]

            inputs = [] 

            for trg_key in keys:
                input_data = {"texts":texts, "trg_key":trg_key}
                model_input = input_template.format(**input_data)
                inputs.append(model_input)

            ## batch inference for num_keys
            prefix_input_ids = tokenizer(
                inputs, 
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
                    example_latency += latency

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
                    example_latency += latency

                current_idx = output_num_mentions.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)

                if step_idx == 0:
                    output_idx = current_idx
                else:
                    output_idx = torch.cat((output_idx, current_idx), dim=-1)
                
                kv_caches = output_num_mentions.past_key_values
                
                all_done = True
                for sequence in output_idx:
                    if sep_id not in sequence and eos_id not in sequence:
                        all_done = False
                        break
        
                if all_done:
                    break
                else:
                    prefix_input_ids = torch.cat(
                        (prefix_input_ids, 
                        current_idx
                        ), dim=1)
                
                num_mentions = tokenizer.batch_decode(output_idx, skip_special_tokens=True)
                print(num_mentions)
                

            num_mention_pred = {}
            for trg_key, num in zip(keys,num_mentions):
                num_mention_pred[trg_key] = num

            final_inputs = []
            for trg_key, num_mentions in num_mention_pred.items():

                if num_mentions != "0":
                    try:
                        ## sanity check
                        num_mentions = str(int(num_mentions))
                    except Exception as e:
                        num_mentions = str(int(num_mentions[0]))

                    for order in range(int(num_mentions)):
                        input_data = {"texts":texts, "trg_key":trg_key}
                        model_input = input_template.format(**input_data)+num_mentions+f"\n<第{order+1}文段>"
                        final_inputs.append(model_input)

            example_pred = {}

            if len(final_inputs) == 0:
                ## no entity exists
                pass
            else:
                final_input_ids = tokenizer(final_inputs, return_tensors='pt', padding=True)['input_ids'].to(model.device)
                
                torch.cuda.synchronize()
                value_start_time = time.time()

                final_preds = model.generate(
                        final_input_ids, 
                        return_dict_in_generate=True, 
                        output_scores=True, 
                        **default_generate_params
                        )

                torch.cuda.synchronize()
                value_latency = time.time() - value_start_time
                example_latency += value_latency

                transition_scores = model.compute_transition_scores(
                        final_preds.sequences, 
                        final_preds.scores, 
                        normalize_logits=True
                        )

                 ## compute scores of new generated tokens by summing all logit scores except for the eos token
                new_token_scores = torch.sum(transition_scores, dim=1).cpu()
                new_token_prob = np.exp(new_token_scores)
                pred_texts = tokenizer.batch_decode(final_preds.sequences.cpu(), skip_special_tokens=True)

                for pred, prob in zip(pred_texts, new_token_prob):
                    trg_key = pred.split("指定NER标签:\n")[1].split("\n\n", 1)[0]
                    mention = pred.split("指定NER标签:\n")[1].split("\n\n", 1)[1].split("文段>")[1]
                    if trg_key in example_pred:
                        example_pred[trg_key].append((mention, prob.item()))
                    else:
                        example_pred[trg_key] = [(mention, prob.item())]


            gt = {}
            if len(test_example) > 1:

                for item in test_example[1:]:
                    output_k, output_v = mapping[item[3]], item[2]
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


            test_preds.append({"id":example_id, "texts": texts, "pred": example_pred, "gt": gt, "latency":example_latency})
            json.dump(test_preds, open(output_path, "w", encoding="utf-8"), indent=4, ensure_ascii=False)