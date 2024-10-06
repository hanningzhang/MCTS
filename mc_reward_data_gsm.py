import json
import re
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Optional
from vllm import LLM, SamplingParams
# from eval_gsm8k import is_number, extract_answer_number
# from eval_math import remove_boxed, process_results
import util
import time
import os
from util import is_number, extract_answer_number, remove_boxed, process_results
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, pipeline


@dataclass
class ScriptArguments:
    completion_model_name_or_path: str = field(default="", metadata={"help": "the completion model name or path locally or from huggingface."})
    dataset_path: str = field(default="", metadata={"help": "dataset path for generator data."})
    output_dir: str = field(default="mc_data",metadata={"help":"location to store the PRM data."})
    tensor_parallel_size: int = field(default=1,metadata={"help":""})
    num_gpus: int = field(default=2)
    local_rank:int = field(default=0)
    sampling_num:int = field(default=16)
    split:int = field(default=0)
    batch_size: int = field(default=100)
    num_batches_per_save: int = field(
        default=10, metadata={"help": "save every {save_frequency} batches"}
    )

def process_dataset(raw_dataset):
    new_dataset = []
    for sample in tqdm(raw_dataset):
        new_dataset.append(sample['text'])
    return new_dataset

def check_math_answer(content,ground_truth):
    
    split_ans = content.split('The answer is: ')
    if len(split_ans) > 1:
        ans = split_ans[-1]
        extract_ans_temp = ans.split('ки')[0]
        extract_ans_temp = extract_ans_temp.strip()
        if len(extract_ans_temp)>0 and extract_ans_temp[-1] == '.':
            extract_ans = extract_ans_temp[0:-1]
        else:
            extract_ans = extract_ans_temp
        extract_ans = extract_ans.strip()
        
        gt_ans = remove_boxed(util.last_boxed_only_string(ground_truth))
        #print(f"gt ans is {gt_ans}")
        # gt_ans = ground_truth.split('The answer is: ')
        # gt = gt_ans[-1]
        # extract_gt_temp = gt.split('ки')[0]
        # extract_gt_temp = extract_gt_temp.strip()
        # if len(extract_gt_temp)>0 and extract_gt_temp[-1] == '.':
        #     extract_gt = extract_gt_temp[0:-1]
        # else:
        #     extract_gt = extract_gt_temp
        # extract_gt = extract_gt.strip()
        if util.is_equiv(extract_ans, gt_ans):
            return True
        else:
            return False
    else:
        return False
    
def check_answer(sample,content,ground_truth):

    if "GSM" in sample['task']:
        temp_ans = sample['ground_truth'].split('#### ')[1]
        temp_ans = int(temp_ans.replace(',', ''))
        #print(float(extract_answer_number(sample['ground_truth'])))
        if not extract_answer_number(content):
            return 1
        if float(extract_answer_number(content)) == float(temp_ans):
            label = 0
        else:
            label = 1
    else:
        if check_math_answer(content,sample['ground_truth']):
            label = 0
        else:
            label = 1
            
    return label
    

def generate_completion(llm,sampling_params,sample,prompt,ground_truth):
    
    #label = None    ## 0 denotes good completion. 1 denotes bad completion.
    label_list = []
    if isinstance(prompt, list):
        pass
    else:
        prompt = [prompt]

    completions = llm.generate(prompt, sampling_params)
    completions_list = []
    for output in completions:
        prompt_temp = output.prompt
        generated_text = [output.outputs[i].text for i in range(len(output.outputs))]
        completions_list.append([prompt_temp,generated_text])

    for completion in completions_list:
        results = [check_answer(sample,content,ground_truth) for content in completion[1]]
    
        if 0 in results:
            label = 0
        else:
            label = 1
        
        label_list.append({"prompt":completion[0],"label":label})
        
    return label_list

if __name__ == "__main__":
    
    parser = HfArgumentParser((ScriptArguments))
    args = parser.parse_args_into_dataclasses()[0]

    raw_dataset = load_dataset(args.dataset_path,split='train')
    dataset = process_dataset(raw_dataset)
    print("------------")
    print("begin to preprocess the sampling data")
    print("------------")
    
    #step_tag = 'ки' 
    processed_dataset = []
    if args.split == 0:
        dataset = dataset[:int(0.5*len(dataset))]
    else:
        dataset = dataset[int(0.5*len(dataset)):]
    dataset = dataset[int((args.local_rank)/args.num_gpus * len(dataset)):int((args.local_rank+1)/args.num_gpus * len(dataset))]
    for sample in tqdm(dataset):
        step_list = sample.split('\n')
        step_list = [i.strip() for i in step_list]
        temp_str = ""
        for i in range(len(step_list)-1):
            temp_str += step_list[i] + "\n"
            processed_dataset.append(temp_str)
        temp_str += step_list[-1]
        processed_dataset.append(temp_str)
    
    stop_tokens = []
    sampling_params = SamplingParams(n=args.sampling_num, temperature=1, top_p=1, max_tokens=1024, stop=stop_tokens)
    print('sampling =====', sampling_params)
    llm = LLM(model=args.completion_model_name_or_path,tensor_parallel_size=args.tensor_parallel_size, enforce_eager=True, dtype = "float16",gpu_memory_utilization=0.9,swap_space=32)
    print("------------")
    print("begin to label with markov process.")
    print("------------")
    
    prompt = [{"role":"user","content":i+"\nPlease reason step by step, and put your final answer within \\boxed{}."} for i in processed_dataset]

    tokenizer = llm.get_tokenizer()
    format_prompt = []
    for i in prompt[:]:
        conversations = tokenizer.apply_chat_template(
            [i],
            tokenize=False,
            add_generation_prompt=True, 
        )
        format_prompt.append(conversations)

    # Resumes from previous inference
    completions_list = []

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = f"{args.output_dir}/data_gsm_split{args.split}_{args.local_rank}.json"
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            completions_list = json.load(f)

    # Inferences
    num_sample = len(format_prompt)
    num_processed_sample = len(completions_list)
    batch_size = args.batch_size
    assert num_processed_sample % batch_size == 0

    for start_index in tqdm(range(num_processed_sample, num_sample, batch_size)):
        end_index = min(start_index + batch_size, num_sample)
        batch_prompt = format_prompt[start_index:end_index]
        completions = llm.generate(
            batch_prompt,
            sampling_params,
            use_tqdm=False,
        )

        for count, output in enumerate(completions):
            prompt_temp = output.prompt
            generated_text = [
                output.outputs[i].text for i in range(len(output.outputs))
            ]

            index = count + start_index
            if "The answer is" in processed_dataset[index]:
                completions_list.append({
                    "prompt": processed_dataset[index],
                    "completions": ["" for i in range(len(generated_text))]
                })
            else:
                completions_list.append({
                    "prompt": processed_dataset[index],
                    "completions": generated_text
                })

        # Saves results every {num_batches_per_save} batches
        num_batch_processed = start_index // batch_size + 1
        if num_batch_processed % args.num_batches_per_save == 0:
            with open(output_file, 'w') as f:
                json.dump(completions_list, f, indent=4, ensure_ascii=False)
