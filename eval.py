import json
from fact.fact_verification import FactVerification
from utils import prompt_wrapper
from utils import util_tool
import wandb
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, cohen_kappa_score
import argparse
from tqdm import tqdm
import os
from itertools import zip_longest
import random
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--dataset', type=str, default='HybriDial', choices=['OpendialKG', 'HybriDial'])
parser.add_argument('--split', type=str, default='test', choices=['train', 'test', 'valid'])
parser.add_argument('--model', type=str)
parser.add_argument('--batch_size', type=int, default=5)
parser.add_argument('--temperature', type=float, default=1)
parser.add_argument('--mode', type=str, default='plain', choices=['cot', 'plain', "few-shot", "few-shot-cot"])
parser.add_argument('--distill_mode', type=str, default='plain', choices=['cot', 'plain', 'none'])
parser.add_argument('--train_output_root_dir', type=str, default='')
parser.add_argument('--use_ckpt', action='store_true')
parser.add_argument('--skip_non_factual_claim', action='store_true')
args = parser.parse_args()

def set_global_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 如果有多个GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def is_valid(sample, args):
    if sample["factualLabel"] is None:
        return False

    if getattr(args, "skip_non_factual_claim", False):
        return bool(sample.get("isFactualClaim") == "verifiable")

    return True

if __name__ == '__main__':
    set_global_seeds(args.seed)
    data_path = "data"
    input_path = "%s/%s/annotated_%s_data.json" % (data_path, args.dataset, args.split)
    output_directory = "output/%s" % (args.dataset)
    if (not os.path.exists(output_directory)):
        os.makedirs(output_directory)
    output_path = "output/%s/%s_%s_%s_%s_%s.json" % (args.dataset, args.split, args.model, args.mode, args.seed, args.temperature)
    if args.use_ckpt:
        output_path = output_path.replace(".json", "_ck_dis(%s).json"%args.distill_mode)
        fact_verification = FactVerification(args.model, args.train_output_root_dir)
    else:
        fact_verification = FactVerification(args.model)
    wandb.init(
        # set the wandb project where this run will be logged
        project="DialEval",
        name="%s"%(args.dataset),
        # track hyperparameters and run metadata
        config={
            "dataset": args.dataset,
            "batch_size": args.batch_size,
            "model": args.model,
            "mode": args.mode,
            "ckpt": args.use_ckpt,
            "train_output_root_dir": args.train_output_root_dir,
            "distill_mode": args.distill_mode,
            "seed": args.seed,
            "temperature": args.temperature,
            "skip_non_factual_claim": args.skip_non_factual_claim
        })
    data = util_tool.read_json_file(input_path)
    history_list = [sample['history'] for sample in data if is_valid(sample, args)]
    question_list = [sample['A'] for sample in data if is_valid(sample, args)]
    last_utterance_list = [sample['atomic_fact'] for sample in data if is_valid(sample, args)]
    context_list = [prompt_wrapper.extract_context(history, 8) for history in history_list]
    dialogue_list = ["%s\nA: %s"%(context, question) for context, question in zip(context_list, question_list)]
    selected_evidence_list = [
        [
            evidence
            for evidence in sample["evidence"]
            if evidence["selected"] == True
        ]
        for sample in data
        if is_valid(sample, args)
    ]
    evidence_list = [sample['evidence'] for sample in data if is_valid(sample, args)]
    retrieved_evidence_list = []
    ground_truth_list = [sample['factualLabel'] for sample in data if is_valid(sample, args)]
    dialogue_batch = [dialogue_list[i:i + args.batch_size] for i in range(0, len(dialogue_list), args.batch_size)]
    last_utterance_batch = [last_utterance_list[i:i + args.batch_size] for i in range(0, len(last_utterance_list), args.batch_size)]
    selected_evidence_batch = [selected_evidence_list[i:i + args.batch_size] for i in range(0, len(selected_evidence_list), args.batch_size)]
    predicted_results = list()
    predicted_explanations = list()
    prompts_list = list()
    system_prompts_list = list()
    for dialogue, evidence, last_utterance in tqdm(zip(dialogue_batch, selected_evidence_batch, last_utterance_batch), total=len(dialogue_batch)):
        prediction_list, explanation_list, prompt_list, system_prompt_list = fact_verification.verify(dialogue, last_utterance, evidence, args.mode, args.temperature)
        predicted_explanations.extend(explanation_list)
        predicted_results.extend(prediction_list)
        prompts_list.extend(prompt_list)
        system_prompts_list.extend(system_prompt_list)
    output = [{"dialogue": dialogue, "last_utterance": last_utterance, "ground_truth": ground_truth, "prediction": prediction, "explanation": explanation, "selected_evidence":selected_evidence,
               "prompt": prompt, "sys_prompt":system_prompt}
                         for dialogue, last_utterance, ground_truth, prediction, explanation, selected_evidence, prompt, system_prompt
                         in zip_longest(dialogue_list, last_utterance_list, ground_truth_list, predicted_results, predicted_explanations, selected_evidence_list, prompts_list, system_prompts_list, fillvalue="EMPTY")]
    util_tool.write_output(output_path, json.dumps(output), "w")
    ground_truth_list = [x.lower() for x in ground_truth_list]
    predicted_results = [x.lower() for x in predicted_results]
    accuracy = accuracy_score(ground_truth_list, predicted_results)
    kappa = cohen_kappa_score(ground_truth_list, predicted_results)
    precision = precision_score(ground_truth_list, predicted_results, average='macro')
    recall = recall_score(ground_truth_list, predicted_results, average='macro')
    f1 = f1_score(ground_truth_list, predicted_results, average='macro')
    all_metrics = {}
    all_metrics['f_precision'] = precision
    all_metrics['f_recall'] = recall
    all_metrics['f_f1'] = f1
    all_metrics['accuracy'] = accuracy
    all_metrics['kappa'] = kappa
    wandb.log(all_metrics)


