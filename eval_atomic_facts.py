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
import jsonlines
from collections import Counter
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

if __name__ == '__main__':
    set_global_seeds(args.seed)
    data_path = "data"
    input_path = "%s/%s/GPT_annotated_test_data.json" % (data_path, args.dataset)
    output_directory = "output/%s" % (args.dataset)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    output_path = "output/%s/atomic_facts_%s_%s_%s_%s_%s.json" % (
        args.dataset, args.split, args.model, args.mode, args.seed, args.temperature
    )
    if args.use_ckpt:
        output_path = output_path.replace(".json", "_ck_dis(%s).json" % args.distill_mode)
        fact_verification = FactVerification(args.model, args.train_output_root_dir)
    else:
        fact_verification = FactVerification(args.model)

    wandb.init(
        project="DialEval",
        name="%s" % (args.dataset),
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
            "input_path": input_path
        })

    with jsonlines.open(input_path) as reader:
        data = [x for x in reader]

    dialogue_list = []
    last_utterance_list = []
    selected_evidence_list = []
    ground_truth_list = []
    llm_name_list = []

    output_dialogues = []

    for idx, sample in enumerate(data):
        A = sample.get("A", "")
        history = sample.get("history", [])
        context = prompt_wrapper.extract_context(history, 8)

        dialogue_entry = {
            "dialogue": "%s\nA: %s" % (context, A),
            "atomic_facts": {llm_name: [] for llm_name in ['plain', 'Meta-Llama-3.1-8B-Instruct', 'flan-t5-xxl']}
        }

        for llm_name in dialogue_entry["atomic_facts"].keys():
            for atomic_fact in sample[llm_name]['atomic_facts'][0]:
                if ('predicted_factual_claim' in atomic_fact and
                    atomic_fact['predicted_factual_claim'] is True and
                    'predicted_factual_label' in atomic_fact):

                    evidence = [e for e in atomic_fact["evidence"] if e["predicted_selection"] is True]

                    dialogue_list.append("%s\nA: %s" % (context, A))
                    last_utterance_list.append(atomic_fact['atomic_fact'])
                    selected_evidence_list.append(evidence)
                    ground_truth_list.append(atomic_fact['predicted_factual_label'])
                    llm_name_list.append(llm_name)

                    dialogue_entry["atomic_facts"][llm_name].append({
                        "last_utterance": atomic_fact['atomic_fact'],
                        "ground_truth": atomic_fact['predicted_factual_label'],
                        "selected_evidence": evidence,
                        "prediction": None,
                        "explanation": None,
                        "prompt": None,
                        "sys_prompt": None
                    })

        output_dialogues.append(dialogue_entry)

    dialogue_batch = [dialogue_list[i:i + args.batch_size] for i in range(0, len(dialogue_list), args.batch_size)]
    last_utterance_batch = [last_utterance_list[i:i + args.batch_size] for i in range(0, len(last_utterance_list), args.batch_size)]
    selected_evidence_batch = [selected_evidence_list[i:i + args.batch_size] for i in range(0, len(selected_evidence_list), args.batch_size)]

    predicted_results = []
    predicted_explanations = []
    prompts_list = []
    system_prompts_list = []

    for dialogue, evidence, last_utterance in tqdm(
        zip(dialogue_batch, selected_evidence_batch, last_utterance_batch),
        total=len(dialogue_batch)
    ):
        prediction_list, explanation_list, prompt_list, system_prompt_list = fact_verification.verify(
            dialogue, last_utterance, evidence, args.mode, args.temperature
        )
        predicted_explanations.extend(explanation_list)
        predicted_results.extend(prediction_list)
        prompts_list.extend(prompt_list)
        system_prompts_list.extend(system_prompt_list)

    fact_idx = 0
    for dialogue in output_dialogues:
        for llm_name, facts in dialogue["atomic_facts"].items():
            for atomic_fact in facts:
                atomic_fact["prediction"] = predicted_results[fact_idx]
                atomic_fact["explanation"] = predicted_explanations[fact_idx]
                atomic_fact["prompt"] = prompts_list[fact_idx]
                atomic_fact["sys_prompt"] = system_prompts_list[fact_idx]
                fact_idx += 1

    counts = Counter(predicted_results)
    print("counts:", counts)

    total = len(predicted_results)
    ratios = {k + "_portion": v / total for k, v in counts.items()}
    print("portion:", ratios)

    ground_truth_list = [x.lower() for x in ground_truth_list]
    predicted_results = [x.lower() for x in predicted_results]

    accuracy = accuracy_score(ground_truth_list, predicted_results)
    kappa = cohen_kappa_score(ground_truth_list, predicted_results)
    precision = precision_score(ground_truth_list, predicted_results, average='macro')
    recall = recall_score(ground_truth_list, predicted_results, average='macro')
    f1 = f1_score(ground_truth_list, predicted_results, average='macro')

    all_metrics = {}
    all_metrics.update(counts)
    all_metrics.update(ratios)
    all_metrics['f_precision'] = precision
    all_metrics['f_recall'] = recall
    all_metrics['f_f1'] = f1
    all_metrics['accuracy'] = accuracy
    all_metrics['kappa'] = kappa
    wandb.log(all_metrics)

    util_tool.write_output(output_path, json.dumps(output_dialogues, indent=2, ensure_ascii=False), "w")


