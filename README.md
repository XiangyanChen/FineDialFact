# FineDialFact: A Benchmark for Fine-Grained Dialogue Fact Verification

## Overview

FineDialFact is a benchmark for fine-grained dialogue fact verification introduced in LREC2026. The benchmark decomposes dialogue responses into atomic facts and verifies each fact independently, which makes it possible to handle responses that contain a mix of correct, incorrect, and non-verifiable information. It is built from public dialogue datasets and evaluated with several baseline methods, including Chain-of-Thought prompting and reasoning distillation. The paper reports that the task remains challenging, especially on the human-annotated HybriDialogue set.


## Data Format

Each example in `annotated_test_data.json` is a JSON object that contains the annotated dialogue data used for evaluation.

A typical sample includes:
- `id`: unique example id.
- `dialogue` / `conversation`: the input dialogue context.
- `response`: the model or reference response.
- `atomic_facts`: a list of factual claims extracted from the response.
- `annotations`: human or automatic labels for factuality-related evaluation.

Example:

```json
{
  "id": "example_001",
  "dialogue": [
    {"role": "user", "content": "Who wrote Hamlet?"},
    {"role": "assistant", "content": "Shakespeare wrote Hamlet."}
  ],
  "response": "Shakespeare wrote Hamlet.",
  "atomic_facts": [
    "Shakespeare wrote Hamlet."
  ],
  "annotations": {
    "factual": true
  }
}
```

The evaluation scripts read the annotated test file and compute atomic-fact and response-level metrics based on these fields.

## Setup

Install the project dependencies with:

```bash
pip install -r requirements.txt
```

The provided requirements file includes the main packages used by the project, including `torch`, `transformers`, `datasets`, `sentence-transformers`, `faiss`, `spacy`, `evaluate`, and other supporting libraries. 

## Pipeline

DATASET=HybriDial

MODE=few-shot-cot

MODEL=Llama-3.1-8B-Instruct

DISTILL_MODE=plain

TEMPERATURE=0

### Evaluation for human annotated data

```bash
python eval.py \
  --dataset $DATASET \
  --mode $MODE \
  --model $MODEL \
  --distill_mode $DISTILL_MODE \
  --batch_size 10 \
  --temperature $TEMPERATURE \
  --skip_non_factual_claim
```

### Evaluation for GPT-annotated data

```bash
python eval_atomic_facts.py \
  --dataset $DATASET \
  --mode $MODE \
  --model $MODEL \
  --distill_mode $DISTILL_MODE \
  --batch_size 10 \
  --temperature $TEMPERATURE
```

## Citation

If you use this repository, please cite the FineDialFact paper.

```bibtex
@article{chen2025finedialfact,
  title={FineDialFact: A benchmark for Fine-grained Dialogue Fact Verification},
  author={Chen, Xiangyan and Li, Yufeng and Gan, Yujian and Zubiaga, Arkaitz and Purver, Matthew},
  journal={arXiv preprint arXiv:2508.05782},
  year={2025}
}
```
