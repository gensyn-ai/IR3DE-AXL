import os, argparse
from copy import deepcopy

import torch
from transformers import AutoModel, AutoTokenizer, set_seed

from ir3de_stats.m2d2_utils import get_clm_dataloaders, get_reasoning_dataloaders


TAGS_MAP = {
    # clm datasets
    'cs_l1': 'coding',
    'math_l1': 'math',
    'physics_l1': 'physics',
    'History_and_events': 'history',
    'Philosophy_and_thinking': 'philosophy',
    # reasoning datasets
    'gsm8k': 'math',
    'm_arc': 'multilingual',
    'humaneval': 'coding',
    'ifeval': 'instruction',
}


MODEL_MAP = {
    'gsm8k': "MergeBench/Llama-3.2-3B_math",
    'm_arc': "MergeBench/Llama-3.2-3B_multilingual",
    'humaneval': "MergeBench/Llama-3.2-3B_coding",
    'ifeval': "MergeBench/Llama-3.2-3B_instruction"
}


os.environ["HF_ALLOW_CODE_EVAL"] = "1"


def get_args():
    parser = argparse.ArgumentParser(description="Extract IR3DE statistics")
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument("--dataset", type=str, choices=list(TAGS_MAP.keys()), default='ifeval', help="Domain to extract stats for")  # TODO: required=True
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training and evaluation')
    parser.add_argument('--max-num-steps', type=int, default=1000, help='Maximum number of steps to process from the dataloader')
    parser.add_argument('--max-num-samples', type=int, default=10000, help='Maximum number of samples to use from the dataset')
    parser.add_argument('--print-interval', type=int, default=100, help='Interval (in steps) at which to print progress updates')
    args = parser.parse_args()
    return args


def main():

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    router_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
    router_tokenizer.pad_token = router_tokenizer.eos_token
    model = AutoModel.from_pretrained("meta-llama/Meta-Llama-3-8B", device_map="auto")
    router_embedder = deepcopy(model.embed_tokens).to(torch.float32)

    if args.dataset in ['cs_l1', 'math_l1', 'physics_l1', 'History_and_events', 'Philosophy_and_thinking']:
        dataloaders = get_clm_dataloaders(dataset_name=args.dataset, batch_size=args.batch_size, num_workers=4, drop_last_test=True)
    else:
        dataloaders = get_reasoning_dataloaders(task_name=args.dataset, max_num_samples=args.max_num_samples, tokenizer=router_tokenizer, batch_size=args.batch_size, num_workers=4)
    train_dataloader = dataloaders['train']

    A = torch.zeros((router_embedder.weight.shape[1] + 1, router_embedder.weight.shape[1] + 1), dtype=torch.float32, device=device)
    b = torch.zeros(router_embedder.weight.shape[1] + 1, dtype=torch.float32, device=device)

    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(train_dataloader):
            input_ids = batch["input_ids"].to(device, non_blocking=True)  # no need to decode and then use router tokenizer since the input is already tokenized with the same tokenizer        
            X = router_embedder(input_ids)
            X = X.reshape(-1, X.size(-1))
            X_with_bias = torch.cat((X, torch.ones((X.shape[0], 1), dtype=torch.float32).to(X.device)), dim=1)
            batch_A = X_with_bias.T @ X_with_bias
            batch_b = torch.sum(X_with_bias, dim=0)
            A += batch_A
            b += batch_b
            if (step + 1) % args.print_interval == 0:
                print(f"Processed {step + 1} batches for dataset {args.dataset}.")
            if step + 1 >= args.max_num_steps:
                break
    
    os.makedirs("ir3de_stats", exist_ok=True)
    
    freeze_attn_str = "_freeze_attn" if args.dataset in ['cs_l1', 'math_l1', 'physics_l1', 'History_and_events', 'Philosophy_and_thinking'] else ""
    seed_str = f"_seed={args.seed}" if args.dataset in ['cs_l1', 'math_l1', 'physics_l1', 'History_and_events', 'Philosophy_and_thinking'] else ""
    mergebench_str = "_mergebench" if args.dataset in ['gsm8k', 'm_arc', 'humaneval', 'ifeval'] else ""
    path = os.path.join("ir3de_stats", f"Meta-Llama-3-8B_{args.dataset}{seed_str}{freeze_attn_str}{mergebench_str}.pth")
    dataset_name = f"M2D2/{args.dataset}" if args.dataset in ['cs_l1', 'math_l1', 'physics_l1', 'History_and_events', 'Philosophy_and_thinking'] else args.dataset
    
    torch.save(
        {
            "A": A.cpu(),
            "b": b.cpu(),
            "domain_tags": [TAGS_MAP[args.dataset]],
            "datasets_names": [dataset_name],
            "tokenizer": "meta-llama/Meta-Llama-3-8B",
            "embedder": "meta-llama/Meta-Llama-3-8B",
        },
        path
    )
    print(f"Saved IR3DE statistics for dataset {args.dataset} to {path}")


if __name__ == "__main__":
    args = get_args()
    for tag in TAGS_MAP:
        print(f"Extracting IR3DE statistics for dataset {tag}...")
        args.dataset = tag
        main()