import os
import glob
import random

import torch
import torch.distributed as dist
from torch.utils.data import get_worker_info, DataLoader
from datasets.io.parquet import ParquetDatasetReader
from lm_eval.tasks import get_task_dict


class ShardedDataset(torch.utils.data.IterableDataset):

    def __init__(self, path: str):
        self.path = path
        # Defer reader creation to __iter__ to avoid forking issues with workers
        self._reader = None

    def _ensure_reader(self):
        if self._reader is None:
            self._reader = ParquetDatasetReader(self.path, streaming=True).read()
        return self._reader

    def __iter__(self):
        reader = self._ensure_reader()

        # Discover distributed rank/world size if initialized
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1

        # Account for DataLoader workers per process
        wi = get_worker_info()
        if wi is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = wi.id
            num_workers = wi.num_workers

        # Global sharding across all ranks and workers
        shard_id = rank * num_workers + worker_id
        num_shards = world_size * num_workers

        for idx, sample in enumerate(reader):
            if (idx % num_shards) != shard_id:
                continue
            yield {
                key: torch.tensor(value, dtype=torch.int64, device="cpu")
                for key, value in sample.items()
            }

    # def __len__(self):
    #     return len(self.data_reader)


class IRDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, tokenizer, max_length=1025, dataset_name='gsm8k'):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dataset_name = dataset_name

    def __len__(self):
        return len(self.dataset)

    def get_prompt(self, idx):
        if self.dataset_name == 'gsm8k':
            prompt = f"{self.dataset[idx]['question']}"
        elif self.dataset_name == 'm_arc':
            prompt = f"{self.dataset[idx]['instruction']}"
        elif self.dataset_name == 'humaneval' or self.dataset_name == 'ifeval':
            prompt = f"{self.dataset[idx]['prompt']}"
        else:
            raise ValueError(f"Unsupported dataset name: {self.dataset_name}")
        return prompt

    def __getitem__(self, idx):
        prompt = self.get_prompt(idx)
        out = self.tokenizer(prompt, max_length=self.max_length, padding='max_length', truncation=True)
        return {
            "input_ids": torch.tensor(out['input_ids'][:-1]),
            "label": torch.tensor(out['input_ids'][1:]),
            "attention_mask": torch.tensor(out['attention_mask'][:-1])
        }
    

def get_raw_clm_datasets(dataset_name: str):
    if dataset_name == 'openwebtext':
        path = 'datasets/openwebtext/raw'

    elif dataset_name == 'math_l1':
        path = 'datasets/math_l1'
    elif dataset_name == 'cs_l1':
        path = 'datasets/cs_l1'
    elif dataset_name == 'physics_l1':
        path = 'datasets/physics_l1'
    elif dataset_name == 'History_and_events':
        path = 'datasets/History_and_events'
    elif dataset_name == 'Philosophy_and_thinking':
        path = 'datasets/Philosophy_and_thinking'
                
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not supported.")

    # Resolve dataset directory relative to the repo root (parent of src/)
    workspace_root = os.path.dirname(os.path.dirname(__file__))
    dataset_dir = os.path.join(workspace_root, path)

    # Pre-check: ensure train split exists to avoid opaque HF error
    train_glob = os.path.join(dataset_dir, 'train', '*.parquet')
    if not glob.glob(train_glob):
        raise ValueError(
            f"No Parquet files found for train split at: {train_glob}. "
            f"Verify your dataset path and splits."
        )

    datasets = {
        'train': ShardedDataset(train_glob),
        'validation': ShardedDataset(os.path.join(dataset_dir, 'validation', '*.parquet')),
        'test': ShardedDataset(os.path.join(dataset_dir, 'test', '*.parquet'))
    }

    return datasets


def get_clm_dataloaders(dataset_name: str, batch_size: int, num_workers: int, drop_last_test: bool = False):

    datasets = get_raw_clm_datasets(dataset_name)

    dataloaders = {
        'train': DataLoader(datasets['train'], batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers, drop_last=True),
        'validation': DataLoader(datasets['validation'], batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers),
        'test': DataLoader(datasets['test'], batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers, drop_last=drop_last_test)
    }

    return dataloaders


def get_m_arc_merged_dataset(max_num_samples, task, split):
    remaining_samples = max_num_samples
    dataset = []
    while remaining_samples > 0:
        num_samples_per_arc = max(1, int(remaining_samples / len(task.keys())))  # type: ignore
        for arc_task_name in task.keys():  # type: ignore
            dataset.extend(list(task[arc_task_name].dataset[split])[:num_samples_per_arc])
            remaining_samples -= num_samples_per_arc
            if remaining_samples <= 0:
                break
    return dataset


def get_reasoning_task(task_name):

    NUM_SHOTS = {
        'gsm8k': 8,
        'mathqa': 0,
        'hendrycks_math': 0,
        'm_mmlu': None,
        'humaneval': None,
        'ifeval': None,
        'm_arc': None
    }

    # Define multilingual ARC task group
    if task_name == "m_arc":
        multilingual_arc_tasks = [
            "arc_ar", "arc_bn", "arc_ca", "arc_de", "arc_es", "arc_eu", 
            "arc_fr", "arc_gu", "arc_hi", "arc_hr", "arc_hu", "arc_hy",
            "arc_id", "arc_it", "arc_kn", "arc_ml", "arc_mr", "arc_ne",
            "arc_nl", "arc_pt", "arc_ro", "arc_ru", "arc_sk", "arc_sr",
            "arc_sv", "arc_ta", "arc_te", "arc_uk", "arc_vi", "arc_zh"
        ]
        task_dict = get_task_dict(multilingual_arc_tasks)  # type: ignore
    else:
        task_dict = get_task_dict([task_name])

    if task_name in ("humaneval", "ifeval", "m_arc"):
        limit = None
    else:
        task = task_dict[task_name]
        task.fewshot = NUM_SHOTS[task_name]
        task.bootstrap_iters = 0
        limit = None
    
    return task_dict, limit


def get_reasoning_dataloaders(task_name, max_num_samples, tokenizer, batch_size, num_workers):

    with torch.no_grad():

        task_dict, _ = get_reasoning_task(task_name)

        if task_name == "m_arc":
            train_dataset = get_m_arc_merged_dataset(max_num_samples, task_dict, split='train')
            test_dataset = get_m_arc_merged_dataset(max_num_samples, task_dict, split='test')
        else:
            task = task_dict[task_name]
            if task_name == "humaneval":
                key1, key2 = 'test', 'test'  # humaneval doesn't have a train/test split, so we use the test set for both (as we are not generating any answer from the prompts, just using them for embedding extraction)
            elif task_name == "ifeval":
                key1, key2 = 'train', 'train'  # ifeval doesn't have a train/test split, so we use the train set for both (as we are not generating any answer from the prompts, just using them for embedding extraction)
            else:
                key1, key2 = 'train', 'test'
            train_dataset = list(task.dataset[key1])
            test_dataset = list(task.dataset[key2])

        if len(train_dataset) < max_num_samples:
            original_dataset = train_dataset.copy()
            while len(train_dataset) < max_num_samples:
                remaining_needed = max_num_samples - len(train_dataset)
                if remaining_needed >= len(original_dataset):
                    train_dataset.extend(original_dataset)
                else:
                    train_dataset.extend(original_dataset[:remaining_needed])
                    break
        elif len(train_dataset) > max_num_samples:
            random.shuffle(train_dataset)
            train_dataset = train_dataset[:max_num_samples]
        
        train_dataset = IRDataset(train_dataset, tokenizer, dataset_name=task_name)
        test_dataset = IRDataset(test_dataset, tokenizer, dataset_name=task_name)

        dataloaders = {
            'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers, drop_last=True),
            'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
        }

        return dataloaders
