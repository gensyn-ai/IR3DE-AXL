import glob
import os
import random

import torch
import torch.distributed as dist
from datasets.io.parquet import ParquetDatasetReader
from lm_eval.tasks import get_task_dict
from torch.utils.data import DataLoader, get_worker_info


class ShardedDataset(torch.utils.data.IterableDataset):
    """Streams a Parquet glob, splitting it across DataLoader workers and
    distributed ranks so each sample is read by exactly one of them. Used
    by get_raw_clm_datasets for the CLM (M2D2) datasets."""

    def __init__(self, path: str):
        """`path` is a glob pattern, e.g. '.../train/*.parquet'."""
        self.path = path
        # Defer reader creation to __iter__ to avoid forking issues with workers
        self._reader = None

    def _ensure_reader(self):
        """Lazily create the underlying streaming Parquet reader."""
        if self._reader is None:
            self._reader = ParquetDatasetReader(self.path, streaming=True).read()
        return self._reader

    def __iter__(self):
        """Yield this shard's samples as int64 CPU tensors."""
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


class IRDataset(torch.utils.data.Dataset):
    """Tokenizes one reasoning-benchmark's prompts on the fly for IR3DE stats
    extraction. Built by get_reasoning_dataloaders, one instance per split."""

    def __init__(self, dataset, tokenizer, max_length=1025, dataset_name='gsm8k'):
        """`dataset` is the raw (already sampled/merged) list of examples."""
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dataset_name = dataset_name

    def __len__(self):
        """Number of examples in the underlying (already sampled) dataset."""
        return len(self.dataset)

    def get_prompt(self, idx):
        """Extract example idx's prompt text; the field name varies by dataset."""
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
        """Tokenize example idx into next-token-prediction input/label/mask tensors."""
        prompt = self.get_prompt(idx)
        out = self.tokenizer(prompt, max_length=self.max_length, padding='max_length', truncation=True)
        return {
            "input_ids": torch.tensor(out['input_ids'][:-1]),
            "label": torch.tensor(out['input_ids'][1:]),
            "attention_mask": torch.tensor(out['attention_mask'][:-1])
        }
    

def get_raw_clm_datasets(dataset_name: str):
    """Build train/validation/test ShardedDatasets for one of the M2D2 CLM
    domains (math_l1, cs_l1, ...) from Parquet files under datasets/. Called
    by get_clm_dataloaders, which extract_ir3de_stats.py uses for CLM_DATASETS."""
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
    """Wrap get_raw_clm_datasets' splits in DataLoaders. Entry point used by
    extract_ir3de_stats.py for CLM_DATASETS (math_l1, cs_l1, physics_l1, ...)."""
    datasets = get_raw_clm_datasets(dataset_name)

    dataloaders = {
        'train': DataLoader(datasets['train'], batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers, drop_last=True),
        'validation': DataLoader(datasets['validation'], batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers),
        'test': DataLoader(datasets['test'], batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers, drop_last=drop_last_test)
    }

    return dataloaders


def get_m_arc_merged_dataset(max_num_samples, task, split):
    """Round-robin up to max_num_samples examples across all m_arc language
    subtasks, advancing each language's own read offset every pass so no
    sample is taken twice. Called by get_reasoning_dataloaders for m_arc."""
    remaining_samples = max_num_samples
    dataset = []
    offsets = {name: 0 for name in task.keys()}  # type: ignore
    while remaining_samples > 0:
        num_samples_per_arc = max(1, int(remaining_samples / len(task.keys())))  # type: ignore
        made_progress = False
        for arc_task_name in task.keys():  # type: ignore
            start = offsets[arc_task_name]
            chunk = list(task[arc_task_name].dataset[split])[start:start + num_samples_per_arc]
            if chunk:
                made_progress = True
            dataset.extend(chunk)
            offsets[arc_task_name] = start + len(chunk)
            remaining_samples -= len(chunk)
            if remaining_samples <= 0:
                break
        if not made_progress:
            break  # every language's dataset is exhausted; can't reach max_num_samples
    return dataset


def get_reasoning_task(task_name):
    """Load an lm-eval-harness task (or, for m_arc, all 30 language
    subtasks merged into one dict) and apply its few-shot config."""
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
    """Build train/test DataLoaders for a reasoning benchmark (gsm8k,
    m_arc, humaneval, ifeval), resampling to exactly max_num_samples.
    Entry point used by extract_ir3de_stats.py for non-CLM datasets."""
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
