
import datasets

import getpass
from datetime import datetime

from collections import defaultdict
import tqdm
import random
import numpy as np
from typing import Dict, List, Optional, Iterator, Callable, Union, Tuple
import time
import  sys
import os
import datetime
from operator import eq
from ltcc5 import LTCUnit
import argparse
import mempute as mp

class TemporarilySeededRandom:
    def __init__(self, seed):
        """Temporarily set the random seed, and then restore it when exiting the context."""
        self.seed = seed
        self.stored_state = None
        self.stored_np_state = None

    def __enter__(self):
        # Store the current random state
        self.stored_state = random.getstate()
        self.stored_np_state = np.random.get_state()

        # Set the random seed
        random.seed(self.seed)
        np.random.seed(self.seed)

    def __exit__(self, exc_type, exc_value, traceback):
        # Restore the random state
        random.setstate(self.stored_state)
        np.random.set_state(self.stored_np_state)

class Trainer:
    def __init__(self, args, policy): 
        self.seed = args.seed
        self.loss_name = args.loss_name
        self.tokenizer = policy.tokenizer
        self.m_name = policy.m_name
        
    def open_logf(self, type):
        log_name = f"{self.m_name}/{type}.log"
        self.logfp = open(log_name, 'w')

    def logf(self, format, *args):
        data = format % args
        print(data)
        #if self.logfp is not None:
        #    self.logfp.write(data + '\n')
    def logf2(self, format, *args):
        data = format % args
        print(data)
        if self.logfp is not None:
            self.logfp.write(data + '\n')
            self.logfp.flush()
    def close_logf(self):
        self.logfp.close()

def concatenated_inputs(batch: Dict[str, List]) -> Dict[str, np.array]:
    concatenated_batch = {}
    for k in batch:
        if k.startswith('chosen'):
            concatenated_key = k.replace('chosen', 'concatenated')
            concatenated_batch[concatenated_key] = batch[k]
    for k in batch:
        if k.startswith('rejected'):
            concatenated_key = k.replace('rejected', 'concatenated')
            concatenated_batch[concatenated_key] = np.concatenate((concatenated_batch[concatenated_key], batch[k]),axis=0)
            concatenated_batch[concatenated_key] = np.array(concatenated_batch[concatenated_key], dtype='int64')

    concatenated_batch['concatenated_input_ids'] = np.expand_dims(concatenated_batch['concatenated_input_ids'], -1)
    
    return concatenated_batch

def get_collate_fn(tokenizer) -> Callable[[List[Dict]], Dict[str, List]]:
    """Returns a collate function for the given tokenizer.
    
       The collate function takes a list of examples (dicts, where values are lists of
         ints [tokens] or strings [the original texts]) and returns a batch of examples,
         PyTorch tensors padded to the maximum length. Strings are passed through."""
    def collate_fn(batch):
        # first, pad everything to the same length
        dict_batch = {}
        for k in batch[0].keys():
            dict_batch[k] = [ex[k] for ex in batch]

        return dict_batch
    return collate_fn

def tokenize_batch_element(prompt: str, chosen: str, rejected: str, truncation_mode: str, tokenizer, max_length: int, max_prompt_length: int) -> Dict:
    """Tokenize a single batch element.
    
       At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
         in case the prompt + chosen or prompt + rejected responses is/are too long. First
         we truncate the prompt; if we're still too long, we truncate the chosen/rejected.
       
       We also create the labels for the chosen/rejected responses, which are of length equal to
         the sum of the length of the prompt and the chosen/rejected response, with -100 for the
         prompt tokens.
    """
    chosen_tokens = {'input_ids': tokenizer.encode(chosen)}
    rejected_tokens = {'input_ids': tokenizer.encode(rejected)} 
    prompt_tokens = {'input_ids': tokenizer.encode(prompt)} 

    assert tokenizer.bos_token not in prompt_tokens['input_ids'], f"Prompt contains BOS token: {prompt}"
    assert tokenizer.bos_token not in chosen_tokens['input_ids'], f"Chosen response contains BOS token: {chosen}"
    assert tokenizer.bos_token not in rejected_tokens['input_ids'], f"Rejected response contains BOS token: {rejected}"

    prompt_tokens['input_ids'].insert(0, tokenizer.bos_token)

    longer_response_length = max(len(chosen_tokens['input_ids']), len(rejected_tokens['input_ids']))

    # if combined sequence is too long, truncate the prompt
    if len(prompt_tokens['input_ids']) + longer_response_length > max_length:
        if truncation_mode == 'keep_start':
            prompt_tokens = {k: v[:max_prompt_length] for k, v in prompt_tokens.items()}
        elif truncation_mode == 'keep_end':
            prompt_tokens = {k: v[-max_prompt_length:] for k, v in prompt_tokens.items()}
        else:
            raise ValueError(f'Unknown truncation mode: {truncation_mode}')

    # if that's still too long, truncate the response
    if len(prompt_tokens['input_ids']) + longer_response_length > max_length:
        chosen_tokens = {k: v[:max_length - max_prompt_length] for k, v in chosen_tokens.items()}
        rejected_tokens = {k: v[:max_length - max_prompt_length] for k, v in rejected_tokens.items()}

    # Create labels
    chosen_sequence_tokens = {k: prompt_tokens[k] + chosen_tokens[k] for k in chosen_tokens}
    rejected_sequence_tokens = {k: prompt_tokens[k] + rejected_tokens[k] for k in rejected_tokens}
    chosen_sequence_tokens['labels'] = chosen_sequence_tokens['input_ids'][:]
    chosen_sequence_tokens['labels'][:len(prompt_tokens['input_ids'])] = [0] * len(prompt_tokens['input_ids'])
    rejected_sequence_tokens['labels'] = rejected_sequence_tokens['input_ids'][:]
    rejected_sequence_tokens['labels'][:len(prompt_tokens['input_ids'])] = [0] * len(prompt_tokens['input_ids'])
    #padding
    chosen_sequence_tokens['input_ids'] = chosen_sequence_tokens['input_ids'] + [tokenizer.pad_token] * (max_length - len(chosen_sequence_tokens['input_ids']))
    chosen_sequence_tokens['labels'] = chosen_sequence_tokens['labels'] + [0] * (max_length - len(chosen_sequence_tokens['labels']))
    rejected_sequence_tokens['input_ids'] = rejected_sequence_tokens['input_ids'] + [tokenizer.pad_token] * (max_length - len(rejected_sequence_tokens['input_ids']))
    rejected_sequence_tokens['labels'] = rejected_sequence_tokens['labels'] + [0] * (max_length - len(rejected_sequence_tokens['labels']))

    batch = {}
    """
    batch['prompt'] = prompt
    batch['chosen'] = prompt + chosen
    batch['rejected'] = prompt + rejected
    batch['chosen_response_only'] = chosen
    batch['rejected_response_only'] = rejected
    """
    for k, toks in {'chosen': chosen_sequence_tokens, 'rejected': rejected_sequence_tokens, 'prompt': prompt_tokens}.items():
        for type_key, tokens in toks.items():
            if type_key == 'token_type_ids':
                continue
            batch[f'{k}_{type_key}'] = tokens
    #print(batch)
    return batch

def extract_anthropic_prompt(prompt_and_response):
    """Extract the anthropic prompt from a prompt and response pair."""
    search_term = '\n\nAssistant:'
    search_term_idx = prompt_and_response.rfind(search_term)
    assert search_term_idx != -1, f"Prompt and response does not contain '{search_term}'"
    return prompt_and_response[:search_term_idx + len(search_term)]

def get_hh(split: str, silent: bool = False, cache_dir: str = None) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    """Load the Anthropic Helpful-Harmless dataset from Huggingface and convert it to the necessary format.
    
       The dataset is converted to a dictionary with the following structure:
       {
           'prompt1': {
               'responses': List[str],
               'pairs': List[Tuple[int, int]],
               'sft_target': str
           },
           'prompt2': {
               ...
           },
       }

       Prompts should be structured as follows:
         \n\nHuman: <prompt>\n\nAssistant:
       Multiple turns are allowed, but the prompt should always start with \n\nHuman: and end with \n\nAssistant:.
       
       For this dataset, the sft_target is just the chosen response.
    """
    print(f'Loading HH HA dataset ({split} split) from Huggingface... dir: {cache_dir}')
    dataset = datasets.load_dataset('Anthropic/hh-rlhf', split=split)#, cache_dir=cache_dir)
    print('done')
    
    def split_prompt_and_responses(ex):
        prompt = extract_anthropic_prompt(ex['chosen'])
        chosen_response = ex['chosen'][len(prompt):]
        rejected_response = ex['rejected'][len(prompt):]
        return prompt, chosen_response, rejected_response

    data = defaultdict(lambda: defaultdict(list))
    for row in tqdm.tqdm(dataset, desc='Processing HH', disable=silent):
        prompt, chosen, rejected = split_prompt_and_responses(row)
        responses = [chosen, rejected]
        n_responses = len(data[prompt]['responses'])
        data[prompt]['pairs'].append((n_responses, n_responses + 1))
        data[prompt]['responses'].extend(responses)

    return data

def get_kcd(split: str, silent: bool = False, cache_dir: str = None) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    """Load the Anthropic Helpful-Harmless dataset from Huggingface and convert it to the necessary format.
    
       The dataset is converted to a dictionary with the following structure:
       {
           'prompt1': {
               'responses': List[str],
               'pairs': List[Tuple[int, int]],
               'sft_target': str
           },
           'prompt2': {
               ...
           },
       }
    """
    print(f'Loading KCD SFT dataset ({split} split) from Huggingface... dir: {cache_dir}')
    dataset = datasets.load_dataset('MarkrAI/KoCommercial-Dataset', split=split)#, cache_dir=cache_dir)
    print('done')
    
    #data = defaultdict(lambda: defaultdict(list))
    data = defaultdict(list)
    for row in tqdm.tqdm(dataset, desc='Processing KCD', disable=silent):
        data['prompt'].append(row['instruction'] + row['input'])
        data['output'].append(row['output'])
    #print('111: ', data['prompt'][0])
    #print('222: ', data['output'][0])
    return data

def get_dataset(name: str, split: str, cache_dir: str = None):

    if name == 'kcd':
        data = get_kcd(split, cache_dir=cache_dir)
        
    else:
        if name == 'hh':
            data = get_hh(split, cache_dir=cache_dir)
        else:
            raise ValueError(f"Unknown dataset '{name}'")

        assert set(list(data.values())[0].keys()) == {'responses', 'pairs'}, \
            f"Unexpected keys in dataset: {list(list(data.values())[0].keys())}"
            
    return data

def get_local_dir(prefixes_to_resolve: List[str]) -> str:
    """Return the path to the cache directory for this user."""
    for prefix in prefixes_to_resolve:
        if os.path.exists(prefix):
            return f"{prefix}/{getpass.getuser()}"
    os.makedirs(prefix)
    return f"{prefix}/{getpass.getuser()}"

def get_batch_iterator(names: List[str],
                       tokenizer,
                       split: str = 'train',
                       batch_size: int = 1,
                       shuffle: bool = True,
                       max_length: int = 512,
                       max_prompt_length: int = 128,
                       n_epochs: Optional[int] = None,
                       n_examples: Optional[int] = None,
                       seed:int = 0,
                       cache_dir: Optional[str] = None) -> Iterator[Dict]:
   
    assert n_epochs is not None or n_examples is not None, "Must specify either n_epochs or n_examples"

    with TemporarilySeededRandom(seed):
        permutation_seeds = iter(np.random.randint(0, 2**30, size=1000000))
        flat_data = []
        for name in names:
            truncation_mode = 'keep_end' if name == 'hh' else 'keep_start'
            for prompt, data in get_dataset(name, split, cache_dir=cache_dir).items():
                flat_data.append((prompt, data['responses'], data['pairs'], truncation_mode))
            
    collate_fn = get_collate_fn(tokenizer)

    epoch_idx = 0
    example_idx = 0
    done = False
    while True:
        if n_epochs is not None and epoch_idx >= n_epochs:
            print(f'Finished generating {n_epochs} epochs on {split} split')
            break
        if shuffle:
            with TemporarilySeededRandom(next(permutation_seeds)):
                random.shuffle(flat_data)

        batch = []
        for prompt, responses, pairs, truncation_mode in flat_data:
            if done:
                break
            for p in pairs:
                if done:
                    break
                batch_element = tokenize_batch_element(prompt, responses[p[0]], responses[p[1]], truncation_mode, tokenizer, max_length, max_prompt_length)
                batch.append(batch_element)
                example_idx += 1
                if len(batch) == batch_size:
                    yield collate_fn(batch)
                    if n_examples is not None and example_idx >= n_examples:
                        print(f'FINISHED {n_examples} EXAMPLES on {split} split')
                        done = True
                    batch = []
        if done:
            break

        epoch_idx += 1

def sigmoid(x):
    return 1 / (1 +np.exp(-x))

class HATrainer(Trainer):
    def __init__(self, args, policy, reference_model):
        super().__init__(args, policy)
        self.bata = args.beta
        self.reference_free = args.reference_free
        self.eval_every = args.eval_every

        data_iterator_kwargs = dict(
            names=args.datasets,
            tokenizer=self.tokenizer,
            shuffle=True,
            max_length=args.max_length,
            max_prompt_length=args.max_prompt_length,
        )

        self.policy = policy
        self.reference_model = reference_model

        self.train_iterator = get_batch_iterator(**data_iterator_kwargs, split='train', n_epochs=args.n_epochs, n_examples=args.n_examples, batch_size=args.batch_size, cache_dir=get_local_dir(args.local_dirs))
        print(f'Loaded train data iterator')
        self.eval_iterator = get_batch_iterator(**data_iterator_kwargs, split='test', n_examples=args.n_eval_examples, batch_size=args.eval_batch_size, cache_dir=get_local_dir(args.local_dirs))
        print(f'Loaded eval batches of size {args.eval_batch_size}')

    def calc_batch_logps(self, concatenated_logps: np.array, labels: np.array, batch_sz, average_log_prob: bool = False) -> np.array:
    
        assert logits.shape[:-1] == labels.shape

        labels = labels[:, 1:].copy() #bos cut
        loss_mask = (labels != 0)

        #concatenated_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        if average_log_prob:
            all_logps = np.sum((concatenated_logps * loss_mask), axis = -1) / np.sum(loss_mask, axis = -1)
        else:
            all_logps = np.sum((concatenated_logps * loss_mask), axis = -1)

        chosen_logps = all_logps[:batch_sz]
        rejected_logps = all_logps[batch_sz:]

        return chosen_logps, rejected_logps

    def evaluate(self):

        reference_free = self.reference_free if self.loss_name == 'dpo' else False
        metrics = {}

        for batch in self.eval_iterator:

            concatenated_batch = concatenated_inputs(batch)

            if self.loss_name != 'orpo':
                concatenated_logps = self.reference_model.xpredict(concatenated_batch['concatenated_input_ids'], concatenated_batch['concatenated_labels'])
                reference_chosen_logps, reference_rejected_logps = calc_batch_logps(concatenated_logps, concatenated_batch['concatenated_labels'], batch['chosen_input_ids'].shape[0])

            concatenated_logps = self.policy.xpredict(concatenated_batch['concatenated_input_ids'], concatenated_batch['concatenated_labels'])
            policy_chosen_logps, policy_rejected_logps = calc_batch_logps(concatenated_logps, concatenated_batch['concatenated_labels'], batch['chosen_input_ids'].shape[0])

            if self.loss_name == 'orpo':
                pos_prob = policy_chosen_logps
                neg_prob = policy_rejected_logps
                # Calculate log odds
                log_odds = (pos_prob - neg_prob) - (np.log1p(-np.exp(pos_prob)) - np.log1p(-np.exp(neg_prob)))
                sig_ratio = sigmoid(log_odds)
                ratio = np.log(sig_ratio)
                
                mean_eval_metrics[f'Positive Geometric Mean'] = pos_prob
                mean_eval_metrics[f'Negative Geometric Mean'] = neg_prob
                mean_eval_metrics[f'Log Odds Ratio'] = ratio
                mean_eval_metrics[f'Log Odds'] = log_odds
            else:
                chosen_rewards = self.bata * (policy_chosen_logps - reference_chosen_logps)
                rejected_rewards = self.bata * (policy_rejected_logps - reference_rejected_logps)
                reward_accuracies = (chosen_rewards > rejected_rewards).float()

                metrics[f'rewards_eval/chosen'].extend(chosen_rewards.tolist())
                metrics[f'rewards_eval/rejected'].extend(rejected_rewards.tolist())
                metrics[f'rewards_eval/accuracies'].extend(reward_accuracies.tolist())
                metrics[f'rewards_eval/margins'].extend((chosen_rewards - rejected_rewards).tolist())

        return metrics

    def train(self):

        self.example_counter = 1
        self.batch_counter = 0

        for batch in self.train_iterator:

            if self.example_counter % self.eval_every == 0:
                all_eval_metrics  = self.evaluate()
                mean_eval_metrics = {k: sum(v) / len(v) for k, v in all_eval_metrics.items()}
                print(f'eval after {self.example_counter}: {formatted_dict(mean_eval_metrics)}')

            concatenated_batch = concatenated_inputs(batch)
            #print(concatenated_batch['concatenated_input_ids'].shape)
            #print(concatenated_batch['concatenated_input_ids'][0])
            if self.loss_name == 'orpo':
                concatenated_logps = concatenated_batch[concatenated_labels]
            else:
                concatenated_logps = self.reference_model.xpredict(concatenated_batch['concatenated_input_ids'], concatenated_batch['concatenated_labels'])
                concatenated_logps = np.concatenate((concatenated_batch[concatenated_labels], concatenated_logps), axis=1)

            rewards = self.policy.xtrain(concatenated_batch['concatenated_input_ids'], concatenated_logps)
            metrics = {}
            chosen_rewards = rewards[:batch['chosen_input_ids'].shape[0]]
            rejected_rewards = rewards[batch['chosen_input_ids'].shape[0]:]

            if self.loss_name == 'orpo':
                pos_prob = chosen_rewards
                neg_prob = rejected_rewards
                # Calculate log odds
                log_odds = (pos_prob - neg_prob) - (np.log1p(-np.exp(pos_prob)) - np.log1p(-np.exp(neg_prob)))
                sig_ratio = sigmoid(log_odds)
                ratio = np.log(sig_ratio)
                
                mean_eval_metrics[f'Positive Geometric Mean'] = np.mean(pos_prob)
                mean_eval_metrics[f'Negative Geometric Mean'] = np.mean(neg_prob)
                mean_eval_metrics[f'Log Odds Ratio'] = np.mean(ratio)
                mean_eval_metrics[f'Log Odds'] = np.mean(log_odds)
            else:
                reward_accuracies = (chosen_rewards > rejected_rewards).float()
                metrics[f'rewards_train/chosen'] = chosen_rewards.tolist()
                metrics[f'rewards_train/rejected'] = rejected_rewards.tolist()
                metrics[f'rewards_train/accuracies'] = reward_accuracies.tolist()
                mean_eval_metrics = {k: sum(v) / len(v) for k, v in metrics.items()}

            print(f'train after {self.example_counter}: {formatted_dict(mean_eval_metrics)}')

            self.batch_counter += 1
            self.example_counter += self.config.batch_size

#========================================================================================
#========================================== SFT =========================================
#========================================================================================

def get_collate_sft(tokenizer) -> Callable[[List[Dict]], Dict[str, np.array]]:

    def collate_sft(batch):
        # first, pad everything to the same length
        dict_batch = {}
        for k in batch[0].keys():
            """
            print('000: ', len(batch))
            print('aaa: ', batch[0])
            print('333: ', k)
            print('444: ', batch[0][k])
            for i, ex in enumerate(batch):
                print(f'666: {i}: ', ex[k])
            aa = [ex['input_ids'] for ex in batch]
            """
            dict_batch[k] = [ex[k] for ex in batch]
            if k != 'prompt_len' and k != 'tot_len':
                dict_batch[k] = np.array(dict_batch[k], dtype='int64')
                dict_batch[k] = np.expand_dims(dict_batch[k], -1)
        #print('222: ', dict_batch)
        return dict_batch
    return collate_sft

def tokenize_batch_sft(prompt, output, tokenizer, max_length: int) -> Dict:

    prompt_tokens = tokenizer.encode(prompt)
    output_tokens = tokenizer.encode(output)

    assert tokenizer.bos_token not in prompt_tokens, "prompt contains BOS token"
    assert tokenizer.bos_token not in output_tokens, "output contains BOS token"

    row_tokens = {}
    row_tokens['input_ids'] = [tokenizer.bos_token] + prompt_tokens
    row_tokens['prompt_len'] = len(row_tokens['input_ids'])
    row_tokens['input_ids'] = row_tokens['input_ids'] + output_tokens
    row_tokens['tot_len'] = len(row_tokens['input_ids'])

    if len(row_tokens['input_ids']) > max_length:
        n = len(row_tokens['input_ids']) - max_length 
        row_tokens['input_ids'] = row_tokens['input_ids'][:-n]
        if row_tokens['prompt_len'] > len(row_tokens['input_ids']) - 2:
            row_tokens['prompt_len'] = len(row_tokens['input_ids']) // 2
    # Create labels
    row_tokens['labels'] = prompt_tokens + output_tokens + [tokenizer.eos_token]
    if len(row_tokens['labels']) > max_length:
        n = len(row_tokens['labels']) - max_length
        row_tokens['labels'] = row_tokens['labels'][:-n]
    #padding
    pad = [tokenizer.pad_token] * (max_length - len(row_tokens['input_ids']))
    row_tokens['input_ids'] = row_tokens['input_ids'] + pad
    row_tokens['labels'] = row_tokens['labels'] + pad
    #print('111: ', row_tokens)
    
    return row_tokens

def export_dataset(names: List[str], cache_dir: str = None):

    cat_name=''
    for name in names:
        cat_name = cat_name + name
    data_fp = open(f"./{cat_name}.txt", 'w', encoding='UTF8')

    for name in names:
        for split in ['train']:#, 'test']:
            data = get_dataset(name, split, cache_dir=cache_dir)
            for prompt, output in zip(data['prompt'], data['output']):
                data_fp.write(prompt + output + '\n')

    data_fp.close()

def sft_batch_iterator(names: List[str],
                       tokenizer,
                       trainer,
                       split: str = 'train',
                       batch_size: int = 1,
                       shuffle: bool = True,
                       max_length: int = 512,
                       n_epochs: Optional[int] = None,
                       n_examples: Optional[int] = None,
                       seed:int = 0,
                       cache_dir: Optional[str] = None) -> Iterator[Dict]:
   
    assert n_epochs is not None or n_examples is not None, "Must specify either n_epochs or n_examples"

    with TemporarilySeededRandom(seed):
        permutation_seeds = iter(np.random.randint(0, 2**30, size=1000000))
        flat_data = []
        for name in names:
            data = get_dataset(name, split, cache_dir=cache_dir)
            for prompt, output in zip(data['prompt'], data['output']):
                flat_data.append((prompt, output))
    
    collate_sft = get_collate_sft(tokenizer)

    trainer.epoch_idx = 0
    example_idx = 0
    done = False

    if batch_size == 0:
        batch_size = (len(flat_data) // 32) * 32
    while True:
        if n_epochs is not None and trainer.epoch_idx >= n_epochs:
            print(f'Finished generating {n_epochs} epochs on {split} split')
            break
        if shuffle:
            with TemporarilySeededRandom(next(permutation_seeds)):
                random.shuffle(flat_data)
 
        batch = []
        for prompt, output in flat_data:
            if done:
                break
            batch_element = tokenize_batch_sft(prompt, output, tokenizer, max_length)
            batch.append(batch_element)
            example_idx += 1
            if len(batch) == batch_size:
                yield collate_sft(batch)
                if n_examples is not None and example_idx >= n_examples:
                    print(f'FINISHED {n_examples} EXAMPLES on {split} split')
                    done = True
                batch = []
        if done:
            break

        trainer.epoch_idx += 1



class SFTTrainer(Trainer):
    def __init__(self, args, policy):
        super().__init__(args, policy)

        self.by_accu = args.by_accu
        
        data_iterator_kwargs = dict(
            names=args.datasets,
            tokenizer=self.tokenizer,
            trainer=self,
            shuffle=True,
            max_length=args.max_length,
        )

        self.policy = policy

        self.train_iterator = sft_batch_iterator(**data_iterator_kwargs, split='train', n_epochs=args.n_epochs, n_examples=args.n_examples, batch_size=args.batch_size, cache_dir=get_local_dir(args.local_dirs))
        print(f'Loaded train data iterator')
        self.train_accu_iter = sft_batch_iterator(**data_iterator_kwargs, split='train', n_epochs=args.n_epochs, n_examples=args.n_examples, batch_size=args.eval_batch_size, cache_dir=get_local_dir(args.local_dirs))
        print(f'Loaded train accu data iterator')
        self.test_accu_iter = sft_batch_iterator(**data_iterator_kwargs, split='test', n_examples=args.n_eval_examples, batch_size=args.eval_batch_size, cache_dir=get_local_dir(args.local_dirs))
        print(f'Loaded test accu batches of size {args.eval_batch_size}')

    def evaluate(self, msg, target_ids_f, pred_ids_f, target_ids_r, pred_ids_r):
        
        self.logf("\n=================== %s ===========================", msg)

        for truth_f, pred_f, truth_r, pred_r in zip(target_ids_f, pred_ids_f, target_ids_r, pred_ids_r):
            truth_sent_f = self.tokenizer.decode(truth_f)
            pred_sent_f = self.tokenizer.decode(pred_f)
            truth_sent_r = self.tokenizer.decode(truth_r)
            pred_sent_r = self.tokenizer.decode(pred_r)
            self.logf("[Truth_f] %s\n", truth_sent_f)
            self.logf("[Truth_r] %s\n", truth_sent_r)
            self.logf("[Translated_f] %s\n", pred_sent_f)
            self.logf("[Translated_r] %s\n", pred_sent_r)

    def accuracy(self, itr):

        query = []
        preds_front = []
        rights_front = []
        preds_rear = []
        rights_rear = []

        try:
            batch = next(itr)
        except StopIteration:
            pass

        nequal = 0
        nright = 0
        toks = batch['input_ids']
        qlen = batch['prompt_len']
        tlen = batch['tot_len']
        for i, (ids, n, t) in enumerate(zip(toks, qlen, tlen)):
            z = np.zeros((ids.shape[0] - n, 1), dtype = ids.dtype)
            q = np.concatenate((ids[:n], z), axis=0) #bos + query[1/2] + zero padding endian [seq, 1]
            q = np.expand_dims(q, axis=0) #batch dim 0 [1, seq, 1]
            pred = mp.xpredict(self.policy.net, q, 1, n) #[1, seq]api에서 q의 공간에 직접 쓰고 q를 리턴하므로 q와 pred는 등일 하다. 
            ids = np.squeeze(ids)#[seq]
            pred = np.squeeze(pred)#[seq]
            r_front = ids[:n]
            r_rear = ids[n:t]
            p_front = pred[:n]
            p_rear = pred[n:t]
            rights_front.append(r_front)
            rights_rear.append(r_rear)
            preds_front.append(p_front)
            preds_rear.append(p_rear)
            nequal += np.sum(np.equal(p_rear, r_rear))#go + query cut
            #print(nequal)
            nright += (t - n)#go + query cut
        #rights = np.array(rights)
        #preds = np.array(preds)
        #query = np.array(query)
        #a = np.equal(preds[:,n:], rights[:,n:])#go + query cut
        #accu = np.mean(np.sum(a, -1) / (rights.shape[1] - n))
        accu = nequal / nright
        #print(accu)
        return accu, rights_front, preds_front, rights_rear, preds_rear

    def train(self):
        now = datetime.datetime.now()
        print(now)

        self.open_logf('train')
        max_v = 0 
        i_step = 1
        i_max = 0
        train_accu = 0

        for batch in self.train_iterator:
            try:
                self.policy.xtrain(batch['input_ids'], batch['labels'])
                now = datetime.datetime.now()
                print(now)
            except StopIteration:
                break
            print(f"step i: {i_step}")
            
            if i_step % self.by_accu == 0: #nfetch_t 단위
                now = datetime.datetime.now()
                print(now)
                train_accu, train_right_f, train_predict_f, train_right_r, train_predict_r = self.accuracy(self.train_accu_iter)
                self.evaluate('train batch', train_right_f, train_predict_f, train_right_r, train_predict_r)

                if self.test_accu_iter is not None:
                    try:
                        test_accu, test_right_f, test_predict_f, test_right_r, test_predict_r = self.accuracy(self.test_accu_iter)
                    except:
                        self.test_accu_iter = None
                        test_accu = 0
                    else:
                        self.evaluate('test batch', test_right_f, test_predict_f, test_right_r, test_predict_r)
                else:
                    test_accu = 0

                self.logf2("epoch: %d step: %d, train(A): %f, test(B): %f, B-A: %f max: %f imax: %d",
                    self.epoch_idx, i_step, train_accu, test_accu, test_accu-train_accu, max_v, i_max)

                if max_v < train_accu: 
                    max_v = train_accu
                    i_max = i_step
                    self.logf2("regist step: %d accu: %f", i_step, train_accu)
                    mp.regist(self.policy.net)
                if train_accu > 0.99999: break
                now = datetime.datetime.now()
                print(now)
                time.sleep(300)
            i_step += 1
        self.logf2("regist end step: %d accu: %f", i_step, train_accu)
        mp.regist(self.policy.net)
        self.close_logf()
        
    def test(self, nstep):
        self.open_logf('test')
        for _ in range(nstep):
            train_accu, train_right_f, train_predict_f, train_right_r, train_predict_r = self.accuracy(self.train_accu_iter)
            self.evaluate('train batch', train_right_f, train_predict_f, train_right_r, train_predict_r)
            
            if self.test_accu_iter is not None:
                try:
                    test_accu, test_right_f, test_predict_f, test_right_r, test_predict_r = self.accuracy(self.test_accu_iter)
                except:
                    self.test_accu_iter = None
                    test_accu = 0
                else:
                    self.evaluate('test batch', test_right_f, test_predict_f, test_right_r, test_predict_r)
            else:
                test_accu = 0

            self.logf2("train(A): %f, test(B): %f, B-A: %f", train_accu, test_accu, test_accu-train_accu)

def main(args, policy_args):

    print(policy_args)
    
    if args.batch_size and args.eval_every % args.batch_size != 0:
        print('WARNING: eval_every must be divisible by batch_size')
        print('Setting eval_every to', args.eval_every - args.eval_every % args.batch_size)
        args.eval_every = args.eval_every - args.eval_every % args.batch_size

    if args.export:
        export_dataset(args.datasets, cache_dir=get_local_dir(args.local_dirs))
        return

    policy_args.case = args.case #1 - policy init, data load 2 - policy init, not data load 3 - policy load, not data load
    policy = LTCUnit(policy_args)
    
    if args.loss_name == 'dpo':
        reference = LTCUnit(policy_args, policy.tokenizer)
    else:
        reference = None
    if args.loss_name in {'dpo', 'ipo', 'orpo'}:
        trainer = HATrainer(args, policy, reference)
    else:
        trainer = SFTTrainer(args, policy)
        
    if args.case == 4:#test, policy load, not data load
        trainer.test(2)
    else:
        trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--case',           default=2,     type=int, help='execution case')
    parser.add_argument('--policy_path',     default='./korpora',   type=str,   help='specify save args path')
    #parser.add_argument('--datasets', nargs='+', default = ['hh'], help='string list')
    parser.add_argument('--datasets', nargs='+', default = ['kcd'], help='string list')
    parser.add_argument('--max_length',   default=512,   type=int,   help='the number of seq length')
    parser.add_argument('--max_prompt_length',   default=128,   type=int,   help='the number of seq length')
    parser.add_argument('--n_epochs',     default=100,   type=int,   help='the number of decoder layers')
    parser.add_argument('--by_accu',    default=100,   type=int,   help='batch size')
    parser.add_argument('--n_examples',     default=None,   type=int,   help='the number of decoder layers')
    parser.add_argument('--batch_size',   default=9600,   type=int,   help='the number of seq length')
    parser.add_argument('--eval_every',   default=9600*100,   type=int,   help='the number of seq length')
    parser.add_argument('--local_dirs',     default='./cache',   type=str,   help='the number of decoder layers')
    parser.add_argument('--seed',   default=777,   type=int,   help='the number of seq length')
    parser.add_argument('--beta',   default=0.5,   type=float,   help='the number of seq length')
    parser.add_argument('--reference_free',   default=False,   type=bool,   help='the number of seq length')
    #parser.add_argument('--loss_name',     default='dpo',   type=str,   help='the number of decoder layers')
    parser.add_argument('--loss_name',     default='sft',   type=str,   help='the number of decoder layers')
    parser.add_argument('--n_eval_examples',     default=256,   type=int,   help='the number of decoder layers')
    parser.add_argument('--eval_batch_size',     default=16,   type=int,   help='the number of decoder layers')
    parser.add_argument('--export',           default=0,     type=int, help='execution case')


    args = parser.parse_args()

    print(args)

    if hasattr(args, 'policy_path'):
        from ltcc5 import load_args
        policy_args = load_args(args.policy_path)
    else:
        print('non policy')
        exit(0)

    main(args, policy_args)

    #python hatrain.py --case 2 #init, train
    #python hatrain.py --case 3 #load, train
