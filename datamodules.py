import json, os, random
from sys import implementation
from tkinter import NONE
from pytorch_lightning import LightningDataModule
from tqdm import tqdm
from PIL import ImageFile, Image

ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np
import utils
from typing import Optional
import logging
logger = logging.getLogger(__name__)


import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data import Dataset
import json
import numpy as np
    
class NLX_GPT_BaseDataset(LightningDataModule):
    def __init__(
        self,
        tokenizer,
        hparams,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(hparams)
        
        self.cfg = hparams
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        self.resolution = hparams.img_size
        self.patch_resize_transform = transforms.Compose([
        lambda image: image.convert("RGB"),
        transforms.Resize((self.resolution, self.resolution), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(), 
        transforms.Normalize(mean=mean, std=std)
        ])
        self.tokenizer = tokenizer
        self.nle_anno_path = self.cfg.nle_anno_path
        self.image_dir = self.cfg.nle_image_dir
        
        num_new_tokens = self.tokenizer.add_special_tokens({'pad_token': '<pad>','additional_special_tokens': ['<question>', '<answer>', '<explanation>']})
        self.q_segment_id, self.a_segment_id, self.e_segment_id = self.tokenizer.convert_tokens_to_ids(['<question>', '<answer>', '<explanation>'])
        self.dataset = {}
        self.dataset["test"] = self.get_data(is_train = "test")
        self.dataset["train"] = self.get_data(is_train = "train")
        self.dataset["validation"] = self.get_data(is_train = "val")         
    
    def get_data(self, is_train = None):
        file_name = f"vqax_{is_train}.json" 
        data_path = os.path.join(self.nle_anno_path,file_name)
        
        cached_features_file = os.path.join(self.cfg.cached_dir, f"cached_nlx_total_{is_train}.pt")
        # Init features and dataset from cache if it exists
        if os.path.exists(cached_features_file):
            logger.info("Loading features from cached file %s", cached_features_file)
            features_and_dataset = torch.load(cached_features_file)
            datasets = features_and_dataset["datasets"]
        else:
            data = json.load(open(data_path, 'r'))
        
            ids_list = list(data.keys())
            index_tracker = {k: len(v['explanation']) - 1 for k,v in data.items()}
            
            for k,v in tqdm(data.items(), desc= "Data to list and dictionary..."):   
                if len(v['explanation']) > 1:   # some questions have more than one explanation
            # duplicate them for loading. -1 because one explanation is already in ids_list
                    ids_list += [str(k)] * (len(v['explanation']) - 1)
                
                
            datasets = []
            for i in tqdm(range(len(data)), desc= f"nlx_gpt_{is_train}_VQA-X preprocessing..."):
                
                question_id = ids_list[i]
                sample = data[question_id]
                img_name = sample['image_name']
                image_id = sample["image_id"]
                question_txt = utils.proc_ques(sample['question'])    # question
                question_txt = f" {question_txt}?"
                answer_txt = utils.proc_ans(sample['answers'])
                exp_idx = index_tracker[question_id]
                
                if is_train == "train":
                    img_path = os.path.join(os.path.join(self.image_dir, "train2014"), img_name)
                else:
                    img_path = os.path.join(os.path.join(self.image_dir, "val2014"), img_name)
                
                
                # if one more explanations
                if exp_idx > 0:
                    index_tracker[question_id] -= 1    # decrease usage
                    
                explain = sample['explanation'][exp_idx]
                
                # For question
                question_txt = self.tokenizer.tokenize(question_txt)
                labels = [-100] * len(question_txt)
                segment_ids = [self.q_segment_id] * len(question_txt)
                
                if is_train == "train" or is_train =="val":    
                    explain_txt = f" because {explain} {self.tokenizer.eos_token}"
                    answer_txt = f"{self.tokenizer.bos_token} the answer is {answer_txt}"
                    
                    answer_txt = self.tokenizer.tokenize(answer_txt)
                    explain_txt = self.tokenizer.tokenize(explain_txt)
                    
                    labels += [-100] + answer_txt[1:] + explain_txt
                    input_ids = question_txt + answer_txt + explain_txt
                    
                    exp_len = len(explain_txt)
                    answer_len = len(answer_txt)

                    segment_ids += [self.a_segment_id] * answer_len
                    segment_ids += [self.e_segment_id] * exp_len
                    
                else:
                    answer_txt = f"{self.tokenizer.bos_token} the answer is"
                    
                    question_txt += self.tokenizer.tokenize(answer_txt)
                    answer_len = len(self.tokenizer.tokenize(answer_txt))
                    segment_ids += [self.a_segment_id] * answer_len
                    input_ids = self.tokenizer.convert_tokens_to_ids(question_txt)

                        
                img = Image.open(img_path)
                img_ids = self.patch_resize_transform(img)
                
                input_ids = torch.tensor(input_ids, dtype=torch.long)
                segment_ids = torch.tensor(segment_ids, dtype=torch.long)
                if is_train == "train" or is_train =="val":
                    labels = torch.tensor(labels, dtype=torch.long)
                    datasets.append({"image_path": img_path, "input_ids" : input_ids,"segment_ids": segment_ids, "labels": labels, "img": img_ids, "image_id": image_id })
                    
                else:
                    datasets.append({"image_path": img_path, "input_ids" : input_ids,"segment_ids": segment_ids, "img": img_ids, "image_id": image_id })
            torch.save({"datasets": datasets}, cached_features_file)
        
        return datasets
    
    def collate_fn(self, batch):
        # Collate function definition
        value_lst = [list(lst.values()) for lst in batch]
        batch = list(zip(*value_lst))
        sample = {}
        
        # max len
        input_max_len = max([x.size(0) for x in batch[1]])
        
        if self.cfg.max_seq_len < input_max_len:
            input_max_len = self.cfg.max_seq_len
        else:
            pass
        
        input_ids = []
        segment_ids = []
        labels = []


        if batch[3][0] is not None:
            # input ids and segmend ids padding
            for idx,input in enumerate(batch[1]):
                if len(input) > input_max_len:
                    input_ids.append(batch[1][idx][:input_max_len])
                    segment_ids.append(batch[2][idx][:input_max_len])
                    labels.append(batch[3][idx][:input_max_len])
                else:
                    padding_len = input_max_len - len(input)
                    input_ids.append(torch.cat([batch[1][idx], torch.tensor([self.tokenizer.pad_token_id] * padding_len, dtype = torch.long)]))
                    segment_ids.append(torch.cat([batch[2][idx], torch.tensor([self.tokenizer.pad_token_id] * padding_len, dtype = torch.long)]))
                    labels.append(torch.cat([batch[3][idx], torch.tensor([self.tokenizer.pad_token_id] * padding_len, dtype = torch.long)]))

            
    # Stack
        sample["inputs"] = torch.stack(input_ids, 0)
        sample["labels"] = torch.stack(labels, 0)
        sample["image_path"] = batch[0]
        sample["image"] = torch.stack(batch[4], 0)
        sample["image_id"] = batch[5]
        sample["segment_ids"] = torch.stack(segment_ids, 0)

        return sample
    
    def __len__(self):
        return len(self.dataset["train"])
        
    def train_dataloader(self):
        return DataLoader(self.dataset["train"], shuffle=True, batch_size=self.cfg.train_batch_size, \
                        pin_memory=True, num_workers=self.cfg.n_train_workers, collate_fn = self.collate_fn)
    def val_dataloader(self):
        return DataLoader(self.dataset["validation"], batch_size=self.cfg.eval_batch_size, \
            pin_memory=True, num_workers=self.cfg.n_valid_workers, collate_fn = self.collate_fn)
    
    def test_dataloader(self):
        return DataLoader(self.dataset["test"], batch_size=1, pin_memory=True, num_workers=self.cfg.n_test_workers)