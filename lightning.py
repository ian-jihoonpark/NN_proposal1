import json
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data
from torch.optim import AdamW
from pytorch_lightning import LightningModule
from PIL import Image
from torchvision import transforms
from nlx_gpt.gpt import NLX_GPT as nlx_gpt
from nlx_gpt.modules import ImageEncoder
from transformers import (
    top_k_top_p_filtering,
    get_linear_schedule_with_warmup,
    GPT2Tokenizer, 
    AutoConfig,
    AutoConfig, 
    top_k_top_p_filtering, 
    )
from nlx_gpt.distilledBERT import DistilBertForSequenceClassification
from nlx_gpt.gpt import GPT2LMHeadModel

from utils import top_filtering, filter_and_get_scores
import os
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from PIL import Image
import os 
import logging
logger = logging.getLogger(__name__)
class NLX_GPT(LightningModule):
    def __init__(self, tokenizer, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.tokenizer = tokenizer
        self.mode = None
        self.img_encoded = hparams.img_encoded

        config = AutoConfig.from_pretrained('distilgpt2')
        # Add configs
        # setattr(config, 'max_seq_len', None)   
        config.img_size = hparams.img_size
        # config.max_seq_len = hparams.max_seq_len 
        config.add_cross_attention = True
        config.relevance_map = False
        
        # Load model
        self.image_encoder = ImageEncoder(self.device)
        self.lm_model = GPT2LMHeadModel.from_pretrained('distilgpt2', config = config)
        self.model = nlx_gpt(self.image_encoder, self.lm_model)
        self.model.lm.resize_token_embeddings(len(self.tokenizer))
        self.weight_ckpt_pth = os.path.join(hparams.checkpoints_dir,hparams.experiment_name)
        self.pre_loss = torch.tensor(10.0)

    def setup(self,stage):
        if stage=="fit":
            self.total_steps = len(self.trainer.datamodule) // self.hparams.gradient_accumulation_steps // self.hparams.ngpu * self.hparams.max_epochs
            self.warmup_steps = self.hparams.warmup_steps
        elif stage=="test" or stage=="predict":
            self.results_full = []
            self.results_exp = []
            self.eval_results = []
            
            SEG_TOKENS = ['<question>', '<answer>', '<explanation>']
            self.seg_token_ids = self.tokenizer.convert_tokens_to_ids(SEG_TOKENS)
            self.because_token_id = self.tokenizer.convert_tokens_to_ids('Ġbecause')
            self.eos_token_id = [self.tokenizer.eos_token_id]
            self.special_token_ids = [self.tokenizer.eos_token_id, self.tokenizer.pad_token_id] + self.seg_token_ids
        
    
    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.total_steps,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "name": "lr"}
        return [optimizer], [scheduler]

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self,  batch, batch_idx):
        outputs = self(
            input_ids=batch["inputs"],
            segment_ids=batch["segment_ids"],
            image=batch["image"],
            labels=batch["labels"],
            )
        loss = outputs.loss
        # self.log(f"{self.hparams.selfe_mode}_train_loss", loss)
        self.log(f"{self.hparams.model_path}_train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch["inputs"],
            segment_ids=batch["segment_ids"],
            image=batch["image"],
            labels=batch["labels"],
            )
        
        loss = outputs.loss
        
        if self.pre_loss > loss:
            self.model.lm.save_pretrained(self.weight_ckpt_pth)
            self.pre_loss = loss
        # self.log(f"{self.hparams.selfe_mode}_val_loss", loss)
        self.log(f"{self.hparams.model_path}_val_loss", loss)

        return loss

    def test_step(self,batch,batch_idx):
        
        max_len = 20
        always_exp = False
        always_ans = False
        no_sample = True
        current_output = []
        current_logits = []
        do_sample = False
        
        input_ids = batch["input_ids"]
        segment_ids = batch["segment_ids"]
        for step in range(max_len + 1):
            if step == max_len:
                break
            outputs = self(
                input_ids=input_ids,
                segment_ids=segment_ids,
                image=batch["img"],
                labels = None,
                )
            
            lm_logits = outputs.logits 
            logits = lm_logits[:, -1, :] / self.hparams.temperature
            filtered_logits = top_k_top_p_filtering(logits, top_k=self.hparams.top_k, top_p=self.hparams.top_p)
            probs = F.softmax(filtered_logits, dim=-1)
            prev = torch.multinomial(probs, dim=-1) if do_sample else torch.argmax(filtered_logits, dim=-1).unsqueeze(-1)
            if prev.item() in self.special_token_ids:
                break
            
            if not always_exp:
                if prev.item() != self.because_token_id:
                    new_segment = self.seg_token_ids[-2]   # answer segment
                else:
                    new_segment = self.seg_token_ids[-1]   # explanation segment
                    always_exp = True
            else:
                new_segment = self.seg_token_ids[-1]   # explanation segment                  
                    
            new_segment = torch.LongTensor([new_segment]).to(torch.cuda.current_device())
            current_output.append(prev.item())
            current_logits.append(logits.unsqueeze(1))
            input_ids = torch.cat((input_ids, prev), dim=1)
            segment_ids = torch.cat((segment_ids, new_segment.unsqueeze(0).expand(segment_ids.shape[0],-1)), dim=1)
        decoded_sequences = self.tokenizer.decode(current_output, skip_special_tokens=True).lstrip()
        question = self.tokenizer.decode(batch["input_ids"][0], skip_special_tokens=True).lstrip()
        
        image_name = batch["image_path"][0].split("/")[-1]
        self.results_full.append({"image_id" :batch["image_id"][0] ,"image_pth": image_name, "caption": decoded_sequences, "question": question})
        if 'because' in decoded_sequences:
            cut_decoded_sequences = decoded_sequences.split('because')[-1].strip()
        else:
            cut_decoded_sequences = decoded_sequences.split('so the answer is')[0].strip()   
        
        self.eval_results.append({"image_pth": image_name,"question": question, "gt_explanation": cut_decoded_sequences, "image_id" :batch["image_id"][0],"image_name": image_name})
        self.results_exp.append({"image_id" :batch["image_id"][0], "image_pth": image_name, "caption": cut_decoded_sequences})         
        return {"reults_full" : self.results_full, "results_exp": self.results_exp}


    def test_epoch_end(self, batch_parts):
        if not os.path.exists(self.hparams.output_dir):
            os.mkdir(self.hparams.output_dir)

        resFileExp = os.path.join(self.hparams.output_dir , 'captions.json')
        unf_resFileExp = os.path.join(self.hparams.output_dir , 'unf_captions.json') 
        unf_resFileFull = os.path.join(self.hparams.output_dir , 'unf_captions_full.json')
        save_scores_pathExp = os.path.join(self.hparams.output_dir , 'scores.json')
        save_eval_datasets = os.path.join(self.hparams.output_dir , 'eval_data.json')
        
        with open(unf_resFileExp, 'w') as w:
            json.dump(self.results_exp, w)
            
        with open(unf_resFileFull, 'w') as w:
            json.dump(self.results_full, w)

        with open(save_eval_datasets, 'w') as w:
            json.dump(self.eval_results, w)
        
        filter_and_get_scores(resFileExp, save_scores_pathExp, self.results_full, self.results_exp)

    
    
    
class ExplainPredict(LightningModule):
    def __init__(self, tokenizer, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        # Load model
        self.model = DistilBertForSequenceClassification.from_pretrained(hparams.model_path)
        self.tokenizer = tokenizer
        
        answer_dic_pth = os.path.join(hparams.vqax_test_anno_path, f"answer_dic.json")
        tst_data_pth = os.path.join(hparams.vqax_test_anno_path, f"test_data.json")
        self.tst_data = json.load(open(tst_data_pth, 'r'))
        self.answer_dic = json.load(open(answer_dic_pth, 'r'))
        self.inv_answer_dic = {v: k for k, v in self.answer_dic.items()}
        
        
    def setup(self,stage):
        if stage=="fit":
            self.total_steps = len(self.trainer.datamodule) // self.hparams.gradient_accumulation_steps // self.hparams.ngpu * self.hparams.max_epochs
            self.warmup_steps = self.hparams.warmup_steps
        elif stage=="test" or stage=="predict":
            self.correct_ids = []
    
    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.total_steps,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "name": "lr"}
        return [optimizer], [scheduler]

    def forward(self, **inputs):
        return self.model(**inputs)

    def test_step(self,  batch, batch_idx):
        qid = batch["qid"]
        predictions = self(input_ids=batch["input_ids"],
                                    attention_mask=batch["attention_masks"], 
                                    head_mask=None, 
                                    inputs_embeds=None,
                                    labels=None,
                                    output_attentions=False,
                                    output_hidden_states=False,
                                    return_dict=True)
        _, ind = torch.max(predictions.logits, dim=-1)
        for i,pred in enumerate(ind):
            pred_answer = self.inv_answer_dic[pred.item()]
            gt_answers = list(set(self.tst_data[str(qid[i].item())]['all_answers_raw']))
            if pred_answer in gt_answers:
                self.correct_ids.append(qid[i].item())    
        return self.correct_ids


    def test_epoch_end(self, batch_parts):
        accuracy = len(self.correct_ids) / len(self.trainer.datamodule)
        print("Accuracy: {:.2f} %".format(accuracy * 100))
        
        if not os.path.exists(self.hparams.output_dir):
            os.mkdir(self.hparams.output_dir)
            
        save_correct_ids = os.path.join(self.hparams.output_dir, f"correct_ids")
        with open(save_correct_ids, 'w') as w:
            json.dump(self.correct_ids, w)
        
        return accuracy
    
    
