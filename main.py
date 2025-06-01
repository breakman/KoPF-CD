import sys
sys.path.insert(0, "/content/drive/MyDrive/PT-CD/OpenPrompt")

import tqdm
from openprompt.data_utils.text_classification_dataset import AgnewsTitleProcessor, KRnewsTitleProcessor
# from openprompt.data_utils.text_classification_dataset import  NewstitleProcessor, AgnewsTitleProcessor, SnippetsProcessor
import torch
from openprompt.data_utils.utils import InputExample
import argparse
import numpy as np
from openprompt import *

from openprompt import PromptDataLoader

from openprompt.prompts import ManualTemplate
from openprompt.prompts import CptVerbalizer, SoftVerbalizer


parser = argparse.ArgumentParser("")
parser.add_argument("--shot", type=int, default=5)
parser.add_argument("--seed", type=int, default=144)
parser.add_argument("--plm_eval_mode", action="store_true")
parser.add_argument("--model", type=str, default='bert')
parser.add_argument("--model_name_or_path", default='bert-base-cased')
parser.add_argument("--verbalizer", type=str)
parser.add_argument("--calibration", action="store_true")
parser.add_argument("--filter", default="none", type=str)
parser.add_argument("--template_id", type=int)
parser.add_argument("--dataset", type=str)
parser.add_argument("--result_file", type=str, default="./results/results.txt")
parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
parser.add_argument("--max_epochs", type=int, default=5)
parser.add_argument("--kptw_lr", default=0.06, type=float)
parser.add_argument("--pred_temp", default=1.0, type=float)
parser.add_argument("--max_token_split", default=-1, type=int)
args = parser.parse_args()

import random

this_run_unicode = str(random.randint(0, 1e10))

from openprompt.utils.reproduciblity import set_seed

set_seed(args.seed)

from openprompt.plms import load_plm

plm, tokenizer, model_config, WrapperClass = load_plm(args.model, args.model_name_or_path)

dataset = {}

if args.dataset == "DL-Clickbait":
    dataset['train'] = AgnewsTitleProcessor().get_train_examples("./datasets/TextClassification/DL-Clickbait/")
    dataset['test'] = AgnewsTitleProcessor().get_test_examples("./datasets/TextClassification/DL-Clickbait/")
    class_labels = AgnewsTitleProcessor().get_labels()
    scriptsbase = "TextClassification/DL-Clickbait"
    scriptformat = "txt"
    cutoff = 0.5
    max_seq_l = 128
    batch_s = 1

elif args.dataset == "KR-Clickbait":
    dataset['train'] = KRnewsTitleProcessor().get_train_examples("./datasets/TextClassification/KR-Clickbait/")
    dataset['test'] = KRnewsTitleProcessor().get_test_examples("./datasets/TextClassification/KR-Clickbait/")
    class_labels = KRnewsTitleProcessor().get_labels()
    scriptsbase = "TextClassification/KR-Clickbait"
    scriptformat = "txt"
    cutoff = 0.5
    max_seq_l = 128
    batch_s = 1

# elif args.dataset == "SC-Clickbait":
#     dataset['train'] = SnippetsProcessor().get_train_examples("./datasets/TextClassification/SC-Clickbait/")
#     dataset['test'] = SnippetsProcessor().get_test_examples("./datasets/TextClassification/SC-Clickbait/")
#     class_labels = SnippetsProcessor().get_labels()
#     scriptsbase = "TextClassification/snippets"
#     scriptformat = "txt"
#     cutoff = 0.5
#     max_seq_l = 128
#     batch_s = 1

# elif args.dataset == "W23-Clickbait":
#     dataset['train'] = NewstitleProcessor().get_train_examples("./datasets/TextClassification/W23-Clickbait/")
#     dataset['test'] = NewstitleProcessor().get_test_examples("./datasets/TextClassification/W23-Clickbait/")
#     class_labels = NewstitleProcessor().get_labels()
#     scriptsbase = "TextClassification/W23-Clickbait"
#     scriptformat = "txt"
#     cutoff = 0.5
#     max_seq_l = 128
#     batch_s = 5



else:
    raise NotImplementedError


mytemplate = ManualTemplate(tokenizer=tokenizer).from_file(path=f"./scripts/{scriptsbase}/manual_template.txt",
                                                           choice=args.template_id)
if args.verbalizer == "soft":
    # myverbalizer = CptVerbalizer(tokenizer, classes=class_labels, candidate_frac=cutoff, pred_temp=args.pred_temp,
    #                              max_token_split=args.max_token_split).from_file(
    #     path=f"./scripts/{scriptsbase}/cpt_verbalizer.{scriptformat}")
    myverbalizer = SoftVerbalizer(tokenizer,plm,classes=class_labels).from_file(
        path=f"./scripts/{scriptsbase}/cpt_verbalizer.{scriptformat}")
    for i, words in enumerate(myverbalizer.label_words):
        label_name = myverbalizer.classes[i] if myverbalizer.classes else i
        print(f"Label {label_name} → {words}")
elif args.verbalizer == "cpt":
    myverbalizer = CptVerbalizer(tokenizer, classes=class_labels, candidate_frac=cutoff, pred_temp=args.pred_temp,
                                 max_token_split=args.max_token_split).from_file(
        path=f"./scripts/{scriptsbase}/cpt_verbalizer.{scriptformat}")
    # myverbalizer = SoftVerbalizer(tokenizer,plm,classes=class_labels).from_file(
    #     path=f"./scripts/{scriptsbase}/cpt_verbalizer.{scriptformat}")

from openprompt import PromptForClassification

use_cuda = True
prompt_model = PromptForClassification(plm=plm, template=mytemplate, verbalizer=myverbalizer, freeze_plm=False,
                                       plm_eval_mode=args.plm_eval_mode)
if use_cuda:
    prompt_model = prompt_model.cuda()


from openprompt.data_utils.data_sampler import FewShotSampler
sampler = FewShotSampler(num_examples_per_label=args.shot, also_sample_dev=True, num_examples_per_label_dev=args.shot)
dataset['train'], dataset['validation'] = sampler(dataset['train'], seed=args.seed)
fewshot_sampler_for_test = FewShotSampler(num_examples_per_label=500)
dataset['test'] = fewshot_sampler_for_test(dataset['test'], seed=args.seed)
train_dataloader = PromptDataLoader(dataset=dataset["train"], template=mytemplate, tokenizer=tokenizer,
                                    tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l,
                                    decoder_max_length=3,
                                    batch_size=batch_s, shuffle=True, teacher_forcing=False, predict_eos_token=False,
                                    truncate_method="tail")
validation_dataloader = PromptDataLoader(dataset=dataset["validation"], template=mytemplate, tokenizer=tokenizer,
                                         tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l,
                                         decoder_max_length=3,
                                         batch_size=batch_s, shuffle=False, teacher_forcing=False,
                                         predict_eos_token=False,
                                         truncate_method="tail")
# zero-shot test
test_dataloader = PromptDataLoader(dataset=dataset["test"], template=mytemplate, tokenizer=tokenizer,
                                   tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l, decoder_max_length=3,
                                   batch_size=batch_s, shuffle=False, teacher_forcing=False, predict_eos_token=False,
                                   truncate_method="tail")

def evaluate(prompt_model, dataloader, desc):
    prompt_model.eval()
    allpreds = []
    alllabels = []
    pbar = tqdm.tqdm(dataloader, desc=desc)
    for step, inputs in enumerate(pbar):
        if use_cuda:
            inputs = inputs.cuda()
        logits = prompt_model(inputs)
        labels = inputs['label']
        input_text=tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
        alllabels.extend(labels.cpu().tolist())
        allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
    acc = sum([int(i == j) for i, j in zip(allpreds, alllabels)]) / len(allpreds)
    return acc




from transformers import AdamW, get_linear_schedule_with_warmup

loss_func = torch.nn.CrossEntropyLoss()


def prompt_initialize(verbalizer, prompt_model, init_dataloader):
    dataloader = init_dataloader
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Init_using_{}".format("train")):
            if use_cuda:
              batch = batch.cuda()
            logits = prompt_model(batch)
        verbalizer.optimize_to_initialize()


if args.verbalizer == "cpt":
    no_decay = ['bias', 'LayerNorm.weight']

    # it's always good practice to set no decay to biase and LayerNorm parameters
    optimizer_grouped_parameters1 = [
        {'params': [p for n, p in prompt_model.plm.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in prompt_model.plm.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    optimizer1 = AdamW(optimizer_grouped_parameters1, lr=3e-5)
    optimizer2 = AdamW(prompt_model.verbalizer.parameters(), lr=args.kptw_lr)

    tot_step = len(train_dataloader) // args.gradient_accumulation_steps * args.max_epochs
    scheduler1 = get_linear_schedule_with_warmup(
        optimizer1,
        num_warmup_steps=0, num_training_steps=tot_step)


    scheduler2 = None

elif args.verbalizer == "manual":
    no_decay = ['bias', 'LayerNorm.weight']

    # it's always good practice to set no decay to biase and LayerNorm parameters
    optimizer_grouped_parameters1 = [
        {'params': [p for n, p in prompt_model.plm.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in prompt_model.plm.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    # Using different optimizer for prompt parameters and model parameters

    optimizer1 = AdamW(optimizer_grouped_parameters1, lr=3e-5)

    tot_step = len(train_dataloader) // args.gradient_accumulation_steps * args.max_epochs
    scheduler1 = get_linear_schedule_with_warmup(
        optimizer1,
        num_warmup_steps=0, num_training_steps=tot_step)

    optimizer2 = None
    scheduler2 = None

elif args.verbalizer == "soft":


    no_decay = ['bias', 'LayerNorm.weight']

    # it's always good practice to set no decay to biase and LayerNorm parameters
    optimizer_grouped_parameters1 = [
        {'params': [p for n, p in prompt_model.plm.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in prompt_model.plm.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    # Using different optimizer for prompt parameters and model parameters

    optimizer_grouped_parameters2 = [
        {'params': prompt_model.verbalizer.group_parameters_1, "lr":3e-5},
        {'params': prompt_model.verbalizer.group_parameters_2, "lr":3e-4},
    ]


    optimizer1 = AdamW(optimizer_grouped_parameters1, lr=3e-5)
    optimizer2 = AdamW(optimizer_grouped_parameters2)

    tot_step = len(train_dataloader) // args.gradient_accumulation_steps * args.max_epochs
    scheduler1 = get_linear_schedule_with_warmup(
        optimizer1, 
        num_warmup_steps=0, num_training_steps=tot_step)

    scheduler2 = get_linear_schedule_with_warmup(
        optimizer2, 
        num_warmup_steps=0, num_training_steps=tot_step)
    # group1 = [
    #     {"params": prompt_model.parameters()},  # PLM + template
    #     {"params": prompt_model.verbalizer.group_parameters_1},  # head 내 중간 계층 (optional)
    # ]
    # group2 = [
    #     {"params": prompt_model.verbalizer.group_parameters_2}  # head 마지막 계층 (soft verbalizer 자체)
    # ]
    

    # optimizer3 = AdamW(group1, lr=3e-5)
    # optimizer4 = AdamW(group2, lr=1e-3)  # 보통 더 큰 lr을 씀
    # scheduler3 = get_linear_schedule_with_warmup(
    #     optimizer3, 
    #     num_warmup_steps=0, num_training_steps=tot_step)

    # scheduler4 = get_linear_schedule_with_warmup(
    #     optimizer4, 
    #     num_warmup_steps=0, num_training_steps=tot_step)

tot_loss = 0
log_loss = 0
best_val_acc = 0
classes = ["낚시성", "비낚시성"]
for epoch in range(args.max_epochs):
    tot_loss = 0
    prompt_model.train()
    for step, inputs in enumerate(train_dataloader):
        if use_cuda:
            inputs = inputs.cuda()
        logits = prompt_model(inputs)
        labels = inputs['label']
        loss = loss_func(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(prompt_model.parameters(), 1.0)
        tot_loss = tot_loss + loss.item()
        print(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}")  # step별 로스
        optimizer1.step()
        scheduler1.step()
        optimizer1.zero_grad()
        if optimizer2 is not None:
            optimizer2.step()
            optimizer2.zero_grad()
        if scheduler2 is not None:
            scheduler2.step()
        
        preds = torch.argmax(logits, dim=-1)
        input_text = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
        predicted_label = classes[preds]
        # print(f"Input: {input_text} → Predicted class: {predicted_label}")
    val_acc = evaluate(prompt_model, validation_dataloader, desc="Valid")
    if val_acc >= best_val_acc:
        # torch.save(prompt_model.state_dict(), f"./ckpts/{this_run_unicode}.ckpt")
        torch.save(prompt_model.state_dict(), f"/content/{this_run_unicode}.ckpt")
        print(f"save {this_run_unicode}.ckpt")
        best_val_acc = val_acc
    avg_loss = tot_loss / len(train_dataloader)
    print(f"Epoch {epoch} finished. Avg Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}")  # epoch별 평균 로스


# prompt_model.load_state_dict(torch.load(f"./ckpts/{this_run_unicode}.ckpt"))
prompt_model.load_state_dict(torch.load(f"/content/{this_run_unicode}.ckpt"))
# Step 1: 전체 vocab embedding 불러오기
vocab_embeddings = plm.get_input_embeddings().weight  # shape: (vocab_size, hidden_dim)

# Step 2: soft verbalizer의 head weight 가져오기
soft_weights = prompt_model.verbalizer.head_last_layer.weight.detach()

# Step 3: cosine similarity로 가장 유사한 단어 찾기
import torch.nn.functional as F

topk = 30
for i, label in enumerate(prompt_model.verbalizer.classes):
    sims = F.cosine_similarity(soft_weights[i].unsqueeze(0), vocab_embeddings)
    topk_vals, topk_ids = torch.topk(sims, topk)
    tokens = tokenizer.convert_ids_to_tokens(topk_ids.tolist())
    print(f"Label '{label}' → Top-{topk} closest tokens: {tokens}")
if use_cuda:
  prompt_model = prompt_model.cuda()
test_acc = evaluate(prompt_model, test_dataloader, desc="Test")

content_write = "=" * 20 + "\n"
content_write += f"dataset {args.dataset}\t"
content_write += f"temp {args.template_id}\t"
content_write += f"seed {args.seed}\t"
content_write += f"shot {args.shot}\t"
content_write += f"verb {args.verbalizer}\t"
content_write += f"cali {args.calibration}\t"
content_write += f"filt {args.filter}\t"
content_write += f"maxsplit {args.max_token_split}\t"
content_write += f"kptw_lr {args.kptw_lr}\t"
content_write += "\n"
content_write += f"Acc: {test_acc}"
content_write += "\n\n"

print(content_write)

with open(f"{args.result_file}", "a") as fout:
    fout.write(content_write)

import os

# os.remove(f"./ckpts/{this_run_unicode}.ckpt")