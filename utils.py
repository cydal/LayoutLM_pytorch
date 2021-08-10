from transformers import LayoutLMTokenizer
from layoutlm.data.funsd import FunsdDataset, InputFeatures
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
import numpy as np
from transformers import LayoutLMForTokenClassification
import torch
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)


# class to turn the keys of a dict into attributes (thanks Stackoverflow)
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self



def get_labels(path):
    with open(path, "r") as f:
        labels = f.read().splitlines()
    if "O" not in labels:
        labels = ["O"] + labels
    return labels



def get_pretrained_tokenizer(name="microsoft/layoutlm-base-uncased"):
    tokenizer = LayoutLMTokenizer.from_pretrained(name)
    return(tokenizer)


def get_dataLoader(args, tokenizer, labels, pad_token_label_id, mode, batch_size):
    # the LayoutLM authors already defined a specific FunsdDataset, so we are going to use this here
    dataset = FunsdDataset(args, tokenizer, labels, pad_token_label_id, mode)
    sampler = RandomSampler(dataset) if mode == 'train' else SequentialSampler(dataset)
    dataloader = DataLoader(dataset,
                                sampler=sampler,
                                batch_size=batch_size)
    return(dataloader)



def get_device():
    return(torch.device("cuda" if torch.cuda.is_available() else "cpu"))



def get_pretrained_model(num_labels, name="microsoft/layoutlm-base-uncased"):
    model = LayoutLMForTokenClassification.from_pretrained(name, num_labels=num_labels)
    return(model)


def train(model, epochs, optimizer, train_dataloader, device):

    global_step = 0
    t_total = len(train_dataloader) * epochs
    #put the model in training mode
    model.train()
    for epoch in range(epochs):
        for batch in tqdm(train_dataloader, desc="Training"):
            input_ids = batch[0].to(device)
            bbox = batch[4].to(device)
            attention_mask = batch[1].to(device)
            token_type_ids = batch[2].to(device)
            labels = batch[3].to(device)

            # forward pass
            outputs = model(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids,
                            labels=labels)
            loss = outputs.loss

            # print loss every 100 steps
            if global_step % 100 == 0:
                print(f"Loss after {global_step} steps: {loss.item()}")

            # backward pass to get the gradients 
            loss.backward()

            # update
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1