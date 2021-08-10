from torch.nn import CrossEntropyLoss
from utils import *






labels = get_labels("data/labels.txt")
num_labels = len(labels)
label_map = {i: label for i, label in enumerate(labels)}
# Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
pad_token_label_id = CrossEntropyLoss().ignore_index


print(labels)

args = {'local_rank': -1,
        'overwrite_cache': True,
        'data_dir': '/content/data',
        'model_name_or_path':'microsoft/layoutlm-base-uncased',
        'max_seq_length': 512,
        'model_type': 'layoutlm',}

args = AttrDict(args)

tokenizer = get_pretrained_tokenizer()
train_dataloader = get_dataLoader(args, tokenizer, labels, pad_token_label_id, 'train', 2)
val_dataloader = get_dataLoader(args, tokenizer, labels, pad_token_label_id, 'test', 2)

print(len(train_dataloader))
print(len(val_dataloader))


batch = next(iter(train_dataloader))
input_ids = batch[0][0]

print(tokenizer.decode(input_ids))


from transformers import LayoutLMForTokenClassification
import torch

device = get_device()

model = LayoutLMForTokenClassification.from_pretrained("microsoft/layoutlm-base-uncased", num_labels=num_labels)
model.to(device)