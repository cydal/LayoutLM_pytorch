from torch.nn import CrossEntropyLoss
from utils import *
from transformers import AdamW






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



device = get_device()

model = get_pretrained_model(num_labels)
model.to(device)


### Fine-tune
optimizer = AdamW(model.parameters(), lr=5e-5)
epochs = 5


model_train(model, epochs, optimizer, train_dataloader, device)
model_eval(model, val_dataloader, device)