from torch.nn import CrossEntropyLoss
from utils import *
from transformers import AdamW
import argparse
import os
from PIL import Image, ImageDraw, ImageFont



parser = argparse.ArgumentParser()
parser.add_argument("--model_dir",
                    default=".",
                    help="Directory containing params file")



if __name__ == "__main__":
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, "PARAMS.JSON")
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)




labels = get_labels("data/labels.txt")
num_labels = len(labels)
label_map = {i: label for i, label in enumerate(labels)}
# Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
pad_token_label_id = CrossEntropyLoss().ignore_index

print(f"-----------------Printing Labels-----------------")
print(labels)


args = {'local_rank': -1,
        'overwrite_cache': True,
        'data_dir': params.DATA_DIR,
        'model_name_or_path':params.MODEL_NAME_OR_PATH,
        'max_seq_length': params.MAX_SEQ_LENGTH,
        'model_type': params.MODEL_TYPE,}

args = AttrDict(args)

tokenizer = get_pretrained_tokenizer()
train_dataloader = get_dataLoader(args, tokenizer, labels, pad_token_label_id, 'train', params.BATCH_SIZE)
val_dataloader = get_dataLoader(args, tokenizer, labels, pad_token_label_id, 'test', params.BATCH_SIZE)

print(f"-----------------Train Loader-----------------")
print(len(train_dataloader))

print(f"-----------------Val Loader-----------------")
print(len(val_dataloader))


batch = next(iter(train_dataloader))
input_ids = batch[0][0]

print(f"-----------------Tokenizer Loader-----------------")
print(tokenizer.decode(input_ids))

device = get_device()

model = get_pretrained_model(num_labels)
model.to(device)

print(model)

### Fine-tune

optimizer = AdamW(model.parameters(), lr=params.LR)


model_train(model, params.EPOCHS, optimizer, train_dataloader, device)
model_eval(model, val_dataloader, pad_token_label_id, label_map, device)

torch.save(model, params.SAVE_MODEL_PATH)