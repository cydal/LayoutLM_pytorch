import torch
from transformers import LayoutLMTokenizer
from layoutlm.data.funsd import FunsdDataset, InputFeatures

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import LayoutLMForTokenClassification

import pytesseract

from tqdm import tqdm
import numpy as np
import os

from PIL import Image, ImageDraw, ImageFont

import json

from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

label2color = {'question':'blue', 'answer':'green', 'header':'orange', 'other':'violet'}


# class to turn the keys of a dict into attributes (thanks Stackoverflow)
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class Params:
    """Load hyperparameters from a json file
    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5
    ```
    """

    def __init__(self, json_path):
        self.update(json_path)

    def save(self, json_path):
        """Saves parameters to json file"""
        with open(json_path, "w") as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def __str__(self) -> str:
        return str(self.__dict__)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']`"""
        return self.__dict__




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


def model_train(model, epochs, optimizer, train_dataloader, device):

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




def model_eval(model, val_dataloader, pad_token_label_id, label_map, device):

    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None

    # put model in evaluation mode
    model.eval()
    for batch in tqdm(val_dataloader, desc="Evaluating"):
        with torch.no_grad():
            input_ids = batch[0].to(device)
            bbox = batch[4].to(device)
            attention_mask = batch[1].to(device)
            token_type_ids = batch[2].to(device)
            labels = batch[3].to(device)

            # forward pass
            outputs = model(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids,
                            labels=labels)
            # get the loss and logits
            tmp_eval_loss = outputs.loss
            logits = outputs.logits

            eval_loss += tmp_eval_loss.item()
            nb_eval_steps += 1

            # compute the predictions
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = labels.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(
                    out_label_ids, labels.detach().cpu().numpy(), axis=0
                )

    # compute average evaluation loss
    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=2)

    out_label_list = [[] for _ in range(out_label_ids.shape[0])]
    preds_list = [[] for _ in range(out_label_ids.shape[0])]

    for i in range(out_label_ids.shape[0]):
        for j in range(out_label_ids.shape[1]):
            if out_label_ids[i, j] != pad_token_label_id:
                out_label_list[i].append(label_map[out_label_ids[i][j]])
                preds_list[i].append(label_map[preds[i][j]])

    results = {
        "loss": eval_loss,
        "precision": precision_score(out_label_list, preds_list),
        "recall": recall_score(out_label_list, preds_list),
        "f1": f1_score(out_label_list, preds_list),
    }
    print(results)



def normalize_box(box, width, height):
    return [
        int(1000 * (box[0] / width)),
        int(1000 * (box[1] / height)),
        int(1000 * (box[2] / width)),
        int(1000 * (box[3] / height)),
    ]

def get_boxes(coordinates, width, height):
    actual_boxes = []
    for idx, row in coordinates.iterrows():
        x, y, w, h = tuple(row) # the row comes in (left, top, width, height) format
        actual_box = [x, y, x+w, y+h] # we turn it into (left, top, left+widght, top+height) to get the actual box 
        actual_boxes.append(actual_box)

    boxes = []
    for box in actual_boxes:
        boxes.append(normalize_box(box, width, height))
        
    return(actual_boxes, boxes)

def infer_pipeline(model, images, tokenizer, device, infer_path, save_path, label_map, args):
  all_boxes = []
  for image in images:

    img = Image.open(image).convert("RGB")

    width, height = img.size
    w_scale = 1000/width
    h_scale = 1000/height

    ocr_df = tesseract_imgs(img, w_scale, h_scale)
                  
    words = list(ocr_df.text)
    coordinates = ocr_df[['left', 'top', 'width', 'height']]


    actual_boxes, boxes = get_boxes(coordinates, width, height)

    input_ids, input_mask, segment_ids, token_boxes, token_actual_boxes = convert_example_to_features(image=img, words=words, 
                                                                                                      boxes=boxes, actual_boxes=actual_boxes, 
                                                                                                      tokenizer=tokenizer, args=args)

    print(tokenizer.decode(input_ids))

    input_ids = torch.tensor(input_ids, device=device).unsqueeze(0)
    attention_mask = torch.tensor(input_mask, device=device).unsqueeze(0)
    token_type_ids = torch.tensor(segment_ids, device=device).unsqueeze(0)
    bbox = torch.tensor(token_boxes, device=device).unsqueeze(0)
    outputs = model(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids)

    # the predictions are at the token level
    token_predictions = outputs.logits.argmax(-1).squeeze().tolist() 

    # let's turn them into word level predictions
    word_level_predictions = []
    final_boxes = []
    for id, token_pred, box in zip(input_ids.squeeze().tolist(), token_predictions, token_actual_boxes):
        if (tokenizer.decode([id]).startswith("##")) or (id in [tokenizer.cls_token_id, 
                                                                tokenizer.sep_token_id, 
                                                                tokenizer.pad_token_id]):
            continue
        else:
            word_level_predictions.append(token_pred)
            final_boxes.append(box)

    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    for prediction, box in zip(word_level_predictions, final_boxes):
        predicted_label = iob_to_label(label_map[prediction]).lower()
        draw.rectangle(box, outline=label2color[predicted_label])
        draw.text((box[0] + 10, box[1] - 10), text=predicted_label, fill=label2color[predicted_label], font=font)

    img.save(os.path.join(save_path, image.split("/")[-1]), "PNG")




def tesseract_imgs(img, w_scale, h_scale):
  ocr_df = pytesseract.image_to_data(img, output_type='data.frame')
  ocr_df = ocr_df.dropna() \
                .assign(left_scaled = ocr_df.left*w_scale,
                        width_scaled = ocr_df.width*w_scale,
                        top_scaled = ocr_df.top*h_scale,
                        height_scaled = ocr_df.height*h_scale,
                        right_scaled = lambda x: x.left_scaled + x.width_scaled,
                        bottom_scaled = lambda x: x.top_scaled + x.height_scaled)
  return(ocr_df)


def convert_example_to_features(image, words, boxes, actual_boxes, tokenizer, args, cls_token_box=[0, 0, 0, 0],
                                 sep_token_box=[1000, 1000, 1000, 1000],
                                 pad_token_box=[0, 0, 0, 0]):
      width, height = image.size

      tokens = []
      token_boxes = []
      actual_bboxes = [] # we use an extra b because actual_boxes is already used
      token_actual_boxes = []
      for word, box, actual_bbox in zip(words, boxes, actual_boxes):
          word_tokens = tokenizer.tokenize(word)
          tokens.extend(word_tokens)
          token_boxes.extend([box] * len(word_tokens))
          actual_bboxes.extend([actual_bbox] * len(word_tokens))
          token_actual_boxes.extend([actual_bbox] * len(word_tokens))

      # Truncation: account for [CLS] and [SEP] with "- 2". 
      special_tokens_count = 2 
      if len(tokens) > args.max_seq_length - special_tokens_count:
          tokens = tokens[: (args.max_seq_length - special_tokens_count)]
          token_boxes = token_boxes[: (args.max_seq_length - special_tokens_count)]
          actual_bboxes = actual_bboxes[: (args.max_seq_length - special_tokens_count)]
          token_actual_boxes = token_actual_boxes[: (args.max_seq_length - special_tokens_count)]

      # add [SEP] token, with corresponding token boxes and actual boxes
      tokens += [tokenizer.sep_token]
      token_boxes += [sep_token_box]
      actual_bboxes += [[0, 0, width, height]]
      token_actual_boxes += [[0, 0, width, height]]
      
      segment_ids = [0] * len(tokens)

      # next: [CLS] token
      tokens = [tokenizer.cls_token] + tokens
      token_boxes = [cls_token_box] + token_boxes
      actual_bboxes = [[0, 0, width, height]] + actual_bboxes
      token_actual_boxes = [[0, 0, width, height]] + token_actual_boxes
      segment_ids = [1] + segment_ids

      input_ids = tokenizer.convert_tokens_to_ids(tokens)

      # The mask has 1 for real tokens and 0 for padding tokens. Only real
      # tokens are attended to.
      input_mask = [1] * len(input_ids)

      # Zero-pad up to the sequence length.
      padding_length = args.max_seq_length - len(input_ids)
      input_ids += [tokenizer.pad_token_id] * padding_length
      input_mask += [0] * padding_length
      segment_ids += [tokenizer.pad_token_id] * padding_length
      token_boxes += [pad_token_box] * padding_length
      token_actual_boxes += [pad_token_box] * padding_length

      assert len(input_ids) == args.max_seq_length
      assert len(input_mask) == args.max_seq_length
      assert len(segment_ids) == args.max_seq_length
      #assert len(label_ids) == args.max_seq_length
      assert len(token_boxes) == args.max_seq_length
      assert len(token_actual_boxes) == args.max_seq_length
      
      return input_ids, input_mask, segment_ids, token_boxes, token_actual_boxes



def iob_to_label(label):
  if label != 'O':
    return label[2:]
  else:
    return "other"