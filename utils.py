from transformers import LayoutLMTokenizer
from layoutlm.data.funsd import FunsdDataset, InputFeatures
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


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