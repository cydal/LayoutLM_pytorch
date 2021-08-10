from utils import *
import glob2
import argparse
import pytesseract
import os

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir",
                    default=".",
                    help="Directory containing params file")



if __name__ == "__main__":
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, "PARAMS.JSON")
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    file_path = os.path.join(params.INFER_PATH, '**/*.png')
    files = glob2.glob(file_path)

    print(f"-----------------Images Path -----------------")
    print(files)

    print(f"-----------------Document-----------------")

    labels = get_labels("data/labels.txt")
    num_labels = len(labels)
    label_map = {i: label for i, label in enumerate(labels)}

    args = {'local_rank': -1,
            'overwrite_cache': True,
            'data_dir': params.DATA_DIR,
            'model_name_or_path':params.MODEL_NAME_OR_PATH,
            'max_seq_length': params.MAX_SEQ_LENGTH,
            'model_type': params.MODEL_TYPE}

    args = AttrDict(args)

    tokenizer = get_pretrained_tokenizer()
    device = get_device()

    model = torch.load(params.SAVE_MODEL_PATH)
    print(model)

    infer_pipeline(model, files, tokenizer, device, params.INFER_PATH, params.SAVE_INFER_PATH, label_map, args)
