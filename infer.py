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

    files = glob2.glob('**/*.png')

    print(f"-----------------Images Path -----------------")
    print(files)

    print(f"-----------------Document-----------------")

    tokenizer = get_pretrained_tokenizer()
    device = get_device()

    infer_pipeline(files, tokenizer, device, params.INFER_PATH, args)