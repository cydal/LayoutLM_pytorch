# LayoutLM_pytorch


The layoutLM by Microsoft is a text and layout image understanding solution. 
It is built on top of the BERT transformer architecture
with two additional input embeddings. The first is a 2-d positional encoding to 
capture the relative positional information

> Taking form understanding as an example, given a
key in a form (e.g., “Passport ID:”), its corresponding value is much
more likely on its right or below instead of on the left or above.


[![image.png](https://i.postimg.cc/RCjXsgpB/image.png)](https://postimg.cc/Jt5Nt5W6)


The second is an image embedding of the image tokens that correspond to the text features. 
Since a document is a combination of textual and visual information, image embedding allows to capture information
that would not normally be present in text like font and boldness.


LayoutLM can be used to extract content and structure information from forms. 
The model is fine-tuned on the FUNSD dataset. It contains almost 200 scanned 
documents, and over 9K semantic entities, and 31K+ words. In each semantic
entity is a unique identifier, label (header, question, answer) and bounding box. 


In pre-training, the document image is passed through an OCR engine (TesseractOCR) 
which returns the recognized text information along with their locations. The
text tokens are passed into the layoutLM architecture while the location
information is used to generate image embeddings for the token images using faster-RCNN.


For inference, a scanned document image is once again passed through tesseractOCR to extract
the text and location information. This information is used to generate text, image
and positional embeddings and passed to the model. For each token, the model predicts one of 
['B-ANSWER', 'B-HEADER', 'B-QUESTION', 'E-ANSWER', 'E-HEADER', 'E-QUESTION', 'I-ANSWER', 'I-HEADER', 
'I-QUESTION', 'O', 'S-ANSWER', 'S-HEADER', 'S-QUESTION']. The B,I,E,O,S tags indicate whether the token
is at the **B**eginning, **I**nside, **E**nd, **O**utside of a given entity. 


### Results - Original
[![image.png](https://i.postimg.cc/52F1cKbS/image.png)](https://postimg.cc/21mghwq1)


### Output
[![image.png](https://i.postimg.cc/26CfZzv0/image.png)](https://postimg.cc/B87yWfmF)


#### to run model
> python model.py --model_dir PATH_TO_PARAMS_FILE

### for inference
> python infer.py --model_dir PATH_TO_PARAMS_FILE


#### params
params.json file contains model parameters including

* number of epochs
* batch size & learning rate
* train/evaluation document folders
* inference file folder
* model and inference save folder



References

* https://arxiv.org/pdf/1912.13318.pdf
* https://arxiv.org/pdf/1905.13538.pdf
* https://huggingface.co/transformers/model_doc/layoutlm.html
* https://github.com/NielsRogge/Transformers-Tutorials
* https://guillaumejaume.github.io/FUNSD/
* FUNSD: Form Understanding in Noisy Scanned Documents

