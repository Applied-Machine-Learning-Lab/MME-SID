from PIL import Image
from transformers import AutoModel, AutoConfig, AutoTokenizer
from transformers import CLIPImageProcessor
import torch
from llm2vec import LLM2Vec
import os
import requests as req
from io import BytesIO
import pandas as pd
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

processor = CLIPImageProcessor.from_pretrained("/LLM2CLIP/clip-vit-large-patch14-336")
model_name_or_path = "/LLM2CLIP/LLM2CLIP-Openai-L-14-336" # or /path/to/local/LLM2CLIP-Openai-L-14-336
model = AutoModel.from_pretrained(
    model_name_or_path, 
    torch_dtype=torch.bfloat16,
    trust_remote_code=True).to('cuda').eval()

llm_model_name = '/LLM2CLIP/LLM2CLIP-Llama-3-8B-Instruct-CC-Finetuned'
config = AutoConfig.from_pretrained(
    llm_model_name, trust_remote_code=True
)
llm_model = AutoModel.from_pretrained(llm_model_name, torch_dtype=torch.bfloat16, config=config, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
llm_model.config._name_or_path = '/llama3-8B-instruct' #  Workaround for LLM2VEC
l2v = LLM2Vec(llm_model, tokenizer, pooling_mode="mean", max_length=4096, doc_max_length=4096)

data_names = ['Beauty', 'Sports_and_Outdoors', 'Toys_and_Games']
data_name = data_names[0]

d=pd.read_csv(f'/dataset/{data_name}/{data_name}_text_img.csv')
d['title']=d['title'].astype(str)
item_word_embs=[torch.zeros(1280).to('cuda')]
item_pic_embs=[torch.zeros(1280).to('cuda')]
for k in range(d.shape[0]):
    image_path = d.iloc[k,2]
    response = req.get(image_path)
    image = Image.open(BytesIO(response.content))
    input_pixels = processor(images=image, return_tensors="pt").pixel_values.to('cuda')
    text_features = l2v.encode([d.iloc[k,1]], convert_to_tensor=True).to('cuda')
    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = model.get_image_features(input_pixels)
        text_features = model.get_text_features(text_features)
    item_word_embs.extend(text_features)
    item_pic_embs.extend(image_features)
a = torch.stack(tensors=item_word_embs, dim=0)
torch.save(a, f'/dataset/{data_name}/{data_name}_llm2clip_text_emb.pt')
b= torch.stack(tensors=item_pic_embs, dim=0)
torch.save(b, f'/dataset/{data_name}/{data_name}_llm2clip_pic_emb.pt')
