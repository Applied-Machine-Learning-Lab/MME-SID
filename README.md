# CIKM25-MME-SID
Code for paper titled '*Empowering Large Language Model for Sequential Recommendation via Multimodal Embeddings and Semantic IDs*' submitted to CIKM 2025 Full Research track. Tesla V100 or A100 GPUs are preferred. In the following, we take Amazon Beauty dataset as an example.

**Preparation**

First, download and implement [Llama3-8B-instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) and [LLM2CLIP](https://huggingface.co/microsoft/LLM2CLIP-Llama-3-8B-Instruct-CC-Finetuned). We use the environment with python 3.9.19 + torch 2.0.1, and install [thuml/Transfer-Learning-Library](https://github.com/thuml/Transfer-Learning-Library) and [LLM2Vec](https://github.com/McGill-NLP/llm2vec/tree/main) by running the following code.
```
cd Transfer-Learning-Library-master
pip install -r requirements.txt
python setup.py install
cd llm2vec
pip install -e .
```

**Stage 1: Encoding**

We save the pre-trained collaborative, textual, and visual embedding into SASRec_item_embed_new.pkl, Beauty_llm2clip_text_emb.pt, and Beauty_llm2clip_pic_emb.pt under dataset folder. Then the MM-RQ-VAE is trained to generate semantic IDs and codebook embeddings.

```
sh ./MM-RQ-VAE/train_tokenizer_MM.sh
sh ./MM-RQ-VAE/tokenize_MM.sh
```

**Stage 2: Fine-tuning**

We fine-tune the Llama3-8B-instruct to conduct sequential recommendation task by running the following code. After training, the evaluation result on the test set will be shown. 
```
python MME-SID.py
```
It takes about 110 mins to train an epoch on 8 A100 GPUs and 160 mins on 8 V100 GPUs.
