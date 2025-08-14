CUDA_VISIBLE_DEVICES='0' python main_new_MM.py \
  --device cuda:0 \
  --data_path_1 /dataset/SASRec_item_embed_new.pkl \
  --data_path_2 /dataset/Beauty_llm2clip_pic_emb.pt \
  --data_path_3 /dataset/Beauty_llm2clip_text_emb.pt \
  --ckpt_dir /checkpoint/ \
  --e_dim 4096 \
  --num_emb_list 256 256 256 256 \
  --layers 4096 4096 4096 \
  --lr 2e-4 \
  --batch_size 1024 \
  --maxe 2000 \
  --loss_type mmd \
  --recon 3


