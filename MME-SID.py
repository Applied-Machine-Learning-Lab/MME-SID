import os

epoch_list=[4]
lr=3e-4
temp_list=[1e-3]
idx=0

#train
for epoch in epoch_list:
    for temp in temp_list:
        output_dir = './LLM4Rec-Beauty-instruct/epoch'+str(epoch)
        run_py = "CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' \
                 torchrun --nproc_per_node=8 --master_port=1234 finetune.py \
                 --base_model /llama3-8B-instruct \
                 --data_path Beauty \
                 --task_type sequential \
                 --output_dir {} \
                 --batch_size 16 \
                 --micro_batch_size 1 \
                 --num_epochs {} \
                 --learning_rate {} \
                 --cutoff_len 4096 \
                 --val_set_size 0 \
                 --lora_r 8 \
                 --lora_alpha 16 \
                 --lora_dropout 0.05 \
                 --lora_target_modules '[gate_proj, down_proj, up_proj]' \
                 --train_on_inputs False \
                 --add_eos_token False \
                 --group_by_length False \
                 --prompt_template_name llama3 \
                 --lr_scheduler 'cosine' \
                 --temp {} \
                 --idx {} \
                 --warmup_steps 100".format(output_dir, epoch, lr,temp, idx)

        os.system(run_py)

#test
for epoch in epoch_list:
    for temp in temp_list:
        output_dir = './LLM4Rec-Beauty-instruct/epoch'+str(epoch)
        if epoch==4:
            checkpoint_dir = output_dir+'/checkpoint-37512/'
        elif epoch==3:
            checkpoint_dir = output_dir+'/checkpoint-28134/'
        run_py = "torchrun --nproc_per_node=8 --master_port=12345 inference.py \
                    --base_model /llama3-8B-instruct \
                    --data_path Beauty \
                    --task_type sequential \
                    --checkpoint_dir {} \
                    --cache_dir cache_dir/ \
                    --output_dir {} \
                    --batch_size 16 \
                    --micro_batch_size 1 \
                    --lora_r 8 \
                    --lora_alpha 16 \
                    --temp {} \
                    --idx {} \
                    --prompt_template_name llama3".format(checkpoint_dir, output_dir,temp,idx)
        os.system(run_py)