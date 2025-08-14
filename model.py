import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import LlamaModel, LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, BitsAndBytesConfig
from transformers.modeling_outputs import SequenceClassifierOutputWithPast

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict
)


class LLM4Rec(nn.Module):
    def __init__(self, **args):
        super(LLM4Rec, self).__init__()
        self.args = args
        self.input_dim, self.output_dim = args['input_dim'], args['output_dim']

        print(f'Initializing language decoder ...')
        # add the lora module
        peft_config = LoraConfig(
            task_type='FEATURE_EXTRACTION',
            r=self.args['lora_r'],
            lora_alpha=self.args['lora_alpha'],
            lora_dropout=self.args['lora_dropout'],
            target_modules=self.args['lora_target_modules'],
            bias='none',
        )
        print(self.args['device_map'])
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        self.llama_model = LlamaModel.from_pretrained(self.args['base_model'], quantization_config=bnb_config, torch_dtype=torch.bfloat16,
                                                      local_files_only=True, cache_dir=args['cache_dir'],
                                                      device_map=self.args['device_map'])

        self.llama_model = prepare_model_for_kbit_training(self.llama_model)
        self.llama_model = get_peft_model(self.llama_model, peft_config)
        self.llama_model.print_trainable_parameters()
        self.llama_model.config.use_cache = False

        self.llama_tokenizer = AutoTokenizer.from_pretrained(self.args['base_model'], use_fast=False, local_files_only=True, cache_dir=args['cache_dir'])
        print("Pad Token:", self.llama_tokenizer.pad_token)
        self.llama_tokenizer.padding_side = "right"
        self.instruct_ids, self.instruct_mask = self.llama_tokenizer(self.args['instruction_text'][0],
                                                                     truncation=True, padding=False,
                                                                     return_tensors='pt', add_special_tokens=False).values()
        self.response_ids, self.response_mask = self.llama_tokenizer(self.args['instruction_text'][1],
                                                                     truncation=True, padding=False,
                                                                     return_tensors='pt', add_special_tokens=False).values()
        print('Language decoder initialized.')
        self.task_type = args['task_type']
        self.input_embeds = nn.Embedding.from_pretrained(self.args['input_embeds'], freeze=True)
        self.llm_embs = nn.Embedding.from_pretrained(self.args['text_embeds'], freeze=True)
        self.pic_embs = nn.Embedding.from_pretrained(self.args['pic_embeds'], freeze=True)
        self.input_proj = nn.Linear(self.input_dim, self.llama_model.config.hidden_size)
        self.input_proj2 = nn.Linear(1280, self.llama_model.config.hidden_size) #LLM2CLIP output dimension=1280
        self.input_proj3 = nn.Linear(1280, self.llama_model.config.hidden_size)
        self.score = nn.Linear(self.llama_model.config.hidden_size, self.output_dim, bias=False)
        self.mlp = nn.Sequential(nn.Linear(self.llama_model.config.hidden_size*6, self.llama_model.config.hidden_size*3),nn.ReLU(),nn.Linear(self.llama_model.config.hidden_size*3, self.llama_model.config.hidden_size))
        self.mlp_head = torch.nn.Sequential(torch.nn.Linear(1,64),torch.nn.GELU(), torch.nn.Linear(64,10))
        self.mlp_meta_embedding = torch.nn.Sequential(nn.Linear(10,64,bias=False),torch.nn.GELU(),torch.nn.Linear(64,4)) #,torch.nn.Softmax(dim=1))
        self.temp = self.args['temp']
        self.act = torch.nn.GELU()
        self.fre = torch.load('/dataset/Beauty_fre.pt')
        self.fre = self.fre.unsqueeze(-1)
        self.fre = torch.log(self.fre+1)
        self.fre = (self.fre - min(self.fre))/(max(self.fre)-min(self.fre))
        idx=self.args['idx']
        self.sid = torch.load('/dataset/mm_ID_SID_4096_num256_'+str(idx)+'.pt')
        self.sid = torch.tensor(self.sid)
        self.codebook = torch.load('/dataset/mm_ID_SID_code_4096_num256_'+str(idx)+'.pt')
        self.e1 = nn.Embedding.from_pretrained(self.codebook[0], freeze=False)
        self.e2 = nn.Embedding.from_pretrained(self.codebook[1], freeze=False)
        self.e3 = nn.Embedding.from_pretrained(self.codebook[2], freeze=False)
        self.e4 = nn.Embedding.from_pretrained(self.codebook[3], freeze=False)

        self.sid2 = torch.load('/dataset/mm_pic_SID_4096_num256_'+str(idx)+'.pt')
        self.sid2 = torch.tensor(self.sid2)
        self.codebook2 = torch.load('/dataset/mm_pic_SID_code_4096_num256_'+str(idx)+'.pt')
        self.e12 = nn.Embedding.from_pretrained(self.codebook2[0], freeze=False)
        self.e22 = nn.Embedding.from_pretrained(self.codebook2[1], freeze=False)
        self.e32 = nn.Embedding.from_pretrained(self.codebook2[2], freeze=False)
        self.e42 = nn.Embedding.from_pretrained(self.codebook2[3], freeze=False)

        self.sid3 = torch.load('/dataset/mm_text_SID_4096_num256_'+str(idx)+'.pt')
        self.sid3 = torch.tensor(self.sid3)
        self.codebook3 = torch.load('/dataset/mm_text_SID_code_4096_num256_'+str(idx)+'.pt')
        self.e13 = nn.Embedding.from_pretrained(self.codebook3[0], freeze=False)
        self.e23 = nn.Embedding.from_pretrained(self.codebook3[1], freeze=False)
        self.e33 = nn.Embedding.from_pretrained(self.codebook3[2], freeze=False)
        self.e43 = nn.Embedding.from_pretrained(self.codebook3[3], freeze=False)

    def predict(self, inputs, inputs_mask):
        bs = inputs.shape[0]
        instruct_embeds = self.llama_model.model.embed_tokens(self.instruct_ids.cuda()).expand(bs, -1, -1)
        response_embeds = self.llama_model.model.embed_tokens(self.response_ids.cuda()).expand(bs, -1, -1)
        instruct_mask = self.instruct_mask.cuda().expand(bs, -1)
        response_mask = self.response_mask.cuda().expand(bs, -1)

        if self.task_type == 'sequential':
            inputs_1 = self.sid.to(inputs.device)[inputs]
            inputs_1 = self.e1(inputs_1[:,:,0])+self.e2(inputs_1[:,:,1])+self.e3(inputs_1[:,:,2])+self.e4(inputs_1[:,:,3])
            inputs_2 = self.sid2.to(inputs.device)[inputs]
            inputs_2 = self.e12(inputs_2[:,:,0])+self.e22(inputs_2[:,:,1])+self.e32(inputs_2[:,:,2])+self.e42(inputs_2[:,:,3])
            inputs_3 = self.sid3.to(inputs.device)[inputs]
            inputs_3 = self.e13(inputs_3[:,:,0])+self.e23(inputs_3[:,:,1])+self.e33(inputs_3[:,:,2])+self.e43(inputs_3[:,:,3])

            token = (self.input_proj2(self.llm_embs(inputs)))
            pic = (self.input_proj3(self.pic_embs(inputs)))
            ind = (self.input_proj(self.input_embeds(inputs)))
        inputs = self.act(self.mlp(torch.cat([ind,inputs_1,pic,inputs_2,token,inputs_3],dim=2)))
        inputs = torch.cat([instruct_embeds, inputs, response_embeds], dim=1)
        attention_mask = torch.cat([instruct_mask, inputs_mask, response_mask], dim=1)
        assert attention_mask.size()[0] == inputs.size()[0] and attention_mask.size()[1] == inputs.size()[1]

        outputs = self.llama_model(inputs_embeds=inputs, attention_mask=attention_mask, return_dict=True)
        pooled_output = outputs.last_hidden_state[:, -1]
        fre=self.mlp_meta_embedding(self.mlp_head(self.fre.to(inputs.device)))
        fre=torch.nn.Softmax(dim=1)(fre/self.temp)
        pooled_logits = fre[:,0]*self.score(pooled_output)+fre[:,1]*torch.matmul(pooled_output,self.input_proj2(self.llm_embs.weight).t())+fre[:,2]*torch.matmul(pooled_output,self.input_proj3(self.pic_embs.weight).t())+fre[:,3]*torch.matmul(pooled_output,self.input_proj(self.input_embeds.weight).t())
        return outputs, pooled_logits.view(-1, self.output_dim)

    def forward(self, inputs, inputs_mask, labels):
        outputs, pooled_logits = self.predict(inputs, inputs_mask)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(pooled_logits, labels.view(-1))

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
