import torch
import transformers

from LLaVA.llava.conversation import conv_templates
from LLaVA.llava.model import *


def remove_alter(s):  # hack expressive instruction
    if 'ASSISTANT:' in s: s = s[s.index('ASSISTANT:')+10:].strip()
    if '</s>' in s: s = s[:s.index('</s>')].strip()
    if 'alternative' in s.lower(): s = s[:s.lower().index('alternative')]
    if '[IMG0]' in s: s = s[:s.index('[IMG0]')]
    s = '.'.join([s.strip() for s in s.split('.')[:2]])
    if s[-1]!='.': s += '.'
    return s.strip()

DEFAULT_IMAGE_TOKEN = '<image>'
DEFAULT_IMAGE_PATCH_TOKEN = '<im_patch>'
DEFAULT_IM_START_TOKEN = '<im_start>'
DEFAULT_IM_END_TOKEN = '<im_end>'

class MGIE_module():
    def __init__(self, ckpt_dir):
        PATH_LLAVA = f'{ckpt_dir}/LLaVA-7B-v1'

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(PATH_LLAVA)
        self.model = LlavaLlamaForCausalLM.from_pretrained(PATH_LLAVA, low_cpu_mem_usage=True, torch_dtype=torch.float16, use_cache=True).cuda()
        self.image_processor = transformers.CLIPImageProcessor.from_pretrained(self.model.config.mm_vision_tower, torch_dtype=torch.float16)

        self.tokenizer.padding_side = 'left'
        self.tokenizer.add_tokens(['[IMG0]', '[IMG1]', '[IMG2]', '[IMG3]', '[IMG4]', '[IMG5]', '[IMG6]', '[IMG7]'], special_tokens=True)
        self.model.resize_token_embeddings(len(self.tokenizer))
        ckpt = torch.load(f'{ckpt_dir}/mgie_7b/mllm.pt', map_location='cpu')
        self.model.load_state_dict(ckpt, strict=False)

        mm_use_im_start_end = getattr(self.model.config, 'mm_use_im_start_end', False)
        self.tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end: self.tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

        vision_tower = self.model.get_model().vision_tower[0]
        vision_tower = transformers.CLIPVisionModel.from_pretrained(vision_tower.config._name_or_path, torch_dtype=torch.float16, low_cpu_mem_usage=True).cuda()
        self.model.get_model().vision_tower[0] = vision_tower
        vision_config = vision_tower.config
        vision_config.im_patch_token = self.tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
        vision_config.use_im_start_end = mm_use_im_start_end
        if mm_use_im_start_end: vision_config.im_start_token, vision_config.im_end_token = self.tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])
        self.image_token_len = (vision_config.image_size//vision_config.patch_size)**2

        _ = self.model.eval()
        self.EMB = ckpt['emb'].cuda()
        with torch.inference_mode(): 
            self.NULL = self.model.edit_head(torch.zeros(1, 8, 4096).half().to('cuda'), self.EMB)
        print('NULL:', self.NULL.shape)
    
    def generate_prompt(self, input_image, instruction):
        img = self.image_processor.preprocess(input_image, return_tensors='pt')['pixel_values'][0]
        inst_txt = instruction
        txt = "what will this image be like if '%s'"%(inst_txt)
        txt = txt+'\n'+DEFAULT_IM_START_TOKEN+DEFAULT_IMAGE_PATCH_TOKEN*self.image_token_len+DEFAULT_IM_END_TOKEN
        conv = conv_templates['vicuna_v1_1'].copy()
        conv.append_message(conv.roles[0], txt), conv.append_message(conv.roles[1], None)
        txt = conv.get_prompt()
        txt = self.tokenizer(txt)
        txt, mask = torch.as_tensor(txt['input_ids']), torch.as_tensor(txt['attention_mask'])
        
        with torch.inference_mode():
            out = self.model.generate(txt.unsqueeze(dim=0).cuda(), images=img.half().unsqueeze(dim=0).cuda(), attention_mask=mask.unsqueeze(dim=0).cuda(), 
                                do_sample=False, max_new_tokens=96, num_beams=1, no_repeat_ngram_size=3, 
                                return_dict_in_generate=True, output_hidden_states=True)
            out, hid = out['sequences'][0].tolist(), torch.cat([x[-1] for x in out['hidden_states']], dim=1)[0]
            
            p = min(out.index(32003)-1 if 32003 in out else len(hid)-9, len(hid)-9)
            hid = hid[p:p+8]

            out = remove_alter(self.tokenizer.decode(out))
            emb = self.model.edit_head(hid.unsqueeze(dim=0), self.EMB)
            
        prompt_embeds = emb
        negative_prompt_embeds = self.NULL
        return prompt_embeds, negative_prompt_embeds