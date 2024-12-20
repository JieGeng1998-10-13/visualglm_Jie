import os
from PIL import Image
from io import BytesIO
import base64
import re
import argparse
import torch
from transformers import AutoTokenizer
from sat.model.mixins import CachedAutoregressiveMixin
from sat.quantization.kernels import quantize
import hashlib
from .visualglm import VisualGLMModel

def get_infer_setting(gpu_device=0, quant=None):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_device)
    args = argparse.Namespace(
        fp16=True,
        skip_init=True,
        device='cuda' if quant is None else 'cpu',
    )
    #model, args = VisualGLMModel.from_pretrained('visualglm-6b', args)
    #model, args = VisualGLMModel.from_pretrained('/mnt/workspace/visualglm_Jie/checkpoints/finetune-visualglm-6b-11-27-10-23', args)
    #finetuned_model = "/mnt/workspace/visualglm_Jie/checkpoints/finetune-visualglm-6b-11-29-13-08"
    finetuned_model = "/mnt/workspace/visualglm_Jie/checkpoints/finetune-visualglm-6b-12-02-14-48"
    #/mnt/workspace/visualglm_Jie/checkpoints/finetune-visualglm-6b-12-02-14-48
    from finetune_visualglm import FineTuneVisualGLMModel
    model, args = FineTuneVisualGLMModel.from_pretrained(finetuned_model, args)
    model.add_mixin('auto-regressive', CachedAutoregressiveMixin())
    assert quant in [None, 4, 8]
    if quant is not None:
        quantize(model.transformer, quant)
    model.eval()
    model = model.cuda()
    tokenizer = AutoTokenizer.from_pretrained("./chatglm", trust_remote_code=True)
    return model, tokenizer

def is_chinese(text):
    zh_pattern = re.compile(u'[\u4e00-\u9fa5]+')
    return zh_pattern.search(text)

def generate_input(input_text, input_image_prompt, history=[], input_para=None, image_is_encoded=True):
    if not image_is_encoded:
        image = input_image_prompt
    else:
        decoded_image = base64.b64decode(input_image_prompt)
        image = Image.open(BytesIO(decoded_image))

    input_data = {'input_query': input_text, 'input_image': image, 'history': history, 'gen_kwargs': input_para}
    return input_data


def process_image(image_encoded):
    decoded_image = base64.b64decode(image_encoded)
    image = Image.open(BytesIO(decoded_image))
    image_hash = hashlib.sha256(image.tobytes()).hexdigest()
    image_path = f'./examples/{image_hash}.png'
    if not os.path.isfile(image_path):
        image.save(image_path)
    return os.path.abspath(image_path)  
