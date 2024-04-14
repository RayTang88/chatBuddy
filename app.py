# import gradio as gr
# import os
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
# from openxlab.model import download

# base_path = './internlm2-1.8b-4bit-awq'
# os.system(f'git clone https://code.openxlab.org.cn/Raytang88/internlm2-1.8b-4bit-awq.git {base_path}')
# os.system(f'cd {base_path} && git lfs pull')

# tokenizer = AutoTokenizer.from_pretrained(base_path,trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(base_path,trust_remote_code=True, torch_dtype=torch.float16).cuda()

# def chat(message,history):
#     for response,history in model.stream_chat(tokenizer,message,history,max_length=2048,top_p=0.7,temperature=1):
#         yield response

# gr.ChatInterface(chat,
#                  title="internlm2-1.8b-4bit-awq",
#                 description="""
# InternLM is mainly developed by Shanghai AI Laboratory.  
#                  """,
#                  ).queue(1).launch()

import gradio as gr
from lmdeploy import pipeline, TurbomindEngineConfig


backend_config = TurbomindEngineConfig(session_len=8192) # 图片分辨率较高时请调高session_len
# pipe = pipeline('liuhaotian/llava-v1.6-vicuna-7b', backend_config=backend_config) 非开发机运行此命令
pipe = pipeline('/share/new_models/liuhaotian/llava-v1.6-vicuna-7b', backend_config=backend_config)

def model(message, history):
    if history is None:
        return [(message, "请说出你的问题。")]
    else:
        response = pipe((message, history)).text
        return [(message, response)]

demo = gr.Interface(fn=model, inputs=[gr.Textbox()], outputs=gr.Chatbot())
demo.launch()  