import gradio as gr
import os
from lmdeploy import pipeline, TurbomindEngineConfig
from openxlab.model import download

base_path = './internlm2-chat-1_8b-4bit-awq'
os.system(f'git clone https://code.openxlab.org.cn/Raytang88/internlm2-1.8b-4bit-awq.git {base_path}')
os.system(f'cd {base_path} && git lfs pull')

backend_config = TurbomindEngineConfig(session_len=8192, model_fomat='awq') # 图片分辨率较高时请调高session_len

pipe = pipeline(base_path, backend_config=backend_config)


def model(text):
    if text is None:
        return [(text, "请输入你的问题。")]
    else:
        response = pipe((text)).text
    return [(text, response)]

# demo = gr.Interface(fn=model, inputs=[gr.Image(type="pil"), gr.Textbox()], outputs=gr.Chatbot())
demo = gr.Interface(fn=model, inputs=[gr.Textbox()], outputs=gr.Chatbot())
demo.launch()


    