
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("SILICONFLOW_API_KEY"),
                base_url="https://api.siliconflow.cn/v1/")
response = client.chat.completions.create(
    model='deepseek-ai/DeepSeek-V3.1',
    #model="Qwen/Qwen3-Next-80B-A3B-Instruct",
    messages=[
        {'role': 'user',
        'content': "请介绍一下人工智能的发展历程"}
    ],
    stream=True
)

for chunk in response:
    if not chunk.choices:
        continue
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
    if chunk.choices[0].delta.reasoning_content:
        print(chunk.choices[0].delta.reasoning_content, end="", flush=True)