from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


import os

llm = ChatOpenAI(
    model="GLM-4.5",
    api_key=os.getenv("ZHIPU_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)


# 发送消息（输出会实时流式显示）
response = llm.invoke([HumanMessage(content="写一首关于春天的诗")])
