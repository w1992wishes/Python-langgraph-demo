from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

import os

llm = ChatOpenAI(
    model="GLM-4.5",
    api_key=os.getenv("ZHIPU_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)

# 创建提示模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业的{domain}专家"),
    ("human", "请解释一下{topic}的概念和应用")
])

# 创建链
chain = prompt | llm

# 调用链
response = chain.invoke({
    "domain": "机器学习",
    "topic": "深度学习"
})

print(response.content)