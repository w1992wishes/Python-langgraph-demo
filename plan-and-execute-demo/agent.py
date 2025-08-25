import os
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_tavily import TavilySearch
from settings import Settings

# 使用非常简单的提示词
prompt = """You are a helpful assistant that can use tools to answer questions.
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original question

Begin!"""

# 初始化工具和LLM
tools = [TavilySearch(max_results=3)]
llm = ChatOpenAI(
    model=Settings.LLM_MODEL,
    temperature=Settings.TEMPERATURE,
    api_key=Settings.OPENAI_API_KEY,
    base_url=Settings.OPENAI_BASE_URL,
)

# 创建agent执行器
agent_executor = create_react_agent(model=llm, tools=tools, prompt=prompt)

# 测试调用
try:
    # 使用更简短、更直接的问题
    response = agent_executor.invoke({
        "messages": [("user", "2024 US Open men's winner")]
    })
    print("Agent response:", response)
except Exception as e:
    print(f"Agent error: {str(e)}")
