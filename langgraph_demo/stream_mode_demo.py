import os
import time
from typing import List, cast, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field

# 1. 初始化LLM（保持你的SiliconFlow配置）
llm = ChatOpenAI(
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    model="deepseek-ai/DeepSeek-V3.1",
    base_url="https://api.siliconflow.cn/v1/",
    temperature=0.1
)


# 2. 天气工具
@tool("weather_tool", description="查询指定城市的天气，仅需传入城市名")
def get_weather(city: str) -> str:
    time.sleep(0.5)
    return f"{city} 今日天气：晴朗，25℃，微风"


# 3. 状态结构（使用正确的消息类型处理）
class SimpleState(BaseModel):
    messages: List[HumanMessage | AIMessage] = Field(default_factory=list)  # 明确消息类型
    current_node: str = "decide"  # 初始节点


# 4. 判断节点（修正消息处理方式）
def decide_node(state: SimpleState) -> SimpleState:
    """判断是否需要调用天气工具"""
    # 构建提示（使用AIMessage作为系统提示）
    prompt = [
        AIMessage(content="""
        你只需做两件事：
        1. 如果用户问天气，回复格式：查天气+城市名（例：查天气北京）
        2. 其他问题，直接用自然语言回答
        """),
        *state.messages  # 包含用户消息
    ]

    # 调用LLM
    llm_reply = llm.invoke(prompt).content

    # 判断是否需要调用工具（直接处理字符串，不解析JSON）
    if llm_reply.startswith("查天气"):
        city = llm_reply.replace("查天气", "").strip()
        return SimpleState(
            messages=state.messages + [AIMessage(content=f"正在查{city}的天气...")],
            current_node="weather"
        )
    else:
        return SimpleState(
            messages=state.messages + [AIMessage(content=llm_reply)],
            current_node=END
        )


# 5. 天气工具节点（修正消息提取方式）
def weather_node(state: SimpleState) -> SimpleState:
    """调用天气工具并返回结果"""
    # 从消息列表中提取城市名（使用属性访问，不使用get()）
    last_msg = next(
        (msg for msg in state.messages if isinstance(msg, AIMessage) and msg.content.startswith("正在查")),
        None
    )

    if last_msg:
        city = last_msg.content.replace("正在查", "").replace("的天气...", "").strip()
        weather_result = get_weather.invoke(city)

        return SimpleState(
            messages=state.messages + [AIMessage(content=weather_result)],
            current_node=END
        )

    # 异常处理
    return SimpleState(
        messages=state.messages + [AIMessage(content="抱歉，未找到要查询的城市信息")],
        current_node=END
    )


# 6. 构建图
builder = StateGraph(SimpleState)
builder.add_node("decide", decide_node)
builder.add_node("weather", weather_node)

# 定义流程
builder.add_edge(START, "decide")
builder.add_conditional_edges(
    source="decide",
    path=lambda state: "weather" if state.current_node == "weather" else END
)
builder.add_edge("weather", END)

graph = builder.compile()


# 7. 测试函数
def test_simple_stream(user_question: str, stream_mode: List[str]):
    print(f"\n=== stream_mode={stream_mode} | 用户问：{user_question} ===")
    initial_state = {"messages": [HumanMessage(content=user_question)]}

    for _, event_data in graph.stream(initial_state, stream_mode=stream_mode):
        if isinstance(event_data, dict):
            for node_name, node_data in event_data.items():
                if "messages" in node_data:
                    # 访问消息内容时使用.content属性，不使用.get()
                    msg = node_data["messages"][-1]
                    print(msg.pretty_repr())
        else:
            message_chunk, message_metadata = cast(
                tuple[BaseMessage, dict[str, Any]], event_data
            )
            print(message_chunk.pretty_print())


# 测试
#test_simple_stream("杭州今天天气怎么样？", ["updates"])
test_simple_stream("杭州今天天气怎么样？", ["messages", "updates"])
#test_simple_stream("什么是LangGraph？", ["messages"])
