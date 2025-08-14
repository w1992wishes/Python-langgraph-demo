import os

import streamlit as st
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI  # DeepSeek兼容OpenAI的API格式
import json
from typing import List, Optional, Dict, Any

from sympy import false

# 设置页面配置
st.set_page_config(page_title="企业级中央Data Agent", page_icon="📊", layout="wide")

# 初始化DeepSeek LLM
@st.cache_resource
def init_llm():
    """初始化DeepSeek模型"""
    # 请确保已设置环境变量DEEPSEEK_API_KEY或在此处直接提供
    # DeepSeek API文档: https://platform.deepseek.com/docs/api
    return ChatOpenAI(
        model="qwen-plus",  # DeepSeek的对话模型
        temperature=0,
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

llm = init_llm()

# 定义数据模型
class UserIntent(BaseModel):
    """用户意图模型，包含数据查询所需的关键信息"""
    metrics: List[str] = Field(description="用户需要查询的指标列表")
    dimensions: List[str] = Field(description="用户需要分析的维度列表")
    filters: Dict[str, Any] = Field(description="用户需要的过滤条件，键为维度，值为过滤值")
    is_complete: bool = Field(description="意图是否完整，即是否包含所有必要的信息")
    missing_info: List[str] = Field(description="缺失的信息列表，如['metrics', 'dimensions', 'filters']")

class TaskPlan(BaseModel):
    """任务计划模型，包含需要执行的任务和工具调用信息"""
    tasks: List[str] = Field(description="需要执行的任务列表")
    tools: List[Dict[str, Any]] = Field(description="工具调用信息，每个工具包含name和parameters")

# 定义状态
class AgentState(BaseModel):
    """Agent的状态模型，包含对话历史、当前意图和任务计划"""
    messages: List[Dict[str, str]] = Field(default_factory=list)
    intent: Optional[UserIntent] = None
    task_plan: Optional[TaskPlan] = None
    step: str = Field(default="identify_intent")  # 状态机步骤：identify_intent, prompt_user, generate_plan

# 初始化记忆
memory = ConversationBufferMemory()

# 意图识别函数
def identify_intent(state: AgentState) -> AgentState:
    """
    识别用户意图，提取指标、维度和过滤条件
    判断意图是否完整，若不完整则记录缺失信息
    """
    st.subheader("意图识别中...")

    # 获取最新的用户消息
    user_message = state.messages[-1]["content"]

    # 构建意图识别提示（针对DeepSeek优化）
    prompt = ChatPromptTemplate.from_template("""
    你是一个专业的数据查询意图识别系统，请仔细分析用户的查询，提取以下关键信息：
    
    1. 指标(metrics)：用户需要查询的具体数据指标，例如销售额、利润、用户数量等
    2. 维度(dimensions)：用户需要分析的维度，例如时间、地区、产品类别、部门等
    3. 过滤条件(filters)：用户指定的过滤条件，例如"2023年"、"华东地区"、"电子产品"等
    
    请检查是否有信息缺失，如果有，请在missing_info中列出缺失项（可能是metrics、dimensions、filters）。
    如果所有信息都完整，is_complete设为True，否则为False。
    
    用户查询：{user_message}
    
    请严格按照以下JSON格式返回结果，不要添加任何额外内容：
    {{
        "metrics": [],
        "dimensions": [],
        "filters": {{}},
        "is_complete": true/false,
        "missing_info": []
    }}
    """)

    # 构建链
    chain = prompt | llm | JsonOutputParser(pydantic_object=UserIntent)

    # 执行意图识别
    try:
        intent = chain.invoke({"user_message": user_message})
        st.json(intent)

        # 关键修改：通过AgentState构造函数创建新实例，而非copy()
        new_state = AgentState(**state.dict())  # 基于原状态字典创建新实例
        new_state.intent = intent  # 更新意图

        # 决定下一步
        if intent.get('is_complete', false):
            new_state.step = "rewrite_question"
        else:
            new_state.step = "prompt_user"

        return new_state
    except Exception as e:
        st.error(f"意图识别出错: {str(e)}")
        return state

# 问题改写函数
def rewrite_question(state: AgentState) -> AgentState:
    """
    基于完整的用户意图，改写用户问题，使其更清晰、准确
    """
    st.subheader("问题改写中...")

    if not state.intent or not state.intent.is_complete:
        st.warning("意图不完整，无法进行问题改写")
        new_state = state.copy()
        new_state.step = "prompt_user"
        return new_state

    # 构建问题改写提示（针对DeepSeek优化）
    prompt = ChatPromptTemplate.from_template("""
    请根据用户的原始查询和识别出的意图，将问题改写成更清晰、准确的表述，
    确保完整包含所有的指标、维度和过滤条件，保持自然的中文表达。
    
    原始查询：{original_query}
    识别的意图：{intent}
    
    改写后的问题：
    """)

    # 获取原始查询
    original_query = state.messages[-1]["content"]

    # 构建链
    chain = prompt | llm | StrOutputParser()

    # 执行问题改写
    try:
        rewritten_question = chain.invoke({
            "original_query": original_query,
            "intent": state.intent.dict()
        })

        st.info(f"改写后的问题: {rewritten_question}")

        # 更新状态，添加改写后的问题到消息列表
        new_state = AgentState(**state.dict())
        new_state.messages.append({"role": "system", "content": f"改写后的问题: {rewritten_question}"})
        new_state.step = "generate_plan"

        return new_state
    except Exception as e:
        st.error(f"问题改写出错: {str(e)}")
        return state

# 用户提示函数
def prompt_user(state: AgentState) -> AgentState:
    """
    根据缺失的信息，生成引导语句，提示用户补充信息
    """
    st.subheader("需要用户补充信息...")

    if not state.intent or state.intent.is_complete:
        st.warning("意图已完整，无需提示用户")
        new_state = AgentState(**state.dict())
        new_state.step = "rewrite_question"
        return new_state

    # 构建提示用户的引导语句（针对DeepSeek优化）
    prompt = ChatPromptTemplate.from_template("""
    用户的查询缺少以下信息：{missing_info}
    请生成一个友好、自然的中文引导语句，询问用户补充这些信息，不要使用技术术语，保持口语化。
    例如，如果缺少metrics和dimensions，不要直接说"请补充metrics和dimensions"，
    而是说"请问您想查询哪些具体指标？需要按什么维度进行分析呢？"
    """)

    # 构建链
    chain = prompt | llm | StrOutputParser()

    # 生成引导语句
    try:
        prompt_text = chain.invoke({"missing_info": ", ".join(state.intent.missing_info)})
        st.info(f"引导用户: {prompt_text}")

        # 更新状态，添加系统提示到消息列表
        new_state = AgentState(**state.dict())
        new_state.messages.append({"role": "system", "content": prompt_text})
        new_state.step = "wait_for_user"  # 等待用户输入新信息

        return new_state
    except Exception as e:
        st.error(f"生成用户提示出错: {str(e)}")
        return state

# 生成任务计划函数
def generate_task_plan(state: AgentState) -> AgentState:
    """
    基于完整的用户意图，生成任务计划和工具调用信息
    """
    st.subheader("生成任务计划中...")

    if not state.intent or not state.intent.is_complete:
        st.warning("意图不完整，无法生成任务计划")
        new_state = AgentState(**state.dict())
        new_state.step = "prompt_user"
        return new_state

    # 构建任务计划生成提示（针对DeepSeek优化）
    prompt = ChatPromptTemplate.from_template("""
    请根据用户的意图，生成数据查询的任务计划和所需的工具调用信息。
    工具可以包括：数据库查询工具、数据分析工具、可视化工具等。
    
    用户意图：{intent}
    
    请严格按照以下JSON格式返回结果，确保每个工具调用包含name和parameters字段，
    不要添加任何额外内容：
    {{
        "tasks": ["任务1", "任务2", ...],
        "tools": [
            {{"name": "工具名称", "parameters": {{"参数1": "值1", "参数2": "值2", ...}}}},
            ...
        ]
    }}
    """)

    # 构建链
    chain = prompt | llm | JsonOutputParser(pydantic_object=TaskPlan)

    # 生成任务计划
    try:
        task_plan = chain.invoke({"intent": state.intent.dict()})
        st.json(task_plan)

        # 更新状态
        new_state = AgentState(**state.dict())
        new_state.task_plan = task_plan
        new_state.messages.append({
            "role": "system",
            "content": f"任务计划已生成: {json.dumps(task_plan, ensure_ascii=False)}"
        })
        new_state.step = "complete"  # 完成处理

        return new_state
    except Exception as e:
        st.error(f"生成任务计划出错: {str(e)}")
        return state

# 定义状态机转换
def get_next_step(state: AgentState) -> str:
    """根据当前状态决定下一步"""
    if state.step == "identify_intent":
        return "identify_intent"
    elif state.step == "prompt_user":
        return "prompt_user"
    elif state.step == "rewrite_question":
        return "rewrite_question"
    elif state.step == "generate_plan":
        return "generate_plan"
    elif state.step == "complete":
        return END
    else:  # wait_for_user，等待用户输入后重新识别意图
        return "identify_intent"

# 构建状态机
workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("identify_intent", identify_intent)
workflow.add_node("prompt_user", prompt_user)
workflow.add_node("rewrite_question", rewrite_question)
workflow.add_node("generate_plan", generate_task_plan)

# 设置边缘
workflow.set_conditional_entry_point(
    get_next_step,
    {
        "identify_intent": "identify_intent",
        "prompt_user": "prompt_user",
        "rewrite_question": "rewrite_question",
        "generate_plan": "generate_plan",
        END: END
    }
)

# 为每个节点设置下一步
workflow.add_conditional_edges(
    "identify_intent",
    lambda s: s.step,
    {
        "prompt_user": "prompt_user",
        "rewrite_question": "rewrite_question"
    }
)

workflow.add_conditional_edges(
    "prompt_user",
    lambda s: s.step,
    {
        "wait_for_user": END,  # 等待用户输入，所以结束当前流程
        "rewrite_question": "rewrite_question"
    }
)

workflow.add_conditional_edges(
    "rewrite_question",
    lambda s: s.step,
    {
        "generate_plan": "generate_plan"
    }
)

workflow.add_conditional_edges(
    "generate_plan",
    lambda s: s.step,
    {
        "complete": END
    }
)

# 编译工作流
app = workflow.compile()

# Streamlit UI
def main():
    st.title("企业级中央Data Agent (DeepSeek版)")
    st.write("这是一个支持多轮会话的Data Agent，可以识别您的数据查询意图并生成相应的任务计划。")

    # 初始化会话状态
    if "state" not in st.session_state:
        st.session_state.state = AgentState()
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 显示对话历史
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 用户输入
    if prompt := st.chat_input("请输入您的数据查询需求..."):
        # 添加用户消息到历史
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 更新Agent状态
        st.session_state.state.messages.append({"role": "user", "content": prompt})

        # 运行工作流
        with st.expander("查看处理过程", expanded=True):
            result = app.invoke(st.session_state.state)

            print(result)

            new_state = AgentState(**result)
            st.session_state.state = new_state

        # 显示系统回复
        if st.session_state.state.messages and st.session_state.state.messages[-1]["role"] == "system":
            system_message = st.session_state.state.messages[-1]
            st.session_state.messages.append(system_message)
            with st.chat_message("assistant"):
                st.markdown(system_message["content"])

        # 如果任务计划已生成，显示最终结果
        if st.session_state.state.task_plan:
            st.success("任务计划已生成！")
            with st.expander("查看详细任务计划"):
                st.json(st.session_state.state.task_plan.dict())

if __name__ == "__main__":
    main()