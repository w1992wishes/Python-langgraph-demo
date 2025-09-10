from typing import Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage,
    FunctionMessage,
    SystemMessage,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch

from parsers import LLMCompilerPlanParser, Task
from tools import get_chat_openai


def create_planner(
    llm: BaseChatModel,
    tools: Sequence[BaseChatModel],
    base_prompt: ChatPromptTemplate
) -> RunnableBranch:
    """创建任务规划器（保留原有逻辑）"""
    # 生成工具描述文本
    tool_descriptions = "\n".join(
        f"{i+1}. {tool.description}\n"
        for i, tool in enumerate(tools)  # 索引从1开始
    )

    # 初始规划提示词（无历史规划）
    planner_prompt = base_prompt.partial(
        replan="",
        num_tools=len(tools) + 1,  # +1 是因为最后要加join工具
        tool_descriptions=tool_descriptions,
    )

    # 重新规划提示词（有历史规划和执行结果）
    replanner_prompt = base_prompt.partial(
        replan=' - You are given "Previous Plan" which is the plan that the previous agent created along with the execution results '
        "(given as Observation) of each plan and a general thought (given as Thought) about the executed results."
        'You MUST use these information to create the next plan under "Current Plan".\n'
        ' - When starting the Current Plan, you should start with "Thought" that outlines the strategy for the next plan.\n'
        " - In the Current Plan, you should NEVER repeat the actions that are already executed in the Previous Plan.\n"
        " - You must continue the task index from the end of the previous one. Do not repeat task indices.",
        num_tools=len(tools) + 1,
        tool_descriptions=tool_descriptions,
    )

    def should_replan(state: list[BaseMessage]) -> bool:
        """判断是否需要重新规划（根据最后一条消息是否为SystemMessage）"""
        return isinstance(state[-1], SystemMessage)

    def wrap_messages(state: list[BaseMessage]) -> dict:
        """包装消息为规划器输入格式"""
        return {"messages": state}

    def wrap_and_get_last_index(state: list[BaseMessage]) -> dict:
        """获取最后一个任务索引，用于继续规划"""
        next_task_idx = 0
        # 倒序查找最后一个FunctionMessage的索引
        for msg in state[::-1]:
            if isinstance(msg, FunctionMessage):
                next_task_idx = msg.additional_kwargs["idx"] + 1
                break
        # 更新最后一条消息内容，添加索引起始信息
        state[-1].content = state[-1].content + f" - Begin counting at : {next_task_idx}"
        return {"messages": state}

    # 构建规划器分支（重新规划/初始规划）
    return (
        RunnableBranch(
            (should_replan, wrap_and_get_last_index | replanner_prompt),
            wrap_messages | planner_prompt,
        )
        | llm
        | LLMCompilerPlanParser(tools=tools)
    )


def get_planner(tools: Sequence[BaseChatModel], base_prompt: ChatPromptTemplate) -> RunnableBranch:
    """快捷获取规划器（使用统一LLM配置）"""
    llm = get_chat_openai()
    return create_planner(llm, tools, base_prompt)