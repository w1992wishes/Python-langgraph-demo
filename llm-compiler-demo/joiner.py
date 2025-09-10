from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field
from typing import List, Union, Dict

from prompts import get_llm_compiler_joiner_prompt
from tools import get_chat_openai


class FinalResponse(BaseModel):
    """最终回答数据结构（保留原有逻辑）"""
    response: str


class Replan(BaseModel):
    """重新规划数据结构（保留原有逻辑）"""
    feedback: str = Field(
        description="Analysis of the previous attempts and recommendations on what needs to be fixed."
    )


class JoinOutputs(BaseModel):
    """合并输出数据结构（保留原有逻辑）"""
    thought: str = Field(
        description="The chain of thought reasoning for the selected action"
    )
    action: Union[FinalResponse, Replan]


def _parse_joiner_output(decision: JoinOutputs) -> Dict[str, List[BaseMessage]]:
    """解析合并器输出，生成下一步消息（保留原有逻辑）"""
    response = [AIMessage(content=f"Thought: {decision.thought}")]
    if isinstance(decision.action, Replan):
        # 需要重新规划：返回SystemMessage携带上下文
        return {
            "messages": response
                        + [
                            SystemMessage(
                                content=f"Context from last attempt: {decision.action.feedback}"
                            )
                        ]
        }
    else:
        # 无需重新规划：返回最终回答
        return {"messages": response + [AIMessage(content=decision.action.response)]}


def select_recent_messages(state: Dict[str, List[BaseMessage]]) -> Dict[str, List[BaseMessage]]:
    """筛选最近的消息（从最后一条HumanMessage开始向前）"""
    messages = state["messages"]
    selected = []
    for msg in messages[::-1]:
        selected.append(msg)
        if isinstance(msg, AIMessage):
            break
    return {"messages": selected[::-1]}


def get_joiner() -> Runnable:
    """获取结果合并器（保留原有逻辑）"""
    # 初始化提示词和LLM
    joiner_prompt = get_llm_compiler_joiner_prompt()
    llm = get_chat_openai()

    # 构建结构化输出链
    runnable = joiner_prompt | llm.with_structured_output(
        JoinOutputs, method="function_calling"
    )

    # 组合：筛选消息 -> 合并判断 -> 解析输出
    return select_recent_messages | runnable | _parse_joiner_output
