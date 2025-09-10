from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import add_messages
from typing import Annotated, Dict, List
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, SystemMessage, AIMessage

from planner import get_planner
from scheduler import plan_and_schedule
from joiner import get_joiner
from prompts import get_llm_compiler_prompt
from tools import get_math_tool, get_tavily_search, get_chat_openai

class State(TypedDict):
    """状态图状态数据结构（保留原有逻辑）"""
    messages: Annotated[List[BaseMessage], add_messages]


def build_state_graph() -> StateGraph:
    """构建Agent状态图（保留原有逻辑）"""
    # 1. 初始化工具和规划器
    llm = get_chat_openai()
    math_tool = get_math_tool(llm)
    search_tool = get_tavily_search()
    tools = [search_tool, math_tool]

    # 2. 初始化核心组件
    base_prompt = get_llm_compiler_prompt()
    planner = get_planner(tools, base_prompt)
    joiner = get_joiner()

    # 3. 定义状态图
    graph_builder = StateGraph(State)

    # -------------------------- 关键修改：修复plan_and_schedule节点 --------------------------
    def plan_and_schedule_node(state: State) -> State:
        """规划+调度节点包装器（补充异常捕获和输出验证）"""
        try:
            # 执行原有规划调度逻辑
            result = plan_and_schedule.invoke(
                {"messages": state["messages"]},
                planner=planner  # 确保planner正确传入（原逻辑可能遗漏该参数！）
            )
            # 验证结果是否为标准State结构，无则补默认值
            if not isinstance(result, dict) or "messages" not in result:
                return {
                    "messages": state["messages"] + [
                        SystemMessage(content=f"plan_and_schedule节点执行异常：返回结果格式错误，原始结果：{str(result)}")
                    ]
                }
            return result
        except Exception as e:
            # 捕获所有异常，避免节点执行中断，确保输出可观测
            import traceback
            error_msg = f"plan_and_schedule节点执行失败：{repr(e)}\n{traceback.format_exc()}"
            return {
                "messages": state["messages"] + [SystemMessage(content=error_msg)]
            }
    # ----------------------------------------------------------------------------------------

    graph_builder.add_node("plan_and_schedule", plan_and_schedule_node)
    graph_builder.add_node("join", joiner)

    # 5. 添加边（规划调度 -> 结果合并）
    graph_builder.add_edge("plan_and_schedule", "join")

    # 6. 条件边（判断是否需要重新规划）
    def should_continue(state: State) -> str | None:
        """判断是否继续循环（保留原有逻辑）"""
        messages = state["messages"]
        if isinstance(messages[-1], AIMessage):
            return END  # 已生成最终回答，结束流程
        return "plan_and_schedule"  # 需要重新规划，回到规划节点

    graph_builder.add_conditional_edges(
        "join",
        should_continue,
    )

    # 7. 起始边（START -> 规划调度）
    graph_builder.add_edge(START, "plan_and_schedule")

    return graph_builder


def get_agent_chain() -> StateGraph:
    """获取编译后的Agent状态链"""
    graph_builder = build_state_graph()
    return graph_builder.compile()