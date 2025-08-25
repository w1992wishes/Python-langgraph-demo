from agent import agent_executor  # 确保你的 agent_executor 定义正确


import asyncio
from typing import Literal, List, Tuple, Annotated
from typing_extensions import TypedDict
from plan import replanner, Response, planner, Step  # 导入 Step 模型
from langgraph.graph import StateGraph, START


# 1. 确保 PlanExecute 状态结构正确（plan 是 List[Step]，与 planner 返回一致）
class PlanExecute(TypedDict):
    input: str  # 用户原始输入
    plan: List[Step]  # 计划（Step 列表，而非字符串列表）
    past_steps: Annotated[List[Tuple], lambda x, y: x + y]  # 已完成步骤（任务: 结果）
    response: str | None  # 最终回复（初始为 None）


# 2. 修复 execute_step：处理 Step 模型（获取 description 作为任务）
async def execute_step(state: PlanExecute):
    plan = state["plan"]
    if not plan:  # 防御：若计划为空，直接返回
        return {"past_steps": [], "response": "No steps left to execute."}

    # 提取第一个步骤的描述（plan 是 Step 列表，需通过 .description 获取任务内容）
    current_step = plan[0]
    plan_str = "\n".join(f"{step.step}. {step.description}" for step in plan)

    # 构造 agent 任务（明确要执行的步骤）
    task_formatted = f"""
    你的任务是执行以下计划的第 {current_step.step} 步：
    完整计划：
    {plan_str}

    当前需执行的步骤：{current_step.description}
    请执行该步骤（例如：调用工具查询信息），并返回执行结果（无需格式，直接文字描述）。
    """

    # 调用 agent 执行步骤（确保 agent_executor 接受 {"messages": [...]}）
    agent_response = await agent_executor.ainvoke(
        {"messages": [("user", task_formatted)]}
    )

    # 返回已完成步骤（任务是步骤描述，结果是 agent 响应）
    return {
        "past_steps": [(current_step.description, agent_response["messages"][-1].content)],
    }


# 3. 修复 plan_step：传递正确的 {"messages": [...]} 给 planner
async def plan_step(state: PlanExecute):
    # 从 state 中获取用户输入，构造 messages（与 planner 模板的 {messages} 匹配）
    user_input = state["input"]
    planner_response = await planner.ainvoke({
        "messages": [("user", user_input)]  # 传递 messages 变量，与模板匹配
    })

    # 返回 plan（planner 返回的是 Plan 模型，取 .steps 赋值给 state["plan"]）
    return {"plan": planner_response.steps}


# 4. 修复 replan_step：处理 plan 的结构（将 Step 列表转为字符串，便于 LLM 理解）
async def replan_step(state: PlanExecute):
    # 处理 plan：将 Step 列表转为 "step编号: 描述" 字符串（便于 LLM 解析）
    plan = state["plan"]
    plan_str = "\n".join(f"{step.step}. {step.description}" for step in plan)

    # 处理 past_steps：将列表转为字符串（便于 LLM 理解）
    past_steps = state["past_steps"]
    past_steps_str = "\n".join(f"- {task}: {result[:50]}..." for task, result in past_steps)  # 截断长结果

    # 调用 replanner：传递与模板匹配的变量（input/plan/past_steps）
    replan_response = await replanner.ainvoke({
        "input": state["input"],
        "plan": plan_str,
        "past_steps": past_steps_str if past_steps else "No steps completed yet."
    })

    # 根据 replan 结果更新 state
    if isinstance(replan_response.action, Response):
        # 若返回直接回复，更新 response 字段（触发工作流结束）
        return {"response": replan_response.action.response, "plan": []}
    else:
        # 若返回新计划，更新 plan 字段（保留剩余步骤）
        return {"plan": replan_response.action.steps}


# 5. 修复条件判断节点：检查 response 是否存在且非空
def should_end(state: PlanExecute) -> Literal["agent", "__end__"]:
    # 若 response 存在且非空，结束工作流；否则继续执行步骤
    if state.get("response") and state["response"].strip():
        return "__end__"
    # 若 plan 为空，也结束工作流（避免死循环）
    elif not state.get("plan"):
        return "__end__"
    else:
        return "agent"


# 6. 构建并编译工作流（保持不变）
workflow = StateGraph(PlanExecute)

# 添加节点
workflow.add_node("planner", plan_step)  # 生成初始计划
workflow.add_node("agent", execute_step)  # 执行步骤
workflow.add_node("replan", replan_step)  # 重新规划

# 添加边：START → planner → agent → replan → 条件判断
workflow.add_edge(START, "planner")
workflow.add_edge("planner", "agent")
workflow.add_edge("agent", "replan")

# 条件边：replan 后判断是否结束
workflow.add_conditional_edges(
    "replan",
    should_end,  # 调用判断函数
    {"agent": "agent", "__end__": "__end__"}  # 映射：判断结果 → 下一个节点
)

# 编译工作流
app = workflow.compile()

# 可视化状态图（可选）
try:
    img_path = "graph.png"
    graph_image = app.get_graph(xray=True).draw_mermaid_png()
    with open(img_path, "wb") as img_file:
        img_file.write(graph_image)
    from IPython.display import Image, display

    display(Image(img_path))
except Exception as e:
    print(f"可视化图表失败：{e}")

# 配置和用户输入
config = {"recursion_limit": 50}  # 防止递归过深
inputs = {
    "input": "what is the hometown of the mens 2024 Australia open winner?",
    "plan": [],  # 初始计划为空
    "past_steps": [],  # 初始无已完成步骤
    "response": None  # 初始无回复
}


# 处理事件流（输出工作流执行过程）
async def main():
    async for event in app.astream(inputs, config=config):
        for node_name, node_output in event.items():
            if node_name == "__end__":
                print("\n=== 工作流结束 ===")
                print(f"最终回复：{node_output.get('response', '无回复')}")
            else:
                print(f"\n=== 节点 {node_name} 执行结果 ===")
                # 打印关键信息（避免输出过长）
                if "plan" in node_output:
                    print("更新后的计划：")
                    for step in node_output["plan"]:
                        print(f"  {step.step}. {step.description}")
                if "past_steps" in node_output and node_output["past_steps"]:
                    task, result = node_output["past_steps"][-1]
                    print(f"已完成步骤：{task}")
                    print(f"步骤结果：{result[:100]}..." if len(result) > 100 else f"步骤结果：{result}")
                if "response" in node_output and node_output["response"]:
                    print(f"直接回复：{node_output['response']}")


# 运行主程序
if __name__ == "__main__":
    asyncio.run(main())