from typing import TypedDict
import uuid

from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver


# 定义图状态
class State(TypedDict):
    summary: str


# 模拟LLM生成摘要
def generate_summary(state: State) -> State:
    return {
        "summary": "The cat sat on the mat and looked at the stars."
    }


# 人工审核编辑节点
def human_review_edit(state: State) -> State:
    result = interrupt({
        "task": "Please review and edit the generated summary if necessary.",
        "generated_summary": state["summary"]
    })
    return {
        "summary": result["edited_summary"]
    }


# 模拟编辑后摘要的下游使用
def downstream_use(state: State) -> State:
    print(f"✅ Using edited summary: {state['summary']}")
    return state


# 构建图
builder = StateGraph(State)
builder.add_node("generate_summary", generate_summary)
builder.add_node("human_review_edit", human_review_edit)
builder.add_node("downstream_use", downstream_use)

builder.set_entry_point("generate_summary")
builder.add_edge("generate_summary", "human_review_edit")
builder.add_edge("human_review_edit", "downstream_use")
builder.add_edge("downstream_use", END)

# 设置内存中检查点以支持中断
checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

try:
    img_path = "3_graph.png"
    graph_image = graph.get_graph(xray=True).draw_mermaid_png()
    with open(img_path, "wb") as img_file:
        img_file.write(graph_image)
    from IPython.display import Image, display

    display(Image(img_path))
except Exception as e:
    print(f"可视化图表失败：{e}")

# 调用图直到遇到中断
config = {"configurable": {"thread_id": uuid.uuid4()}}
# 第一次调用会触发中断
graph.invoke({}, config=config)

# 获取中断信息
state = graph.get_state(config)
interrupt_info = state.interrupts[0]

# 显示生成的摘要给用户
print("\n" + "=" * 80)
print(f"任务: {interrupt_info.value['task']}")
print("\n生成的摘要:")
print(interrupt_info.value['generated_summary'])
print("=" * 80)

# 获取人工编辑的摘要
print("\n请编辑上述摘要（可直接按回车保持不变）：")
edited_summary = input("> ").strip()

# 如果用户没有输入，使用原始摘要
if not edited_summary:
    edited_summary = interrupt_info.value['generated_summary']

# 使用人工编辑的内容恢复图的执行
resumed_result = graph.invoke(
    Command(resume={"edited_summary": edited_summary}),
    config=config
)
print("\n最终结果:", resumed_result)
