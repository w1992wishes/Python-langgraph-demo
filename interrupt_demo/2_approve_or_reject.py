from typing import Literal, TypedDict
import uuid

from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver


# 定义共享的图状态
class State(TypedDict):
    llm_output: str
    decision: str


# 模拟LLM输出节点
def generate_llm_output(state: State) -> State:
    return {"llm_output": "This is the generated output."}


# 人工审核节点
def human_approval(state: State) -> Command[Literal["approved_path", "rejected_path"]]:
    # 触发中断，等待人工输入
    decision = interrupt({
        "question": "Do you approve the following output?",
        "llm_output": state["llm_output"]
    })

    if decision == "approve":
        return Command(goto="approved_path", update={"decision": "approved"})
    else:
        return Command(goto="rejected_path", update={"decision": "rejected"})


# 审核通过后的步骤
def approved_node(state: State) -> State:
    print("✅ Approved path taken.")
    return state


# 审核拒绝后的步骤
def rejected_node(state: State) -> State:
    print("❌ Rejected path taken.")
    return state


# 构建图
builder = StateGraph(State)
builder.add_node("generate_llm_output", generate_llm_output)
builder.add_node("human_approval", human_approval)
builder.add_node("approved_path", approved_node)
builder.add_node("rejected_path", rejected_node)

builder.set_entry_point("generate_llm_output")
builder.add_edge("generate_llm_output", "human_approval")
builder.add_edge("approved_path", END)
builder.add_edge("rejected_path", END)

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

try:
    img_path = "2_graph.png"
    graph_image = graph.get_graph(xray=True).draw_mermaid_png()
    with open(img_path, "wb") as img_file:
        img_file.write(graph_image)
    from IPython.display import Image, display

    display(Image(img_path))
except Exception as e:
    print(f"可视化图表失败：{e}")

# 运行直到中断
config = {"configurable": {"thread_id": uuid.uuid4()}}
# 第一次调用会触发中断
graph.invoke({}, config=config)

# 获取中断信息
state = graph.get_state(config)
interrupt_info = state.interrupts[0]

# 显示审核问题和内容给用户
print("\n" + "=" * 50)
print(f"审核问题: {interrupt_info.value['question']}")
print(f"LLM 输出内容: {interrupt_info.value['llm_output']}")
print("=" * 50)

# 获取人工输入
while True:
    user_input = input("请输入审核结果 (approve/reject): ").strip().lower()
    if user_input in ["approve", "reject"]:
        break
    print("输入无效，请输入 'approve' 或 'reject'")

# 使用人工输入的结果继续流程
final_result = graph.invoke(Command(resume=user_input), config=config)
print("\n最终结果:", final_result)
