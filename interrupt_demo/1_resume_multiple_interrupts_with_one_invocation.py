from typing import TypedDict
import uuid
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import START
from langgraph.graph import StateGraph
from langgraph.types import interrupt, Command
from IPython.display import Image, display

class State(TypedDict):
    text_1: str
    text_2: str


def human_node_1(state: State):
    # 不使用name参数，仅传递必要数据
    return {"text_1": interrupt({"text_to_revise": state["text_1"]})}


def human_node_2(state: State):
    # 不使用name参数，仅传递必要数据
    return {"text_2": interrupt({"text_to_revise": state["text_2"]})}


graph_builder = StateGraph(State)
graph_builder.add_node("human_node_1", human_node_1)
graph_builder.add_node("human_node_2", human_node_2)

# 从START并行添加两个节点
graph_builder.add_edge(START, "human_node_1")
graph_builder.add_edge(START, "human_node_2")

checkpointer = InMemorySaver()
graph = graph_builder.compile(checkpointer=checkpointer)

try:
    img_path = "1_graph.png"
    graph_image = graph.get_graph(xray=True).draw_mermaid_png()
    with open(img_path, "wb") as img_file:
        img_file.write(graph_image)
    from IPython.display import Image, display

    display(Image(img_path))
except Exception as e:
    print(f"可视化图表失败：{e}")


thread_id = str(uuid.uuid4())
config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
# 第一次调用会触发中断
graph.invoke({"text_1": "original text 1", "text_2": "original text 2"}, config=config)

# 获取中断状态
state = graph.get_state(config)
interrupts = state.interrupts

resume_map = {}
for interrupt_item in interrupts:
    revised_text = input(f"请为 '{interrupt_item.value['text_to_revise']}' 输入修订内容：")
    resume_map[interrupt_item.interrupt_id] = revised_text

# 恢复执行
result = graph.invoke(Command(resume=resume_map), config=config)
print(result)
# 预期输出: {'text_1': 'edited text for original text 1', 'text_2': 'edited text for original text 2'}
