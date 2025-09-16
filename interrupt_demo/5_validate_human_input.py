from typing import TypedDict
import uuid

from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver
from IPython.display import Image, display  # 用于图表显示


# 1. 定义图状态（存储年龄）
class State(TypedDict):
    age: int


# 2. 年龄获取与验证节点（触发中断等待人工输入）
def get_valid_age(state: State) -> State:
    # 初始提示语
    prompt = "Please enter your age (must be a non-negative integer)."

    while True:
        # 触发中断：将当前提示语传给用户，等待输入
        user_input = interrupt(prompt)

        # 输入验证逻辑
        try:
            age = int(user_input)  # 尝试转为整数
            if age < 0:  # 验证非负
                raise ValueError("Age must be non-negative.")
            break  # 输入有效，退出循环
        except (ValueError, TypeError):
            # 输入无效，更新提示语（下次中断时会传给用户）
            prompt = f"'{user_input}' is not valid. Please enter a non-negative integer for age."

    # 返回验证后的年龄，更新状态
    return {"age": age}


# 3. 年龄报告节点（使用验证后的年龄）
def report_age(state: State) -> State:
    print(f"\n✅ Human is {state['age']} years old.")  # 打印最终结果
    return state


# 4. 构建图结构
builder = StateGraph(State)
# 添加节点
builder.add_node("get_valid_age", get_valid_age)  # 年龄获取与验证
builder.add_node("report_age", report_age)  # 年龄报告
# 定义流程走向
builder.set_entry_point("get_valid_age")  # 入口：获取年龄
builder.add_edge("get_valid_age", "report_age")  # 验证通过 → 报告年龄
builder.add_edge("report_age", END)  # 报告完成 → 流程结束

# 5. 配置检查点（支持中断状态保存）
checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)  # 编译图

# 6. 可视化图结构（可选，生成PNG或打印错误）
try:
    img_path = "5_graph.png"
    # 生成Mermaid图表PNG
    graph_image = graph.get_graph(xray=True).draw_mermaid_png()
    with open(img_path, "wb") as img_file:
        img_file.write(graph_image)
    print(f"📊 图结构已保存到：{img_path}")
    display(Image(img_path))  # IPython环境下直接显示图片
except Exception as e:
    print(f"⚠️ 可视化图表失败：{e}")
    # 备选：打印Mermaid代码（手动复制到 https://mermaid-js.github.io/mermaid-live-editor/ 可视化）
    print("\n📝 Mermaid图代码（手动可视化）：")
    print(graph.get_graph(xray=True).to_mermaid())

# 7. 核心逻辑：手动输入年龄（替代硬编码模拟）
if __name__ == "__main__":
    # 配置流程（指定唯一thread_id，确保中断状态一致）
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}  # 动态生成唯一ID

    # 第一次调用图：触发初始中断（请求输入年龄）
    graph.invoke({}, config=config)

    # 循环：直到输入有效（无中断则流程结束）
    while True:
        # 获取当前图状态（检查是否有未处理的中断）
        current_state = graph.get_state(config)

        # 若没有中断 → 流程已完成（年龄验证通过，进入report_age），退出循环
        if not current_state.interrupts:
            break

        # 若有中断 → 提取中断提示语，让用户手动输入
        interrupt_info = current_state.interrupts[0]  # 获取当前中断信息
        prompt = interrupt_info.value  # 提取提示语（如"Please enter your age..."或错误提示）

        # 显示提示语，获取用户手动输入
        user_input = input(f"\n{prompt}\nYour input: ").strip()

        # 用用户输入恢复流程（resume=用户输入）
        graph.invoke(Command(resume=user_input), config=config)

    # 流程结束后，获取最终状态（可选）
    final_state = graph.get_state(config).values
    print("\n🎉 流程完成！最终状态：", final_state)