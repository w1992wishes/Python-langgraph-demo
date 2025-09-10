from langchain_core.messages import HumanMessage
from config import Settings
from graph import get_agent_chain


def main():
    """程序主函数（示例调用）"""
    # 1. 获取Agent状态链
    agent_chain = get_agent_chain()

    # 2. 示例查询（数学计算任务，保留原有逻辑）
    user_query = "计算 ((3*(4+5)/0.5)+3245)+8，再计算 32/4.23，最后求和"
    #user_query = "What's the oldest parrot alive, and how much longer is that than the average?中文输出"

    input_messages = [HumanMessage(content=user_query)]

    # 3. 流式执行Agent
    print(f"开始执行查询：{user_query}")
    print("=" * 50)

    # -------------------------- 关键修改：增强打印逻辑 --------------------------
    stream_iter = agent_chain.stream(
        {"messages": input_messages},
        {"recursion_limit": Settings.RECURSION_LIMIT}
    )

    # 迭代所有节点输出，确保plan_and_schedule被捕获
    for step_idx, step in enumerate(stream_iter, 1):
        print(f"\n【步骤 {step_idx}】- 捕获节点输出：")
        # 检查step是否为空（防止无输出）
        if not step or not isinstance(step, dict):
            print("  警告：当前步骤无有效节点输出")
            continue

        # 遍历所有节点（确保plan_and_schedule被打印）
        for node_name, node_output in step.items():
            print(f"  节点: {node_name}")  # 强制打印节点名称
            # 检查节点输出是否含messages
            if isinstance(node_output, dict) and "messages" in node_output:
                messages = node_output["messages"]
                if not messages:
                    print("    节点输出：无messages")
                else:
                    for msg in messages:
                        print(f"    消息类型: {type(msg).__name__}")
                        print(f"    内容: {msg.content[:200]}..." if len(
                            str(msg.content)) > 200 else f"    内容: {msg.content}")
                        if hasattr(msg, "additional_kwargs") and msg.additional_kwargs:
                            print(f"    附加信息: {msg.additional_kwargs}")
            else:
                print(f"    节点输出：格式错误（无messages字段），原始输出：{str(node_output)[:100]}...")
        print("-" * 50)
    # ----------------------------------------------------------------------------------------

    # # 其他示例查询（注释保留，可按需启用）
    # example_1 = "What's the GDP of New York?"
    # example_2 = "What's the oldest parrot alive, and how much longer is that than the average?"
    # for step in agent_chain.stream(
    #     {"messages": [HumanMessage(content=example_2)]},
    #     {"recursion_limit": Settings.RECURSION_LIMIT}
    # ):
    #     print(step)
    #     print("---")


if __name__ == "__main__":
    main()