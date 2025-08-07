import asyncio
import os
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from openai import OpenAI
from langchain_deepseek import ChatDeepSeek

# 初始化 DeepSeek 大模型客户端
llm = ChatDeepSeek(
    model="deepseek-chat", # 指定 DeepSeek 的模型名称
    api_key=os.getenv("DS_API_KEY") # 替换为您自己的 DeepSeek API 密钥
)

# 解析并输出结果
def print_optimized_result(agent_response):
    """
    解析代理响应并输出优化后的结果。
    :param agent_response: 代理返回的完整响应
    """
    messages = agent_response.get("messages", [])
    steps = []  # 用于记录计算步骤
    final_answer = None  # 最终答案

    for message in messages:
        if hasattr(message, "additional_kwargs") and "tool_calls" in message.additional_kwargs:
            # 提取工具调用信息
            tool_calls = message.additional_kwargs["tool_calls"]
            for tool_call in tool_calls:
                tool_name = tool_call["function"]["name"]
                tool_args = tool_call["function"]["arguments"]
                steps.append(f"调用工具: {tool_name}({tool_args})")
        elif message.type == "tool":
            # 提取工具执行结果
            tool_name = message.name
            tool_result = message.content
            steps.append(f"{tool_name} 的结果是: {tool_result}")
        elif message.type == "ai":
            # 提取最终答案
            final_answer = message.content

    # 打印优化后的结果
    print("\n计算过程:")
    for step in steps:
        print(f"- {step}")
    if final_answer:
        print(f"\n最终答案: {final_answer}")


# 定义异步主函数
async def main():
    # 创建客户端，不再使用上下文管理器
    client = MultiServerMCPClient(
        {
            "weather": {
                "url": "http://localhost:8000/sse",
                "transport": "sse",
            }
        }
    )

    # 获取工具
    tools = await client.get_tools()
    agent = create_react_agent(llm, tools)

    # 循环接收用户输入
    while True:
        try:
            # 提示用户输入问题
            user_input = input("\n请输入您的问题（或输入 'exit' 退出）：")
            if user_input.lower() == "exit":
                print("感谢使用！再见！")
                break

            # 调用代理处理问题
            agent_response = await agent.ainvoke({"messages": [{"role": "user", "content": user_input}]})

            # 调用抽取的方法处理输出结果
            print_optimized_result(agent_response)
        except Exception as e:
            print(f"发生错误：{e}")
            continue
    # 关闭客户端连接
    await client.close()


# 使用 asyncio 运行异步主函数
if __name__ == "__main__":
    asyncio.run(main())