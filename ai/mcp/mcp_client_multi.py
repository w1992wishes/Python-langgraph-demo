import asyncio
import os
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
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
            steps.append(f"{tool_name} 的结果是: {tool_result[:100]}..." if len(
                tool_result) > 100 else f"{tool_name} 的结果是: {tool_result}")
        elif message.type == "ai":
            # 提取最终答案
            final_answer = message.content

    # 打印优化后的结果
    print("\n处理过程:")
    for step in steps:
        print(f"- {step}")
    if final_answer:
        print(f"\n📝 综合建议: {final_answer}")


# 定义系统提示词，引导代理正确使用工具链
SYSTEM_PROMPT = """
你是一个智能活动顾问，能够根据天气情况推荐合适的活动并提供详细建议。
请遵循以下流程处理用户请求：

1. 当用户询问活动建议时，首先询问用户所在地点
2. 使用天气工具获取该地点的天气信息
3. 根据天气信息，使用活动推荐工具获取适合的活动列表
4. 选择1-2个最适合的活动，使用活动详情工具获取具体建议
5. 整理所有信息，为用户提供清晰、有用的综合建议

注意：
- 只在需要时调用工具，避免不必要的调用
- 确保工具调用参数正确（地点、天气描述、活动名称）
- 向用户呈现信息时要友好、自然，避免使用技术术语
"""


# 定义异步主函数
async def main():
    # 创建客户端，集成三个MCP服务
    client = MultiServerMCPClient(
        {
            "weather": {
                "url": "http://localhost:8000/sse",
                "transport": "sse",
            },
            "activity": {
                "url": "http://localhost:8001/sse",  # 假设活动推荐服务运行在8001端口
                "transport": "sse",
            },
            "activity_detail": {
                "url": "http://localhost:8002/sse",  # 假设活动详情服务运行在8002端口
                "transport": "sse",
            }
        }
    )

    # 获取所有工具
    tools = await client.get_tools()

    # 创建代理，添加系统提示词引导行为
    agent = create_react_agent(
        llm,
        tools,
        prompt=SYSTEM_PROMPT  # 注入系统提示
    )

    # 循环接收用户输入
    while True:
        try:
            # 提示用户输入问题
            user_input = input("\n请输入您的问题（或输入 'exit' 退出）：")
            if user_input.lower() == "exit":
                print("感谢使用！再见！")
                break

            # 调用代理处理问题
            agent_response = await agent.ainvoke({
                "messages": [{"role": "user", "content": user_input}]
            })

            # 调用抽取的方法处理输出结果
            print_optimized_result(agent_response)
        except Exception as e:
            print(f"发生错误：{e}")
            continue
    # 关闭客户端连接
    await client.close()


# 启动服务提示
def print_startup_message():
    print("=" * 50)
    print("欢迎使用智能活动推荐助手")
    print("已集成服务：")
    print("- 天气查询服务")
    print("- 活动推荐服务")
    print("- 活动详情建议服务")
    print("请输入您想了解的活动建议，例如：'今天适合做什么活动？'")
    print("=" * 50)


# 使用 asyncio 运行异步主函数
if __name__ == "__main__":
    print_startup_message()
    asyncio.run(main())