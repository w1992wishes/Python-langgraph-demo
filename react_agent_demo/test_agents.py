import asyncio
import aiohttp
from typing import Optional

# API基础URL
BASE_URL = "http://localhost:8000"


async def test_health_check():
    """测试健康检查端点"""
    print("\n=== 测试健康检查端点 ===")
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(f"{BASE_URL}/health") as response:
                result = await response.json()
                print(f"健康检查响应: {result}")
                return response.status == 200
        except Exception as e:
            print(f"❌ 健康检查异常: {str(e)}")
            return False


async def test_agent_chat(endpoint: str, message: str, session_id: Optional[str] = None):
    """测试智能体对话端点"""
    print(f"\n=== 测试 {endpoint} 端点 ===")
    print(f"消息: {message}")
    
    payload = {
        "message": message,
        "session_id": session_id
    }
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(f"{BASE_URL}{endpoint}", json=payload) as response:
                result = await response.json()
                print(f"状态码: {response.status}")
                if response.status == 200:
                    print(f"✅ 响应成功: {result['message']}")
                    print(f"会话ID: {result.get('session_id', '未返回')}")
                else:
                    print(f"❌ 响应失败: {result}")
                return response.status == 200
        except Exception as e:
            print(f"❌ 请求异常: {str(e)}")
            return False


async def run_tests():
    """运行所有测试"""
    print("开始测试智能体API...")
    
    # 测试健康检查
    health_ok = await test_health_check()
    
    # 测试指标管理智能体
    metric_ok = await test_agent_chat(
        "/chat/metric",
        "添加一个新指标，名称是CPU使用率，描述是系统CPU使用百分比，值是75.5，单位是%"
    )
    
    # 测试表模型管理智能体
    table_ok = await test_agent_chat(
        "/chat/table",
        "创建一个用户表，包含id（整数）、username（字符串）和email（字符串）字段"
    )
    
    # 测试ETL开发智能体
    etl_ok = await test_agent_chat(
        "/chat/etl",
        "创建一个从CSV文件读取数据并写入数据库的ETL作业"
    )
    
    # 打印测试结果摘要
    print("\n=== 测试结果摘要 ===")
    print(f"健康检查: {'✅ 通过' if health_ok else '❌ 失败'}")
    print(f"指标管理智能体: {'✅ 通过' if metric_ok else '❌ 失败'}")
    print(f"表模型管理智能体: {'✅ 通过' if table_ok else '❌ 失败'}")
    print(f"ETL开发智能体: {'✅ 通过' if etl_ok else '❌ 失败'}")
    
    # 检查是否所有测试都通过
    all_passed = health_ok and metric_ok and table_ok and etl_ok
    print(f"\n总体测试结果: {'✅ 全部通过' if all_passed else '❌ 部分失败'}")
    
    return all_passed


if __name__ == "__main__":
    try:
        asyncio.run(run_tests())
    except KeyboardInterrupt:
        print("\n测试被用户中断")
    except Exception as e:
        print(f"\n测试运行异常: {str(e)}")