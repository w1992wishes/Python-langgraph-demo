import asyncio
import json
import aiohttp
from typing import Dict, Any

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
                if response.status == 200:
                    print("✅ 健康检查测试通过")
                    return True
                else:
                    print(f"❌ 健康检查测试失败: HTTP {response.status}")
                    return False
        except Exception as e:
            print(f"❌ 健康检查异常: {str(e)}")
            return False

async def test_session_management():
    """测试会话管理功能"""
    print("\n=== 测试会话管理功能 ===")
    try:
        async with aiohttp.ClientSession() as session:
            # 首先创建一个会话
            data = {
                "message": "创建一个会话",
                "session_id": "test-clear-session"
            }
            async with session.post(f"{BASE_URL}/chat", json=data) as response:
                if response.status != 200:
                    print(f"❌ 创建会话失败: HTTP {response.status}")
                    return False
                
            # 获取会话状态
            async with session.get(f"{BASE_URL}/chat/sessions/status") as response:
                if response.status == 200:
                    result = await response.json()
                    print(f"✅ 会话状态: {result}")
                else:
                    print(f"❌ 获取会话状态失败: HTTP {response.status}")
                    return False
                    
            # 测试清除特定会话
            async with session.delete(f"{BASE_URL}/chat/session/test-clear-session") as response:
                if response.status == 200:
                    result = await response.json()
                    print(f"✅ 清除会话成功: {result}")
                else:
                    print(f"❌ 清除会话失败: HTTP {response.status}")
                    print(await response.text())
                    return False
                    
            # 测试清除所有会话
            async with session.delete(f"{BASE_URL}/chat/sessions") as response:
                if response.status == 200:
                    result = await response.json()
                    print(f"✅ 清除所有会话成功: {result}")
                else:
                    print(f"❌ 清除所有会话失败: HTTP {response.status}")
                    print(await response.text())
                    return False
        print("✅ 会话管理功能测试通过")
        return True
    except Exception as e:
        print(f"❌ 会话管理测试异常: {str(e)}")
        return False


async def test_chat_endpoint():
    """测试聊天端点和会话管理"""
    print("\n=== 测试聊天端点和会话管理 ===")
    async with aiohttp.ClientSession() as session:
        try:
            # 测试1: 基本聊天
            print("\n测试1: 基本聊天")
            payload = {
                "message": "添加一个新指标，名称是CPU使用率，描述是系统CPU使用百分比，值是75.5，单位是%",
                "session_id": "test-session-1"
            }
            
            async with session.post(f"{BASE_URL}/chat", json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    print(f"✅ 响应成功: {result['response']}")
                    print(f"会话ID: {result.get('session_id', '未返回')}")
                else:
                    print(f"❌ 响应失败: HTTP {response.status}")
                    print(await response.text())
            
            # 测试2: 会话上下文保持 - 使用相同session_id
            print("\n测试2: 使用相同会话ID")
            payload2 = {
                "message": "查看所有指标",
                "session_id": "test-session-1"
            }
            
            async with session.post(f"{BASE_URL}/chat", json=payload2) as response:
                if response.status == 200:
                    result = await response.json()
                    print(f"✅ 响应成功: {result['response']}")
                    print(f"会话ID: {result.get('session_id', '未返回')}")
                else:
                    print(f"❌ 响应失败: HTTP {response.status}")
                    print(await response.text())
            
            # 测试3: 新会话 - 使用不同session_id
            print("\n测试3: 使用新会话ID")
            payload3 = {
                "message": "记得我刚才说什么吗？",
                "session_id": "test-session-2"
            }
            
            async with session.post(f"{BASE_URL}/chat", json=payload3) as response:
                if response.status == 200:
                    result = await response.json()
                    print(f"✅ 响应成功: {result['response']}")
                    print(f"会话ID: {result.get('session_id', '未返回')}")
                else:
                    print(f"❌ 响应失败: HTTP {response.status}")
                    print(await response.text())

            return True
        except Exception as e:
            print(f"❌ 测试异常: {str(e)}")
            return False


async def test_list_metrics():
    """测试列出所有指标端点"""
    print("\n=== 测试列出所有指标端点 ===")
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(f"{BASE_URL}/metrics") as response:
                if response.status == 200:
                    result = await response.json()
                    print(f"✅ 获取指标列表成功")
                    if result.get("status") == "success" and "data" in result:
                        print(f"找到 {len(result['data'])} 个指标:")
                        for metric in result['data']:
                            print(f"  - {metric.get('name')}: {metric.get('value')}{metric.get('unit')}")
                    return True
                else:
                    print(f"❌ 获取指标列表失败: HTTP {response.status}")
                    print(await response.text())
                    return False
        except Exception as e:
            print(f"❌ 列出指标端点异常: {str(e)}")
            return False


async def test_get_metric():
    """测试获取单个指标端点"""
    print("\n=== 测试获取单个指标端点 ===")
    metric_name = "CPU使用率"
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(f"{BASE_URL}/metrics/{metric_name}") as response:
                if response.status == 200:
                    result = await response.json()
                    print(f"✅ 获取指标 '{metric_name}' 成功")
                    if result.get("status") == "success" and "data" in result:
                        metric = result['data']
                        print(f"  名称: {metric.get('name')}")
                        print(f"  描述: {metric.get('description')}")
                        print(f"  值: {metric.get('value')}{metric.get('unit')}")
                    return True
                else:
                    print(f"❌ 获取指标 '{metric_name}' 失败: HTTP {response.status}")
                    print(await response.text())
                    return False
        except Exception as e:
            print(f"❌ 获取指标端点异常: {str(e)}")
            return False


async def test_error_handling():
    """测试错误处理"""
    print("\n=== 测试错误处理 ===")
    non_existent_metric = "不存在的指标"
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(f"{BASE_URL}/metrics/{non_existent_metric}") as response:
                if response.status == 404:
                    result = await response.json()
                    print(f"✅ 错误处理正常: 获取不存在指标返回404 - {result.get('detail')}")
                    return True
                else:
                    print(f"❌ 错误处理异常: 预期404，实际返回 {response.status}")
                    return False
        except Exception as e:
            print(f"❌ 错误处理测试异常: {str(e)}")
            return False


async def main():
    """主测试函数"""
    print("开始测试API服务...")
    print(f"API基础URL: {BASE_URL}")
    
    # 运行所有测试
    tests = [
        # test_health_check,
        test_chat_endpoint,
        test_session_management,
        # test_list_metrics,
        # test_get_metric,
        # test_error_handling
    ]
    
    results = []
    for test in tests:
        result = await test()
        print(result)
        results.append(result)
    
    # 打印总结
    print("\n=== 测试总结 ===")
    success_count = sum(results)
    total_count = len(results)
    
    print(f"总测试数: {total_count}")
    print(f"成功数: {success_count}")
    print(f"失败数: {total_count - success_count}")
    
    if success_count == total_count:
        print("🎉 所有测试通过！")
    else:
        print("⚠️  部分测试失败，请检查API服务是否正常运行。")
    
    print("\n注意: 确保API服务正在运行中才能正常测试。")
    print("您可以通过运行 'python api_server.py' 来启动API服务。")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n测试被用户中断。")
    except Exception as e:
        print(f"\n测试执行异常: {str(e)}")
        print("请确保API服务正在运行，并且地址配置正确。")