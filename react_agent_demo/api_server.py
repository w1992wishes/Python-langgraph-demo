import os
import asyncio
import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional
from agent_utils import session_manager
from metrics_store import MetricsStore, metrics_store
from dotenv import load_dotenv

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api_server.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("api_server")

# 加载环境变量
load_dotenv()

# 创建FastAPI应用
app = FastAPI(
    title="智能指标助手API",
    description="用于管理指标的AI助手API服务",
    version="1.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该设置具体的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 全局异常处理器
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """捕获所有未处理的异常并记录详细信息"""
    logger.error(f"未处理的异常 - 路径: {request.url.path}, 方法: {request.method}")
    logger.exception("异常详情:")  # 这会记录完整的堆栈跟踪
    
    # 对于开发环境，可以返回更多信息；对于生产环境，应该返回更通用的错误消息
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "服务器内部错误",
            "error_type": type(exc).__name__,
            "detail": str(exc) if os.getenv("ENVIRONMENT") != "production" else "请联系管理员"
        }
    )

# 直接使用从metrics_store模块导入的实例

# 定义请求和响应模型
class MessageRequest(BaseModel):
    message: str
    session_id: Optional[str] = None  # 可选的会话ID，用于区分不同用户

class MessageResponse(BaseModel):
    response: str
    session_id: Optional[str] = None
    status: str = "success"

class ErrorResponse(BaseModel):
    status: str = "error"
    message: str

class MetricRequest(BaseModel):
    name: str
    value: float
    tags: Optional[Dict[str, Any]] = None

# 存储会话记忆的字典
sessions: Dict[str, Any] = {}

@app.post("/chat", response_model=MessageResponse)
async def chat_endpoint(request: MessageRequest):
    """
    与智能指标助手进行对话的端点
    
    - **message**: 用户的自然语言请求
    - **session_id**: 可选的会话ID，用于保持对话上下文
    """
    logger.info(f"收到聊天请求: {request.message}, 会话ID: {request.session_id}")
    try:
        # 获取与session_id关联的agent_executor实例
        logger.debug("获取会话对应的agent_executor实例")
        agent_executor = await session_manager.get_or_create_agent(request.session_id)
        
        # 使用会话特定的agent_executor处理请求
        logger.debug("开始调用agent_executor处理请求")
        result = await agent_executor.ainvoke({"input": request.message})
        logger.info(f"agent_executor执行完成，响应: {result}")
        
        output = result.get("output", "抱歉，我无法处理您的请求。")
        logger.info(f"返回聊天响应: {output}")
        return MessageResponse(
            response=output,
            session_id=request.session_id
        )
    except Exception as e:
        logger.error(f"处理聊天请求时出错: {str(e)}")
        logger.exception("聊天请求错误详情:")  # 记录完整的堆栈跟踪
        raise HTTPException(status_code=500, detail=f"处理请求时出错: {str(e)}")

@app.delete("/chat/session/{session_id}", response_model=Dict[str, str])
async def clear_session(session_id: str):
    """
    清除指定会话的上下文数据
    
    - **session_id**: 要清除的会话ID
    """
    logger.info(f"收到清除会话请求: {session_id}")
    try:
        session_manager.clear_session(session_id)
        logger.info(f"会话 {session_id} 已成功清除")
        return {"message": f"会话 {session_id} 已成功清除"}
    except Exception as e:
        logger.error(f"清除会话时出错: {str(e)}")
        logger.exception("清除会话错误详情:")
        raise HTTPException(status_code=500, detail=f"清除会话时出错: {str(e)}")

@app.delete("/chat/sessions", response_model=Dict[str, str])
async def clear_all_sessions():
    """
    清除所有会话的上下文数据
    """
    logger.info("收到清除所有会话请求")
    try:
        session_manager.clear_all_sessions()
        logger.info("所有会话已成功清除")
        return {"message": "所有会话已成功清除"}
    except Exception as e:
        logger.error(f"清除所有会话时出错: {str(e)}")
        logger.exception("清除所有会话错误详情:")
        raise HTTPException(status_code=500, detail=f"清除所有会话时出错: {str(e)}")

@app.get("/chat/sessions/status", response_model=Dict[str, Any])
async def get_sessions_status():
    """
    获取当前会话状态信息
    
    返回当前活跃会话数量等信息
    """
    logger.info("收到获取会话状态请求")
    try:
        active_count = session_manager.get_active_session_count()
        logger.info(f"当前活跃会话数量: {active_count}")
        return {
            "active_sessions": active_count,
            "session_timeout": "1小时"
        }
    except Exception as e:
        logger.error(f"获取会话状态时出错: {str(e)}")
        logger.exception("获取会话状态错误详情:")
        raise HTTPException(status_code=500, detail=f"获取会话状态时出错: {str(e)}")

@app.get("/metrics", response_model=Dict[str, Any])
async def list_metrics():
    """
    获取所有指标的端点
    """
    logger.info("列出所有指标请求")
    try:
        result = await metrics_store.list_metrics()
        logger.info(f"返回指标列表成功")
        return result
    except Exception as e:
        logger.error(f"列出指标失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取指标列表失败: {str(e)}")

@app.get("/metrics/{metric_name}", response_model=Dict[str, Any])
async def get_metric(metric_name: str):
    """
    获取单个指标详情的端点
    """
    logger.info(f"获取指标请求: {metric_name}")
    try:
        result = await metrics_store.get_metric(metric_name)
        if result["status"] == "error":
            logger.warning(f"指标 {metric_name} 不存在")
            raise HTTPException(status_code=404, detail=result["message"])
        logger.info(f"成功获取指标 {metric_name}")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取指标 {metric_name} 失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取指标详情失败: {str(e)}")

@app.post("/health")
async def health_check():
    """
    健康检查端点
    """
    logger.info("健康检查请求")
    try:
        # 可以在这里添加更多的健康检查逻辑，比如检查数据库连接等
        return {"status": "healthy", "message": "智能指标助手API服务运行正常"}
    except Exception as e:
        logger.error(f"健康检查失败: {str(e)}")
        raise HTTPException(status_code=503, detail="服务不可用")

@app.post("/metrics")
async def add_metric(metric: MetricRequest):
    """
    添加新的指标
    """
    logger.info(f"添加指标请求: {metric.name}, 值: {metric.value}")
    try:
        await metrics_store.add_metric(
            name=metric.name,
            value=metric.value,
            tags=metric.tags or {}
        )
        logger.info(f"指标 {metric.name} 添加成功")
        return {"status": "success", "message": "指标添加成功"}
    except Exception as e:
        logger.error(f"添加指标 {metric.name} 失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """
    根路径端点
    """
    return {
        "message": "欢迎使用智能指标助手API",
        "docs": "/docs",
        "redoc": "/redoc"
    }

# 启动服务器的主函数
async def main():
    import uvicorn
    
    # 设置日志级别
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logger.setLevel(getattr(logging, log_level))
    
    port = int(os.getenv("PORT", 8000))
    logger.info(f"正在启动API服务器，监听端口: {port}，日志级别: {log_level}")
    logger.info("API端点:")
    logger.info("- GET  /health        - 健康检查")
    logger.info("- POST /chat          - 聊天接口")
    logger.info("- GET  /metrics       - 获取所有指标")
    logger.info("- GET  /metrics/{name} - 获取特定指标")
    logger.info("- POST /metrics       - 添加新指标")
    
    try:
        # 配置热重载选项，避免无限循环
        # 1. 获取环境变量决定是否启用热重载
        use_reload = os.getenv("ENABLE_RELOAD", "false").lower() == "true"
        
        # 2. 如果启用热重载，配置排除项
        reload_config = {}
        if use_reload:
            reload_config = {
                "reload": True,
                "reload_excludes": ["*.log", "__pycache__", "*.pyc"],  # 排除日志文件和编译文件
                "reload_dirs": ["."]  # 只监控当前目录
            }
        
        logger.info(f"启动配置: 热重载={use_reload}")
        
        # 3. 启动服务器
        # 使用uvicorn.Server和uvicorn.Config类在异步上下文中正确运行服务器
        config = uvicorn.Config(
            "api_server:app",
            host="0.0.0.0",
            port=port,
            log_level=log_level.lower(),
            **reload_config
        )
        server = uvicorn.Server(config)
        
        # 在异步函数中直接await server.serve()
        await server.serve()
    except KeyboardInterrupt:
        logger.info("收到中断信号，正在关闭服务器...")
    except Exception as e:
        logger.error(f"服务器启动失败: {str(e)}")
        logger.exception("启动错误详情:")

if __name__ == "__main__":
    # 检查是否已经在事件循环中运行
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # 如果事件循环已经在运行，使用create_task调用main()
            loop.create_task(main())
            # 等待任务完成（在交互式环境中可能需要）
            try:
                loop.run_until_complete(asyncio.sleep(0))
            except RuntimeError:
                pass
        else:
            # 否则使用asyncio.run()
            asyncio.run(main())
    except RuntimeError:
        # 如果无法获取事件循环，使用asyncio.run()
        try:
            asyncio.run(main())
        except RuntimeError as e:
            logger.error(f"启动失败: {str(e)}")
            logger.error("请尝试在非异步环境中运行此脚本")