import os
import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import asyncio
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 配置日志
log_level = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 导入各个模块的会话管理器
from metric_agent_utils import metric_session_manager
from table_agent_utils import table_session_manager
from etl_agent_utils import etl_session_manager

# 创建FastAPI应用
app = FastAPI(
    title="智能数据助手API",
    description="提供自然语言交互的指标管理、表模型管理和ETL开发服务",
    version="1.0.0",
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该设置具体的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 请求和响应模型
class MessageRequest(BaseModel):
    message: str = Field(..., description="用户消息内容")
    session_id: Optional[str] = Field(None, description="会话ID，可选")


class MessageResponse(BaseModel):
    message: str = Field(..., description="智能体回复内容")
    session_id: Optional[str] = Field(None, description="会话ID")


# 全局异常处理器
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"全局异常: {str(exc)}")
    logger.exception("异常详情:")
    return JSONResponse(
        status_code=500,
        content={"detail": "内部服务器错误"}
    )


# 核心对话接口
@app.post("/chat/metric", response_model=MessageResponse)
async def chat_with_metric_agent(request: MessageRequest):
    """
    与指标管理智能体对话的端点
    """
    session_id = request.session_id
    message = request.message
    logger.info(f"收到指标智能体对话请求: session_id={session_id}, message={message[:50]}...")
    
    try:
        # 获取或创建与session_id关联的agent_executor实例
        agent_executor = await metric_session_manager.get_or_create_agent(session_id)
        
        # 执行agent - 带会话参数
        result = await agent_executor.ainvoke(
            {"input": message},
            config={"configurable": {"session_id": session_id}}
        )
        
        # 从结果中提取输出信息
        if isinstance(result, dict):
            # 尝试多种可能的输出键名
            for key in ["output", "content", "response"]:
                if key in result:
                    response_message = result[key]
                    break
            else:
                response_message = str(result)
        else:
            response_message = str(result)
        
        logger.info(f"指标智能体响应生成成功: session_id={session_id}")
        return MessageResponse(
            message=response_message,
            session_id=session_id
        )
    except Exception as e:
        logger.error(f"指标智能体对话过程中出错: {str(e)}")
        logger.exception("指标智能体对话错误详情:")
        raise HTTPException(status_code=500, detail=f"指标智能体对话失败: {str(e)}")


@app.post("/chat/table", response_model=MessageResponse)
async def chat_with_table_agent(request: MessageRequest):
    """
    与表模型管理智能体对话的端点
    """
    session_id = request.session_id
    message = request.message
    logger.info(f"收到表模型智能体对话请求: session_id={session_id}, message={message[:50]}...")
    
    try:
        # 获取或创建与session_id关联的agent_executor实例
        agent_executor = await table_session_manager.get_or_create_agent(session_id)
        
        # 执行agent - 带会话参数
        result = await agent_executor.ainvoke(
            {"input": message},
            config={"configurable": {"session_id": session_id}}
        )
        
        # 从结果中提取输出信息
        if isinstance(result, dict):
            # 尝试多种可能的输出键名
            for key in ["output", "content", "response"]:
                if key in result:
                    response_message = result[key]
                    break
            else:
                response_message = str(result)
        else:
            response_message = str(result)
        
        logger.info(f"表模型智能体响应生成成功: session_id={session_id}")
        return MessageResponse(
            message=response_message,
            session_id=session_id
        )
    except Exception as e:
        logger.error(f"表模型智能体对话过程中出错: {str(e)}")
        logger.exception("表模型智能体对话错误详情:")
        raise HTTPException(status_code=500, detail=f"表模型智能体对话失败: {str(e)}")


@app.post("/chat/etl", response_model=MessageResponse)
async def chat_with_etl_agent(request: MessageRequest):
    """
    与ETL开发智能体对话的端点
    """
    session_id = request.session_id
    message = request.message
    logger.info(f"收到ETL智能体对话请求: session_id={session_id}, message={message[:50]}...")
    
    try:
        # 获取或创建与session_id关联的agent_executor实例
        agent_executor = await etl_session_manager.get_or_create_agent(session_id)
        
        # 执行agent - 带会话参数
        result = await agent_executor.ainvoke(
            {"input": message},
            config={"configurable": {"session_id": session_id}}
        )
        
        # 从结果中提取输出信息
        if isinstance(result, dict):
            # 尝试多种可能的输出键名
            for key in ["output", "content", "response"]:
                if key in result:
                    response_message = result[key]
                    break
            else:
                response_message = str(result)
        else:
            response_message = str(result)
        
        logger.info(f"ETL智能体响应生成成功: session_id={session_id}")
        return MessageResponse(
            message=response_message,
            session_id=session_id
        )
    except Exception as e:
        logger.error(f"ETL智能体对话过程中出错: {str(e)}")
        logger.exception("ETL智能体对话错误详情:")
        raise HTTPException(status_code=500, detail=f"ETL智能体对话失败: {str(e)}")


# 健康检查端点
@app.get("/health")
async def health_check():
    """
    健康检查端点
    """
    logger.info("健康检查请求")
    try:
        return {"status": "healthy", "message": "智能数据助手API服务运行正常"}
    except Exception as e:
        logger.error(f"健康检查失败: {str(e)}")
        raise HTTPException(status_code=503, detail="服务不可用")


# 根路径
@app.get("/")
async def root():
    """
    API根路径
    """
    return {
        "message": "智能数据助手API服务",
        "version": "1.0.0",
        "available_endpoints": [
            {"method": "POST", "path": "/chat/metric", "description": "与指标管理智能体对话"},
            {"method": "POST", "path": "/chat/table", "description": "与表模型管理智能体对话"},
            {"method": "POST", "path": "/chat/etl", "description": "与ETL开发智能体对话"},
            {"method": "GET", "path": "/health", "description": "健康检查"}
        ]
    }


async def main():
    # 获取配置信息
    port = int(os.getenv("PORT", "8000"))
    log_level = os.getenv("LOG_LEVEL", "INFO")
    
    logger.info(f"正在启动API服务器，监听端口: {port}，日志级别: {log_level}")
    logger.info("可用的核心API端点:")
    logger.info("1. POST /chat/metric - 与指标管理智能体对话")
    logger.info("2. POST /chat/table - 与表模型管理智能体对话")
    logger.info("3. POST /chat/etl - 与ETL开发智能体对话")
    logger.info("4. GET /health - 健康检查")
    
    # 配置Uvicorn
    config = uvicorn.Config(
        app=app,
        host="0.0.0.0",
        port=port,
        log_level=log_level.lower(),
        reload=os.getenv("DEV_MODE", "false").lower() == "true"
    )
    
    # 启动服务器
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    import uvicorn
    
    try:
        # 检查是否已在事件循环中
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # 如果事件循环已经在运行，使用create_task
            loop.create_task(main())
            # 保持主线程运行
            try:
                loop.run_until_complete(asyncio.sleep(3600))  # 运行1小时
            except (KeyboardInterrupt, SystemExit):
                logger.info("服务器正在关闭...")
        else:
            # 如果事件循环未运行，直接运行main
            asyncio.run(main())
    except RuntimeError:
        # 如果无法获取事件循环，创建新的事件循环
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("接收到中断信号，正在关闭服务器...")
    except Exception as e:
        logger.error(f"服务器启动失败: {str(e)}")
        logger.exception("服务器启动错误详情:")
        raise