import os
import time
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.tools import StructuredTool
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.agents import create_tool_calling_agent, AgentExecutor
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from etl_job_store import etl_job_store


# 定义工具的输入参数模型
class AddETLJobInput(BaseModel):
    name: str = Field(..., description="ETL作业名称")
    description: str = Field(..., description="ETL作业描述")
    data_source: Dict[str, Any] = Field(..., description="ETL作业数据源，包含type、connection_info等")
    transform_steps: List[Dict[str, Any]] = Field(..., description="ETL作业转换步骤列表")
    destination: Dict[str, Any] = Field(..., description="ETL作业目标，包含type、connection_info等")
    schedule: Optional[str] = Field(None, description="ETL作业调度表达式，如cron表达式")


class UpdateETLJobInput(BaseModel):
    name: str = Field(..., description="ETL作业名称")
    description: Optional[str] = Field(None, description="ETL作业描述")
    data_source: Optional[Dict[str, Any]] = Field(None, description="ETL作业数据源")
    transform_steps: Optional[List[Dict[str, Any]]] = Field(None, description="ETL作业转换步骤列表")
    destination: Optional[Dict[str, Any]] = Field(None, description="ETL作业目标")
    schedule: Optional[str] = Field(None, description="ETL作业调度表达式")


class DeleteETLJobInput(BaseModel):
    name: str = Field(..., description="要删除的ETL作业名称")


class GetETLJobInput(BaseModel):
    name: str = Field(..., description="要查询的ETL作业名称")


class ListETLJobsInput(BaseModel):
    pass


class GenerateETLCodeInput(BaseModel):
    name: str = Field(..., description="要生成代码的ETL作业名称")
    language: str = Field(default="python", description="生成代码的编程语言，默认为python")


# 异步工具函数
async def run_add_etl_job(name: str, description: str, data_source: Dict[str, Any],
                         transform_steps: List[Dict[str, Any]], destination: Dict[str, Any],
                         schedule: Optional[str] = None) -> Dict[str, Any]:
    """添加一个新的ETL作业"""
    return await etl_job_store.add_job(name, description, data_source, transform_steps, destination, schedule)


async def run_update_etl_job(name: str, description: Optional[str] = None,
                            data_source: Optional[Dict[str, Any]] = None,
                            transform_steps: Optional[List[Dict[str, Any]]] = None,
                            destination: Optional[Dict[str, Any]] = None,
                            schedule: Optional[str] = None) -> Dict[str, Any]:
    """更新一个已有的ETL作业"""
    return await etl_job_store.update_job(name, description, data_source, transform_steps, destination, schedule)


async def run_delete_etl_job(name: str) -> Dict[str, Any]:
    """删除一个ETL作业"""
    return await etl_job_store.delete_job(name)


async def run_get_etl_job(name: str) -> Dict[str, Any]:
    """获取单个ETL作业的详细信息"""
    return await etl_job_store.get_job(name)


async def run_list_etl_jobs() -> Dict[str, Any]:
    """列出所有ETL作业"""
    return await etl_job_store.list_jobs()


async def run_generate_etl_code(name: str, language: str = "python") -> Dict[str, Any]:
    """根据ETL作业配置生成代码"""
    return await etl_job_store.generate_etl_code(name, language)


# 创建工具列表
etl_tools = [
    StructuredTool.from_function(
        func=run_add_etl_job,
        name="AddETLJob",
        description="添加一个新的ETL作业",
        args_schema=AddETLJobInput
    ),
    StructuredTool.from_function(
        func=run_update_etl_job,
        name="UpdateETLJob",
        description="更新一个已有的ETL作业",
        args_schema=UpdateETLJobInput
    ),
    StructuredTool.from_function(
        func=run_delete_etl_job,
        name="DeleteETLJob",
        description="删除一个ETL作业",
        args_schema=DeleteETLJobInput
    ),
    StructuredTool.from_function(
        func=run_get_etl_job,
        name="GetETLJob",
        description="获取单个ETL作业的详细信息",
        args_schema=GetETLJobInput
    ),
    StructuredTool.from_function(
        func=run_list_etl_jobs,
        name="ListETLJobs",
        description="列出所有ETL作业",
        args_schema=ListETLJobsInput
    ),
    StructuredTool.from_function(
        func=run_generate_etl_code,
        name="GenerateETLCode",
        description="根据ETL作业配置生成代码",
        args_schema=GenerateETLCodeInput
    )
]


# 初始化语言模型
def get_llm():
    api_key = os.getenv("SILICONFLOW_API_KEY")
    if not api_key:
        raise ValueError("未设置SILICONFLOW_API_KEY环境变量")

    # 使用SiliconFlow API
    llm = ChatOpenAI(
        api_key=os.getenv("SILICONFLOW_API_KEY"),
        model="deepseek-ai/DeepSeek-V3.1",
        base_url="https://api.siliconflow.cn/v1/",
        temperature=0.1)
    return llm


# 创建ETL Agent
def create_etl_agent():
    # 获取语言模型
    llm = get_llm()

    # 创建提示模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个智能数据开发助手，可以帮助用户创建、管理和生成ETL作业代码。使用提供的工具来完成用户的请求。你需要：1) 分析用户需求，提取ETL作业的数据源、转换步骤和目标；2) 使用AddETLJob工具创建作业；3) 使用GenerateETLCode工具生成相应的ETL代码。"),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])

    # 创建工具调用Agent
    agent = create_tool_calling_agent(
        llm=llm,
        tools=etl_tools,
        prompt=prompt
    )

    # 创建Agent执行器
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=etl_tools,
        verbose=True
    )

    return agent_executor


# ETL作业会话管理器，用于存储和检索基于session_id的agent_executor实例
class ETLAgentSessionManager:
    def __init__(self, session_timeout=3600):
        # 存储session_id到agent_executor的映射
        self.session_agents = {}
        # 存储session_id到最后访问时间的映射
        self.session_last_access = {}
        # 会话超时时间（秒），默认1小时
        self.session_timeout = session_timeout
        # 用于存储每个会话的消息历史
        self._message_histories = {}
    
    def _get_message_history(self, session_id: str) -> BaseChatMessageHistory:
        """获取指定会话ID的消息历史"""
        if session_id not in self._message_histories:
            self._message_histories[session_id] = ChatMessageHistory()
        return self._message_histories[session_id]
    
    async def get_or_create_agent(self, session_id: Optional[str] = None):
        """
        获取或创建与session_id关联的etl agent_executor实例
        
        如果提供了session_id且已存在对应实例，则返回该实例并更新最后访问时间
        否则创建新实例并与session_id关联（如果提供了session_id）
        """
        # 先清理过期会话
        self._cleanup_expired_sessions()
        
        # 创建基本的agent_executor
        base_agent = create_etl_agent()
        
        # 如果没有提供session_id，创建临时会话（不存储）
        if not session_id:
            return base_agent
        
        # 检查是否已存在该会话的agent_executor
        if session_id not in self.session_agents:
            # 使用RunnableWithMessageHistory包装agent_executor
            with_message_history = RunnableWithMessageHistory(
                base_agent,
                self._get_message_history,
                input_messages_key="input",
                history_messages_key="chat_history"
            )
            # 存储包装后的agent_executor
            self.session_agents[session_id] = with_message_history
        
        # 更新最后访问时间
        self.session_last_access[session_id] = time.time()
        
        return self.session_agents[session_id]
    
    def clear_session(self, session_id: str):
        """清除特定会话的agent_executor实例"""
        if session_id in self.session_agents:
            del self.session_agents[session_id]
            del self.session_last_access[session_id]
    
    def clear_all_sessions(self):
        """清除所有会话"""
        self.session_agents.clear()
        self.session_last_access.clear()
    
    def _cleanup_expired_sessions(self):
        """清理过期的会话"""
        current_time = time.time()
        expired_sessions = [
            session_id for session_id, last_access in self.session_last_access.items()
            if current_time - last_access > self.session_timeout
        ]
        
        for session_id in expired_sessions:
            del self.session_agents[session_id]
            del self.session_last_access[session_id]
    
    def get_active_session_count(self):
        """获取当前活跃会话数"""
        self._cleanup_expired_sessions()
        return len(self.session_agents)


# 创建全局ETL会话管理器实例
etl_session_manager = ETLAgentSessionManager()