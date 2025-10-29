import os
import time
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.tools import StructuredTool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.agents import create_tool_calling_agent, AgentExecutor
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from metric_store import metrics_store


# 定义工具的输入参数模型
class AddMetricInput(BaseModel):
    name: str = Field(..., description="指标名称")
    description: str = Field(..., description="指标描述")
    value: float = Field(..., description="指标数值")
    unit: str = Field(..., description="指标单位")


class UpdateMetricInput(BaseModel):
    name: str = Field(..., description="指标名称")
    description: Optional[str] = Field(None, description="指标描述")
    value: Optional[float] = Field(None, description="指标数值")
    unit: Optional[str] = Field(None, description="指标单位")


class DeleteMetricInput(BaseModel):
    name: str = Field(..., description="要删除的指标名称")


class GetMetricInput(BaseModel):
    name: str = Field(..., description="要查询的指标名称")


class ListMetricsInput(BaseModel):
    pass


# 异步工具函数
async def run_add_metric(name: str, description: str, value: float, unit: str) -> Dict[str, Any]:
    """添加一个新的指标"""
    return await metrics_store.add_metric(name, description, value, unit)


async def run_update_metric(name: str, description: Optional[str] = None,
                            value: Optional[float] = None, unit: Optional[str] = None) -> Dict[str, Any]:
    """更新一个已有的指标"""
    return await metrics_store.update_metric(name, description, value, unit)


async def run_delete_metric(name: str) -> Dict[str, Any]:
    """删除一个指标"""
    return await metrics_store.delete_metric(name)


async def run_get_metric(name: str) -> Dict[str, Any]:
    """获取单个指标的详细信息"""
    return await metrics_store.get_metric(name)


async def run_list_metrics() -> Dict[str, Any]:
    """列出所有指标"""
    return await metrics_store.list_metrics()


# 创建工具列表
tools = [
    StructuredTool.from_function(
        func=run_add_metric,
        name="AddMetric",
        description="添加一个新的指标",
        args_schema=AddMetricInput
    ),
    StructuredTool.from_function(
        func=run_update_metric,
        name="UpdateMetric",
        description="更新一个已有的指标",
        args_schema=UpdateMetricInput
    ),
    StructuredTool.from_function(
        func=run_delete_metric,
        name="DeleteMetric",
        description="删除一个指标",
        args_schema=DeleteMetricInput
    ),
    StructuredTool.from_function(
        func=run_get_metric,
        name="GetMetric",
        description="获取单个指标的详细信息",
        args_schema=GetMetricInput
    ),
    StructuredTool.from_function(
        func=run_list_metrics,
        name="ListMetrics",
        description="列出所有指标",
        args_schema=ListMetricsInput
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


# 创建指标Agent
def create_metric_agent():
    # 获取语言模型
    llm = get_llm()

    # 创建提示模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个智能指标助手，可以帮助用户管理各种指标。使用提供的工具来完成用户的请求。"),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])

    # 创建工具调用Agent
    agent = create_tool_calling_agent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )

    # 创建Agent执行器
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True
    )

    return agent_executor


# 指标会话管理器，用于存储和检索基于session_id的agent_executor实例
class MetricAgentSessionManager:
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
        获取或创建与session_id关联的agent_executor实例
        
        如果提供了session_id且已存在对应实例，则返回该实例并更新最后访问时间
        否则创建新实例并与session_id关联（如果提供了session_id）
        """
        # 先清理过期会话
        self._cleanup_expired_sessions()
        
        # 创建基本的agent_executor
        base_agent = create_metric_agent()
        
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
            self.clear_session(session_id)
    
    def get_active_session_count(self):
        """获取当前活跃会话数量"""
        self._cleanup_expired_sessions()  # 先清理过期会话
        return len(self.session_agents)

# 创建全局指标会话管理器实例
metric_session_manager = MetricAgentSessionManager()
