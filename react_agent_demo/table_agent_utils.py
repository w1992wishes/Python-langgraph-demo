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
from typing import Optional, Dict, Any, List
from table_model_store import table_model_store


# 定义工具的输入参数模型
class AddTableInput(BaseModel):
    name: str = Field(..., description="表名称")
    description: str = Field(..., description="表描述")
    fields: List[Dict[str, Any]] = Field(..., description="表字段列表，每个字段包含name(字段名)、type(字段类型)、description(描述，可选)、required(是否必填，可选)")
    primary_key: Optional[str] = Field(None, description="主键字段名称")


class UpdateTableInput(BaseModel):
    name: str = Field(..., description="表名称")
    description: Optional[str] = Field(None, description="表描述")
    fields: Optional[List[Dict[str, Any]]] = Field(None, description="表字段列表，每个字段包含name(字段名)、type(字段类型)、description(描述，可选)、required(是否必填，可选)")
    primary_key: Optional[str] = Field(None, description="主键字段名称")


class DeleteTableInput(BaseModel):
    name: str = Field(..., description="要删除的表名称")


class GetTableInput(BaseModel):
    name: str = Field(..., description="要查询的表名称")


class ListTablesInput(BaseModel):
    pass


# 异步工具函数
async def run_add_table(name: str, description: str, fields: List[Dict[str, Any]], 
                       primary_key: Optional[str] = None) -> Dict[str, Any]:
    """添加一个新的表模型"""
    return await table_model_store.add_table(name, description, fields, primary_key)


async def run_update_table(name: str, description: Optional[str] = None,
                         fields: Optional[List[Dict[str, Any]]] = None,
                         primary_key: Optional[str] = None) -> Dict[str, Any]:
    """更新一个已有的表模型"""
    return await table_model_store.update_table(name, description, fields, primary_key)


async def run_delete_table(name: str) -> Dict[str, Any]:
    """删除一个表模型"""
    return await table_model_store.delete_table(name)


async def run_get_table(name: str) -> Dict[str, Any]:
    """获取单个表模型的详细信息"""
    return await table_model_store.get_table(name)


async def run_list_tables() -> Dict[str, Any]:
    """列出所有表模型"""
    return await table_model_store.list_tables()


# 创建工具列表
table_tools = [
    StructuredTool.from_function(
        func=run_add_table,
        name="AddTable",
        description="添加一个新的表模型",
        args_schema=AddTableInput
    ),
    StructuredTool.from_function(
        func=run_update_table,
        name="UpdateTable",
        description="更新一个已有的表模型",
        args_schema=UpdateTableInput
    ),
    StructuredTool.from_function(
        func=run_delete_table,
        name="DeleteTable",
        description="删除一个表模型",
        args_schema=DeleteTableInput
    ),
    StructuredTool.from_function(
        func=run_get_table,
        name="GetTable",
        description="获取单个表模型的详细信息",
        args_schema=GetTableInput
    ),
    StructuredTool.from_function(
        func=run_list_tables,
        name="ListTables",
        description="列出所有表模型",
        args_schema=ListTablesInput
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


# 创建表模型Agent
def create_table_agent():
    # 获取语言模型
    llm = get_llm()

    # 创建提示模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个智能表模型助手，可以帮助用户管理各种表模型。使用提供的工具来完成用户的请求。"),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])

    # 创建工具调用Agent
    agent = create_tool_calling_agent(
        llm=llm,
        tools=table_tools,
        prompt=prompt
    )

    # 创建记忆组件
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # 创建Agent执行器
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=table_tools,
        verbose=True
    )

    return agent_executor


# 表模型会话管理器，用于存储和检索基于session_id的agent_executor实例
class TableAgentSessionManager:
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
        base_agent = create_agent()
        
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


# 创建全局表模型会话管理器实例
table_session_manager = TableAgentSessionManager()