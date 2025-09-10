import getpass
import os
from typing import Optional


def _get_pass(var: str) -> Optional[str]:
    """获取环境变量，不存在则交互式输入"""
    if var not in os.environ:
        os.environ[var] = getpass.getpass(f"{var}: ")
    return os.getenv(var)


class Settings:
    """全局配置类"""
    # 阿里云DashScope（兼容OpenAI格式）配置
    OPENAI_API_KEY = os.getenv("DASHSCOPE_API_KEY") or _get_pass("DASHSCOPE_API_KEY")
    OPENAI_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    # LLM模型配置
    LLM_MODEL = "qwen-plus"
    TEMPERATURE = 0.0  # 原有逻辑中温度为0，保持一致
    MAX_TOKENS = 1000

    # 工具配置
    TAVILY_MAX_RESULTS = 1  # 搜索工具最大结果数
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY") or _get_pass("TAVILY_API_KEY")

    # LangSmith配置（调试追踪）
    LANGCHAIN_PROJECT = "LLMCompiler"
    LANGSMITH_TRACING = "true"
    LANGSMITH_ENDPOINT = "https://api.smith.langchain.com"
    LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY") or _get_pass("LANGCHAIN_API_KEY")

    # Agent执行配置
    MAX_ITERATIONS = 10
    VERBOSE = True
    RECURSION_LIMIT = 100

    # 缓存配置（原有逻辑未启用，保留配置）
    CACHE_TTL = 3600
    MAX_CACHE_SIZE = 1000


# 初始化LangSmith环境变量
os.environ.update({
    "LANGCHAIN_PROJECT": Settings.LANGCHAIN_PROJECT,
    "LANGSMITH_TRACING": Settings.LANGSMITH_TRACING,
    "LANGSMITH_ENDPOINT": Settings.LANGSMITH_ENDPOINT,
    "LANGCHAIN_API_KEY": Settings.LANGCHAIN_API_KEY
})