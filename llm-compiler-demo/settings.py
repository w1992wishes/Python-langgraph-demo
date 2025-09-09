import os

class Settings:
    """应用配置管理"""
    # OpenAI 相关配置
    OPENAI_API_KEY = os.getenv("DASHSCOPE_API_KEY")
    OPENAI_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    # LLM 模型配置
    LLM_MODEL = "qwen-plus"  # 模型名称，可根据实际调整
    TEMPERATURE = 0.1  # 采样温度
    MAX_TOKENS = 1000  # 最大生成 tokens

    # Agent 执行配置
    MAX_ITERATIONS = 10  # 最大迭代次数
    VERBOSE = True  # 详细日志开关

    # 缓存配置
    CACHE_TTL = 3600  # 缓存过期时间（秒，1小时）
    MAX_CACHE_SIZE = 1000  # 最大缓存条目数
