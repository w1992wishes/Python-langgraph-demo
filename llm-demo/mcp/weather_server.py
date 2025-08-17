from mcp.server.fastmcp import FastMCP
import logging
import random
from typing import List

# 配置日志记录器
logging.basicConfig(
    level=logging.INFO,  # 设置日志级别为 INFO
    format="%(asctime)s - %(levelname)s - %(message)s"  # 日志格式
)
logger = logging.getLogger(__name__)

mcp = FastMCP("Weather")

# 定义多种天气状况和描述
WEATHER_CONDITIONS: List[str] = [
    "天气阳光明媚，晴空万里，非常适合户外活动。",
    "今天有小雨，建议带伞，气温较低请注意保暖。",
    "多云转晴，风力适中，体感舒适。",
    "阴天，可能会有雷阵雨，出行请留意天气变化。",
    "大雾弥漫，能见度较低，驾车请减速慢行。",
    "小雪转中雪，路面可能结冰，请注意安全。",
    "晴朗少云，夜间气温较低，昼夜温差较大。",
    "微风，天气凉爽，是个适合出游的好日子。"
]


@mcp.tool()
async def get_weather(location: str) -> str:
    """Get weather for location."""
    logger.info("The get_weather method is called: location=%s", location)

    # 随机选择一个天气状况返回
    random_weather = random.choice(WEATHER_CONDITIONS)

    # 在返回结果中包含地点信息，使回答更具体
    return f"{location}当前{random_weather}"


if __name__ == "__main__":
    logger.info("Start weather server through MCP")  # 记录服务启动日志
    mcp.run(transport="sse")
