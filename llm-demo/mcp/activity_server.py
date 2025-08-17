from mcp.server.fastmcp import FastMCP
import logging
from typing import Dict, List, Tuple

# 配置日志记录器
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

mcp = FastMCP("ActivityRecommendation", port=8001)

# 定义天气类型与对应推荐活动的映射关系
# 每个天气类型包含多个可能的活动
WEATHER_ACTIVITIES: Dict[str, List[str]] = {
    # 晴天相关天气
    "晴天": [
        "去公园散步或野餐",
        "进行户外运动（如跑步、骑自行车）",
        "去郊外远足或爬山",
        "在户外进行烧烤活动",
        "去海滩享受阳光"
    ],
    # 雨天相关天气
    "雨天": [
        "看电影或电视剧",
        "阅读一本好书",
        "在家烹饪美食",
        "玩桌游或电子游戏",
        "参观室内博物馆或展览馆"
    ],
    # 多云相关天气
    "多云": [
        "进行轻度户外活动（如打羽毛球、钓鱼）",
        "去植物园或动物园",
        "在户外咖啡馆小坐",
        "进行摄影活动",
        "逛商场或购物中心"
    ],
    # 阴天相关天气
    "阴天": [
        "进行室内健身",
        "参观美术馆或画廊",
        "去图书馆学习",
        "尝试新的菜谱做一顿饭",
        "整理房间或衣柜"
    ],
    # 大雾相关天气
    "大雾": [
        "在家看纪录片",
        "学习新技能（如烹饪、编程）",
        "做瑜伽或冥想",
        "写日记或博客",
        "与家人朋友进行视频聊天"
    ],
    # 雪天相关天气
    "雪天": [
        "堆雪人或打雪仗",
        "滑雪或滑雪板运动",
        "喝热饮看雪景",
        "在家做烘焙",
        "看一部温馨的电影"
    ]
}

# 定义天气关键词与天气类型的映射，用于识别天气状况
WEATHER_KEYWORDS: List[Tuple[str, str]] = [
    ("阳光", "晴天"),
    ("晴空", "晴天"),
    ("晴朗", "晴天"),
    ("明媚", "晴天"),
    ("小雨", "雨天"),
    ("雨", "雨天"),
    ("阵雨", "雨天"),
    ("多云", "多云"),
    ("阴天", "阴天"),
    ("大雾", "大雾"),
    ("雾", "大雾"),
    ("雪", "雪天"),
    ("小雪", "雪天"),
    ("中雪", "雪天")
]


def determine_weather_type(weather_description: str) -> str:
    """根据天气描述确定天气类型"""
    for keyword, weather_type in WEATHER_KEYWORDS:
        if keyword in weather_description:
            return weather_type
    # 默认返回阴天类型
    return "阴天"


@mcp.tool()
async def get_recommended_activities(weather_description: str) -> str:
    """
    Get recommended activities based on weather description.
    :param weather_description: Description of current weather
    :return: Recommended activities
    """
    logger.info("Getting recommended activities for weather: %s", weather_description)

    # 确定天气类型
    weather_type = determine_weather_type(weather_description)
    logger.info("Determined weather type: %s", weather_type)

    # 获取该天气类型下的推荐活动
    activities = WEATHER_ACTIVITIES.get(weather_type, WEATHER_ACTIVITIES["阴天"])

    # 格式化返回结果
    return f"根据天气情况，推荐以下活动：\n" + "\n".join([f"- {activity}" for activity in activities])


if __name__ == "__main__":
    logger.info("Starting activity recommendation server through MCP")
    mcp.run(transport="sse")
