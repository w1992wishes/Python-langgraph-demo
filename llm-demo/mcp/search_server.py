from mcp.server.fastmcp import FastMCP
import logging
from typing import Dict, List

# 配置日志记录器
logging.basicConfig(
    level=logging.INFO,  # 设置日志级别为 INFO
    format="%(asctime)s - %(levelname)s - %(message)s"  # 日志格式
)
logger = logging.getLogger(__name__)

mcp = FastMCP("Search")

# 初始化一些内容，以关键字-内容列表的形式存储
SEARCH_CONTENTS: Dict[str, List[str]] = {
    "python": [
        "Python是一种高级编程语言，以其简洁易读的语法而闻名。",
        "Python支持多种编程范式，包括面向对象、命令式和函数式编程。",
        "Python拥有丰富的标准库和第三方库，适用于数据分析、人工智能等多个领域。"
    ],
    "机器学习": [
        "机器学习是人工智能的一个分支，研究计算机如何在没有明确编程的情况下学习。",
        "监督学习、无监督学习和强化学习是机器学习的主要类型。",
        "机器学习算法已被广泛应用于图像识别、自然语言处理等领域。"
    ],
    "北京": [
        "北京是中国的首都，也是全国的政治、文化和国际交往中心。",
        "北京拥有故宫、长城等众多著名历史文化遗迹。",
        "北京是一座现代化大都市，同时保留了丰富的历史文化底蕴。"
    ],
    "足球": [
        "足球是一项全球性的体育运动，每队有11名球员在场上比赛。",
        "国际足联（FIFA）负责组织全球最高水平的足球赛事，包括世界杯。",
        "足球比赛的目标是将球踢进对方球门，得分多的一方获胜。"
    ],
    "健康": [
        "健康不仅指身体没有疾病，还包括心理和社会适应能力的良好状态。",
        "均衡饮食、适量运动和充足睡眠是保持健康的重要因素。",
        "定期体检有助于及早发现健康问题，提高治疗效果。"
    ]
}


@mcp.tool()
async def search_by_keyword(keyword: str) -> str:
    """Search content by keyword."""
    logger.info("The search_by_keyword method is called: keyword=%s", keyword)

    # 转换为小写以实现不区分大小写的检索
    keyword_lower = keyword.lower()

    # 检查是否有完全匹配的关键字
    if keyword_lower in SEARCH_CONTENTS:
        contents = SEARCH_CONTENTS[keyword_lower]
        return f"找到关于'{keyword}'的内容：\n" + "\n".join([f"- {content}" for content in contents])

    # 检查是否有关键字包含检索词
    matched = []
    for key, contents in SEARCH_CONTENTS.items():
        if keyword_lower in key:
            matched.extend([f"- {content}" for content in contents])

    if matched:
        return f"找到与'{keyword}'相关的内容：\n" + "\n".join(matched)

    # 如果没有找到匹配的内容
    return f"没有找到关于'{keyword}'的内容。"


if __name__ == "__main__":
    logger.info("Start keyword search server through MCP")  # 记录服务启动日志
    mcp.run(transport="sse")
