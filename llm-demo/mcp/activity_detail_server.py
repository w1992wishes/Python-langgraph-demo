from mcp.server.fastmcp import FastMCP
import logging
from typing import Dict

# 配置日志记录器
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

mcp = FastMCP("ActivityDetail", port=8002)

# 运动活动详细建议的数据库
ACTIVITY_DETAILS: Dict[str, Dict[str, str]] = {
    "去公园散步或野餐": {
        "适宜时间": "早晨或傍晚，避免正午阳光过强",
        "注意事项": "提前查看天气预报，准备防晒用品，携带足够饮用水",
        "所需装备": "舒适的步行鞋、野餐垫、防晒帽、防晒霜、便携食品",
        "健康益处": "促进消化，增强心肺功能，缓解压力"
    },
    "进行户外运动（如跑步、骑自行车）": {
        "适宜时间": "日出后1小时或日落前2小时",
        "注意事项": "做好热身运动，控制运动强度，避免过度疲劳",
        "所需装备": "专业运动服装、运动鞋、水壶、运动手表（可选）",
        "健康益处": "增强心肺功能，提高新陈代谢，增强免疫力"
    },
    "去郊外远足或爬山": {
        "适宜时间": "清晨出发，避免单独行动，规划好返程时间",
        "注意事项": "提前了解路线难度，告知他人行程，携带急救包",
        "所需装备": "登山鞋、背包、登山杖、头灯、充足食物和水、保暖衣物",
        "健康益处": "锻炼全身肌肉，增强耐力，接触自然有益心理健康"
    },
    "在户外进行烧烤活动": {
        "适宜时间": "下午3点后，避开大风天气",
        "注意事项": "选择允许烧烤的区域，注意防火，食物确保烤熟",
        "所需装备": "烧烤炉、食材、烤具、炭火、一次性餐具、垃圾袋",
        "健康益处": "社交活动促进心理健康，适当放松缓解压力"
    },
    "看电影或电视剧": {
        "适宜时间": "任何时间段，建议每次不超过2小时",
        "注意事项": "保持适当距离，定时起身活动，避免久坐",
        "所需装备": "舒适的座椅或沙发、零食饮料（可选）",
        "健康益处": "放松身心，适当休息恢复精力"
    },
    "阅读一本好书": {
        "适宜时间": "清晨或睡前，选择安静的环境",
        "注意事项": "保持良好坐姿，注意眼睛休息，每小时远眺5分钟",
        "所需装备": "舒适的座椅、书签、阅读灯（光线不足时）",
        "健康益处": "增强知识储备，提高专注力，放松心情"
    },
    "堆雪人或打雪仗": {
        "适宜时间": "雪后天气转晴，气温不是极低的时候",
        "注意事项": "注意保暖，戴防水手套，避免长时间暴露在严寒中",
        "所需装备": "防水外套、防水手套、保暖帽子、围巾、防水靴",
        "健康益处": "增加活动量，促进血液循环，提升情绪"
    },
    "滑雪或滑雪板运动": {
        "适宜时间": "白天光线充足时，避开大风和暴雪天气",
        "注意事项": "佩戴护具，初学者应有教练指导，了解滑雪场规则",
        "所需装备": "滑雪服、滑雪板/雪板、头盔、护目镜、手套",
        "健康益处": "锻炼平衡能力，增强下肢力量，消耗卡路里"
    }
}


@mcp.tool()
async def explain_activity(activity_name: str) -> str:
    """
    Explain details of a specific activity, including suitable time, precautions, required equipment, etc.
    :param activity_name: Name of the activity to explain
    :return: Detailed explanation of the activity
    """
    logger.info("Explaining details for activity: %s", activity_name)

    # 查找活动详情
    details = ACTIVITY_DETAILS.get(activity_name)

    if details:
        # 格式化返回结果
        explanation = [f"关于「{activity_name}」的详细建议："]
        for key, value in details.items():
            explanation.append(f"- {key}：{value}")
        return "\n".join(explanation)
    else:
        return f"暂时没有「{activity_name}」的详细建议信息。"


if __name__ == "__main__":
    logger.info("Starting activity detail explanation server through MCP")
    mcp.run(transport="sse")
