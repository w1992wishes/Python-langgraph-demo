import os
import json
from openai import OpenAI
import requests


# --------------------------
# 1. 天气MCP Server（基于高德天气API）
# --------------------------
class WeatherMCPServer:
    def __init__(self):
        self.amap_key = os.getenv("AMAP_WEATHER_KEY")  # 高德API Key（需提前设置）
        self.base_url = "https://restapi.amap.com/v3/weather/weatherInfo"
        if not self.amap_key:
            raise ValueError("请设置环境变量 AMAP_WEATHER_KEY（高德天气API Key）")

    def get_weather(self, city: str, is_forecast: bool = False) -> dict:
        """调用高德API查询天气（实时/预报）"""
        params = {
            "key": self.amap_key,
            "city": city,
            "extensions": "all" if is_forecast else "base",  # all=预报，base=实时
            "output": "json"
        }

        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            data = response.json()

            if data.get("status") != "1":
                return {"error": f"查询失败：{data.get('info', '未知错误')}"}

            # 格式化实时天气
            if not is_forecast:
                live = data["lives"][0]
                return {
                    "类型": "实时天气",
                    "城市": live["city"],
                    "时间": live["reporttime"],
                    "天气": live["weather"],
                    "温度": f"{live['temperature']}°C",
                    "风向": live["winddirection"],
                    "风力": live["windpower"],
                    "湿度": f"{live['humidity']}%"
                }

            # 格式化未来3天预报
            else:
                forecasts = data["forecasts"][0]["casts"]
                return {
                    "类型": "未来3天预报",
                    "城市": data["forecasts"][0]["city"],
                    "更新时间": data["forecasts"][0]["reporttime"],
                    "预报": [
                        {
                            "日期": day["date"],
                            "星期": ["周日", "周一", "周二", "周三", "周四", "周五", "周六"][int(day["week"])],
                            "天气": day["dayweather"],
                            "温度": f"{day['nighttemp']}°C ~ {day['daytemp']}°C",
                            "风向": day["daywind"],
                            "风力": day["daypower"]
                        } for day in forecasts
                    ]
                }

        except Exception as e:
            return {"error": f"接口调用失败：{str(e)}"}


# --------------------------
# 2. 千义模型调用逻辑（集成天气查询）
# --------------------------
class QwenWeatherAssistant:
    def __init__(self):
        # 初始化千义客户端
        self.client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        # 初始化天气MCP服务
        self.weather_server = WeatherMCPServer()

    def run(self, user_query: str):
        """处理用户查询：判断是否需要调用天气服务，生成回答"""
        # 1. 让模型决定是否需要调用工具（天气查询）
        system_prompt = """
        你是一个智能助手，可调用天气工具查询实时天气和未来预报。
        - 当用户问天气相关问题（如“北京天气”“上海明天热吗”），必须调用工具，返回格式：
          {"action": "get_weather", "city": "城市名", "is_forecast": true/false}
          （is_forecast：是否需要预报，问“明天”“未来”则为true，否则为false）
        - 其他问题直接回答，不调用工具。
        """

        # 2. 首次请求：让模型判断是否调用工具
        tool_response = self.client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ]
        )
        tool_result = tool_response.choices[0].message.content.strip()

        # 3. 解析模型响应：调用天气服务或直接回答
        try:
            # 若模型返回工具调用指令，则执行查询
            tool_json = json.loads(tool_result)
            if tool_json.get("action") == "get_weather":
                city = tool_json["city"]
                is_forecast = tool_json["is_forecast"]
                weather_data = self.weather_server.get_weather(city, is_forecast)

                # 4. 将天气数据传入模型，生成自然语言回答
                final_response = self.client.chat.completions.create(
                    model="qwen-plus",
                    messages=[
                        {"role": "system", "content": "根据天气数据，用自然语言回答用户问题，简洁明了。"},
                        {"role": "user",
                         "content": f"用户问：{user_query}，天气数据：{json.dumps(weather_data, ensure_ascii=False)}"}
                    ]
                )
                return final_response.choices[0].message.content

        except json.JSONDecodeError:
            # 若模型未返回工具调用格式，直接使用其回答（适用于非天气问题）
            return tool_result
        except Exception as e:
            return f"处理失败：{str(e)}"


# --------------------------
# 3. 测试示例
# --------------------------
if __name__ == "__main__":
    # 确保环境变量已设置（高德天气API Key和千义API Key）
    # Windows: $env:AMAP_WEATHER_KEY="你的高德Key"; $env:DASHSCOPE_API_KEY="你的千义Key"
    # macOS/Linux: export AMAP_WEATHER_KEY="你的高德Key"; export DASHSCOPE_API_KEY="你的千义Key"

    assistant = QwenWeatherAssistant()

    # 测试天气查询
    print(assistant.run("杭州今天天气怎么样？"))
    # 输出示例：杭州今天（2023-10-01 15:00更新）天气晴，温度26°C，东北风1级，湿度45%。

    # 测试预报查询
    print(assistant.run("上海未来3天天气如何？"))

    # 测试非天气问题
    print(assistant.run("你是谁？"))