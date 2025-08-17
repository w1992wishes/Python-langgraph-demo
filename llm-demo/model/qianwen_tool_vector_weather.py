import os
import json
import jieba  # 新增中文分词库
from openai import OpenAI
import requests
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


# --------------------------
# 1. 优化后的伪向量检索MCP Server（基于jieba分词匹配）
# --------------------------
class PseudoVectorServer:
    def __init__(self, persist_directory="./pseudo_rag_db"):
        self.documents = []  # 存储文档片段
        os.makedirs(persist_directory, exist_ok=True)
        self.persist_directory = persist_directory

    def load_documents(self, doc_path: str = None, text_content: str = None):
        """加载文档到伪向量库"""
        try:
            if text_content:
                documents = [Document(page_content=text_content, metadata={"source": "manual_text"})]
            elif doc_path:
                loader = TextLoader(doc_path, encoding="utf-8")
                documents = loader.load()
            else:
                return "请提供文档路径或文本内容"

            # 分割文档
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                separators=["\n\n", "\n", "。", "，", " "]
            )
            splits = text_splitter.split_documents(documents)
            self.documents.extend(splits)
            return f"文档加载成功，共添加 {len(splits)} 个片段到伪向量库"
        except Exception as e:
            return f"文档加载失败：{str(e)}"

    def search(self, query: str, k: int = 3) -> dict:
        """优化的检索逻辑：用jieba分词+模糊匹配"""
        results = []
        # 1. 中文分词（提取查询中的有意义词汇）
        query_words = list(jieba.cut(query))  # 分词结果示例：["年假", "有", "多少", "天"]

        # 2. 遍历文档片段，计算匹配度
        for doc in self.documents:
            doc_content = doc.page_content
            match_count = 0
            # 对每个分词进行模糊匹配
            for word in query_words:
                if len(word) < 2:  # 过滤无意义的单字（如“有”“的”）
                    continue
                if word in doc_content:  # 关键词出现在文档中
                    match_count += 1
            if match_count > 0:
                results.append({
                    "content": doc_content,
                    "metadata": doc.metadata,
                    "match_score": match_count  # 匹配到的关键词数量
                })

        # 3. 按匹配度排序，取前k个
        results.sort(key=lambda x: x["match_score"], reverse=True)
        return {
            "query": query,
            "results": results[:k]
        }

# --------------------------
# 2. 天气查询MCP Server（复用之前的高德天气API）
# --------------------------
class WeatherMCPServer:
    def __init__(self):
        self.amap_key = os.getenv("AMAP_WEATHER_KEY")
        self.base_url = "https://restapi.amap.com/v3/weather/weatherInfo"
        if not self.amap_key:
            raise ValueError("请设置环境变量 AMAP_WEATHER_KEY（高德天气API Key）")

    def get_weather(self, city: str, is_forecast: bool = False) -> dict:
        params = {
            "key": self.amap_key,
            "city": city,
            "extensions": "all" if is_forecast else "base",
            "output": "json"
        }
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            data = response.json()
            if data.get("status") != "1":
                return {"error": f"查询失败：{data.get('info', '未知错误')}"}

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
# 3. 千义模型主控制器（自动判断调用哪个工具）
# --------------------------
class QwenAssistant:
    def __init__(self):
        # 初始化千义客户端
        self.client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        # 初始化两个MCP Server（使用伪向量库）
        self.weather_server = WeatherMCPServer()
        self.vector_server = PseudoVectorServer()  # 改用伪向量服务

    def run(self, user_query: str):
        """处理用户查询，自动判断是否调用工具"""
        # 1. 系统提示：定义工具调用规则
        system_prompt = """
        你是智能助手，可调用以下工具：
        1. 天气查询：用户问天气（如“北京天气”“上海明天热吗”），返回格式：
           {"action": "get_weather", "city": "城市名", "is_forecast": true/false}
           （is_forecast：问“明天”“未来”则为true，否则为false）
        2. 向量检索：用户问知识库相关问题（如“公司制度”“产品说明”），返回格式：
           {"action": "vector_search", "query": "查询内容"}
        其他问题直接回答，不调用工具。
        """

        # 2. 第一步：让模型判断是否调用工具
        tool_judge = self.client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ]
        )
        tool_command = tool_judge.choices[0].message.content.strip()

        # 3. 解析指令，调用对应工具
        try:
            tool_json = json.loads(tool_command)
            action = tool_json.get("action")

            # 调用天气工具
            if action == "get_weather":
                weather_data = self.weather_server.get_weather(
                    city=tool_json["city"],
                    is_forecast=tool_json["is_forecast"]
                )
                context = f"天气数据：{json.dumps(weather_data, ensure_ascii=False)}"

            # 调用伪向量检索工具
            elif action == "vector_search":
                search_data = self.vector_server.search(query=tool_json["query"])
                context = f"检索结果：{json.dumps(search_data, ensure_ascii=False)}"

            # 工具调用后的回答生成
            final_answer = self.client.chat.completions.create(
                model="qwen-plus",
                messages=[
                    {"role": "system", "content": "根据工具返回的上下文，用自然语言回答用户问题，简洁准确。"},
                    {"role": "user", "content": f"用户问：{user_query}\n{context}"}
                ]
            )
            return final_answer.choices[0].message.content

        # 无需调用工具，直接回答
        except (json.JSONDecodeError, KeyError):
            return tool_command

    def init_knowledge(self, doc_path: str = None, text_content: str = None):
        """初始化知识库（支持文件或直接传入文本）"""
        return self.vector_server.load_documents(doc_path=doc_path, text_content=text_content)


# --------------------------
# 4. 使用示例
# --------------------------
if __name__ == "__main__":
    # 初始化助手
    assistant = QwenAssistant()

    # 往伪向量库添加示例内容
    sample_knowledge = """
    【公司年假制度】
    1. 入职满1年的员工可享受5天年假，满3年可享受10天年假，满5年可享受15天年假。
    2. 年假需提前3天申请，由部门经理审批。

    【产品说明：智能手表X1】
    1. 续航：普通模式下可使用7天，省电模式下可使用15天。
    2. 价格：标准版1299元，Pro版1899元。
    """

    print("加载示例数据到伪向量库...")
    load_result = assistant.init_knowledge(text_content=sample_knowledge)
    print(load_result)

    # 测试向量检索效果
    print("\n===== 测试伪向量检索 =====")
    test_queries = ["年假有多少天？", "智能手表X1多少钱？"]
    for query in test_queries:
        print(f"\n检索查询：{query}")
        result = assistant.vector_server.search(query=query)
        for i, item in enumerate(result["results"], 1):
            print(f"结果{i}：{item['content'][:100]}...")

    # 原有功能测试
    print("\n===== 工具调用测试 =====")
    print("测试1（天气查询）：")
    print(assistant.run("广州今天的天气如何？"))

    print("\n测试2（向量检索）：")
    print(assistant.run("智能手表X1的续航时间是多久？"))

    print("\n测试3（直接回答）：")
    print(assistant.run("你能做什么？"))
