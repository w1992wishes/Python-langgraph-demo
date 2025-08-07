import os
import json
import jieba
from openai import OpenAI
import requests
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


# --------------------------
# 1. 优化后的伪向量检索MCP Server（支持多角度检索）
# --------------------------
class EnhancedVectorServer:
    def __init__(self, persist_directory="./enhanced_rag_db"):
        self.documents = []  # 存储文档片段
        os.makedirs(persist_directory, exist_ok=True)
        self.persist_directory = persist_directory

    def load_documents(self, doc_path: str = None, text_content: str = None):
        """加载文档到向量库"""
        try:
            if text_content:
                documents = [Document(page_content=text_content, metadata={"source": "manual_text"})]
            elif doc_path:
                loader = TextLoader(doc_path, encoding="utf-8")
                documents = loader.load()
            else:
                return "请提供文档路径或文本内容"

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                separators=["\n\n", "\n", "。", "，", " "]
            )
            splits = text_splitter.split_documents(documents)
            self.documents.extend(splits)
            return f"文档加载成功，共添加 {len(splits)} 个片段到向量库"
        except Exception as e:
            return f"文档加载失败：{str(e)}"

    def search(self, query: str, k: int = 2) -> dict:
        """基于jieba分词的增强检索"""
        results = []
        query_words = [word for word in jieba.cut(query) if len(word) >= 2]

        for doc in self.documents:
            doc_content = doc.page_content
            match_count = sum(1 for word in query_words if word in doc_content)
            if match_count > 0:
                results.append({
                    "content": doc_content,
                    "metadata": doc.metadata,
                    "match_score": match_count
                })

        results.sort(key=lambda x: x["match_score"], reverse=True)
        return {
            "query": query,
            "results": results[:k]
        }


# --------------------------
# 2. ReAct模型控制器（实现迭代式检索逻辑）
# --------------------------
class ReActAssistant:
    def __init__(self):
        # 初始化千问客户端
        self.client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        # 初始化工具
        self.weather_server = WeatherMCPServer()
        self.vector_server = EnhancedVectorServer()
        self.max_iterations = 3  # 最大检索迭代次数

    def transform_query(self, user_query: str) -> str:
        """将用户问题转换为陈述句检索词"""
        prompt = f"""将以下问题转换为陈述句形式的检索关键词，保留核心信息：
        问题：{user_query}
        陈述句："""

        response = self.client.chat.completions.create(
            model="qwen-plus",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()

    def generate_related_queries(self, original_query: str, context: str) -> list:
        """基于原始查询和已有上下文生成相关检索词（多角度）"""
        prompt = f"""基于原始问题和已获取的信息，生成2个不同角度的相关检索词（陈述句），用于补充检索：
        原始问题：{original_query}
        已有信息：{context}
        输出格式：["检索词1", "检索词2"]"""

        response = self.client.chat.completions.create(
            model="qwen-plus",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        try:
            return json.loads(response.choices[0].message.content.strip())
        except:
            return [f"{original_query} 补充信息", f"{original_query} 相关规定"]

    def run(self, user_query: str):
        """ReAct模式主流程：思考→行动→观察→迭代"""
        # 初始化思维链和检索结果
        thought_chain = []
        all_contexts = []

        # 1. 问题分类（判断是否需要工具）
        tool_type = self._determine_tool(user_query)
        if tool_type == "weather":
            return self._handle_weather(user_query)

        # 2. RAG流程：迭代检索增强
        original_query = user_query
        current_query = self.transform_query(user_query)
        thought_chain.append(f"1. 将问题转换为检索词：{current_query}")

        for i in range(self.max_iterations):
            # 执行检索
            search_result = self.vector_server.search(current_query)
            thought_chain.append(
                f"{i + 1}. 检索结果：{json.dumps([r['content'][:50] for r in search_result['results']], ensure_ascii=False)}")

            # 收集新信息
            new_contexts = [r["content"] for r in search_result["results"] if r["content"] not in all_contexts]
            all_contexts.extend(new_contexts)

            # 判断是否需要继续检索
            if not new_contexts:
                thought_chain.append(f"{i + 2}. 未获取新信息，停止检索")
                break

            # 生成下一轮检索词
            if i < self.max_iterations - 1:
                related_queries = self.generate_related_queries(original_query, "\n".join(all_contexts))
                current_query = related_queries[0]
                thought_chain.append(f"{i + 2}. 生成新检索词：{current_query}")

        # 3. 基于所有信息生成回答
        final_prompt = f"""根据以下信息回答用户问题，确保全面准确：
        用户问题：{original_query}
        参考信息：{json.dumps(all_contexts, ensure_ascii=False)}
        回答："""

        final_answer = self.client.chat.completions.create(
            model="qwen-plus",
            messages=[{"role": "user", "content": final_prompt}],
            temperature=0.7
        ).choices[0].message.content

        # 附加思维链（可选，用于调试）
        return f"回答：{final_answer}\n\n思维过程：\n" + "\n".join(thought_chain)

    def _determine_tool(self, query: str) -> str:
        """判断问题类型（天气/知识库/直接回答）"""
        prompt = f"""判断问题类型：
        1. 天气查询：包含城市名和天气相关词汇（如温度、预报）
        2. 知识库查询：关于公司制度、产品说明等预设知识
        3. 其他：直接回答
        问题：{query}
        输出类型（weather/knowledge/other）："""

        response = self.client.chat.completions.create(
            model="qwen-plus",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content.strip()

    def _handle_weather(self, query: str) -> str:
        """处理天气查询"""
        prompt = f"""解析天气查询：
        问题：{query}
        输出格式：{{"city": "城市名", "is_forecast": true/false}}"""

        try:
            response = self.client.chat.completions.create(
                model="qwen-plus",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            weather_params = json.loads(response.choices[0].message.content)
            weather_data = self.weather_server.get_weather(**weather_params)
            return f"天气查询结果：{json.dumps(weather_data, ensure_ascii=False)}"
        except Exception as e:
            return f"天气查询失败：{str(e)}"

    def init_knowledge(self, **kwargs):
        """初始化知识库"""
        return self.vector_server.load_documents(**kwargs)


# --------------------------
# 3. 天气查询MCP Server（保持不变）
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
# 4. 使用示例
# --------------------------
if __name__ == "__main__":
    # 初始化助手
    assistant = ReActAssistant()

    # 加载示例知识库
    sample_knowledge = """
    【公司年假制度】
    1. 入职满1年的员工可享受5天年假，满3年可享受10天年假，满5年可享受15天年假。
    2. 年假需提前3天申请，由部门经理审批。
    3. 年假可累积，但最多不超过20天，未使用的年假可按日工资的1.5倍折算。

    【产品说明：智能手表X1】
    1. 功能：支持心率监测、睡眠分析、运动记录（跑步/游泳/骑行模式）。
    2. 续航：普通模式下可使用7天，省电模式下可使用15天。
    3. 价格：标准版1299元，Pro版1899元（支持独立通话）。
    4. 售后：提供2年质保，支持7天无理由退换。

    【考勤制度】
    1. 工作时间：周一至周五9:00-18:00，午休1小时（12:00-13:00）。
    2. 迟到15分钟内扣50元，迟到30分钟以上按旷工半天处理。
    """
    print("加载示例数据到向量库...")
    print(assistant.init_knowledge(text_content=sample_knowledge))

    # 测试ReAct模式的RAG能力
    print("\n===== 测试迭代式检索 =====")
    test_queries = [
        "我入职3年，能休多少天年假？",
    ]

    for query in test_queries:
        print(f"\n===== 处理查询：{query} =====")
        print(assistant.run(query))
