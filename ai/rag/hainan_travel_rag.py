# from langchain_openai import ChatOpenAI
# from langchain.embeddings.base import Embeddings
# from langchain_community.vectorstores import Milvus
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
# from langchain_core.pydantic_v1 import BaseModel, Field
# from dotenv import load_dotenv
# import json
# import time
# import openai
# import re
# from config import Config
#
# load_dotenv()  # 加载环境变量
#
# def create_embedding(client, text):
#     """
#     Creates an embedding for the given text using the OpenAI client.
#     """
#     return client.embeddings.create(model="embedding", input=text).data[0].embedding
#
# def create_openai_client(api_key, base_url):
#     """
#     Creates and returns an OpenAI client.
#     """
#     return openai.Client(api_key=api_key, base_url=base_url)
#
# # 1. 创建Milvus知识库
# def create_milvus_knowledge_base():
#     """
#     创建Milvus向量数据库存储旅游知识
#     """
#     # Initialize clients
#     openai_client = create_openai_client(Config.OPENAI_API_KEY, Config.OPENAI_BASE_URL)
#
#     documents = [
#         {
#             "title": "海南岛经典4天3晚行程",
#             "content": """**行程安排：**
# Day1：抵达→椰海时光 - 建议入住白沙洲区域酒店，傍晚欣赏日落
# Day2：浮潜一日游 - 推荐珊瑚花园和小鱼群聚集点（约$50- $70/人，含午餐）
# Day3：文化村+夜市 - 参观海南文化村（$10/票），晚上逛夜市品尝海鲜
# Day4：购物+返程 - 上午可去购物街买伴手礼
#
# **住宿推荐：**
# 经济型酒店A（$50/晚），舒适型酒店B（$100-$150/晚）
# """
#         },
#         {
#             "title": "海南岛旅游预算指南",
#             "content": """**中等消费水平4天3晚人均预算：**
# - 住宿（舒适型）：$300-$450（$100-$150/晚×3晚）
# - 餐饮：$130-$180（普通餐厅$10-$15/餐，海鲜大餐$25-$40）
# - 活动：$70（浮潜$50+文化村$10）
# - 交通：$50-$70（岛内Tutu车/租摩托车）
# - 杂费：$50-$50（水/小吃/纪念品）
# **总计：$600-$800/人（不含国际机票）**
# 旺季价格上浮10%-20%
# """
#         },
#         {
#             "title": "海南岛景点活动价格",
#             "content": """**热门活动价格清单：**
# - 浮潜一日游：$50-$70（含装备、午餐、接送）
# - 深潜体验：$90-$120
# - 海南文化村门票：$10
# - 森林公园门票：$15
# - 交通船票价：$20/单程
# - 租摩托车：$15/天 + 油费
# """
#         },
#         {
#             "title": "海南岛住宿餐饮参考",
#             "content": """**住宿价格范围：**
# - 经济型：$40-$80/晚（如酒店A）
# - 舒适型：$100-$150/晚（如酒店B，近海滩）
# - 豪华型：$300+/晚（含早餐）
#
# **餐饮消费水平：**
# - 普通餐厅：$8-$15/餐
# - 海鲜大餐：$25-$40/人
# - 夜市小吃：$3-$8/份
# - 瓶装水：$1-$2
# """
#         },
#         {
#             "title": "海南岛旅游注意事项",
#             "content": """**重要提示：**
# 1. 最佳旅游时间：11月-次年4月（旱季）
# 2. 必备物品：防晒霜、泳衣、防水袋、转换插头
# 3. 文化习俗：进入村落需脱鞋，尊重当地信仰
# 4. 交通：岛内主要靠Tutu车（起步$5）或租摩托车
# 5. 适合人群：情侣/朋友（浮潜活动），亲子需注意安全
# """
#         }
#     ]
#
#     from langchain_core.documents import Document
#     docs = [
#         Document(
#             page_content=f"{doc['title']}\n{doc['content']}",
#             metadata={"source": doc["title"], "category": "travel_guide"}
#         )
#         for doc in documents
#     ]
#
#     from langchain.text_splitter import RecursiveCharacterTextSplitter
#     # 文档分块
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=500,
#         chunk_overlap=50,
#         separators=["\n\n", "\n", ". ", "!", "?"]
#     )
#     splits = text_splitter.split_documents(docs)
#
#     # 自定义Embeddings类
#     class CustomOpenAIEmbeddings(Embeddings):
#         def __init__(self, openai_client):
#             self.openai_client = openai_client
#
#         def embed_query(self, text):
#             return create_embedding(self.openai_client, text)
#
#         def embed_documents(self, texts):
#             return [self.embed_query(text) for text in texts]
#
#     # 连接参数 - 根据你的Milvus配置修改
#     MILVUS_HOST = "172.11.17.23"  # 连接你的Milvus服务地址
#     MILVUS_PORT = "19530"
#     COLLECTION_NAME = "travel_knowledge_base"
#
#     # 构建Milvus向量数据库
#     vector_db = Milvus.from_documents(
#         documents=splits,
#         embedding=CustomOpenAIEmbeddings(openai_client),
#         connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT},
#         collection_name=COLLECTION_NAME,
#         drop_old=True  # 如果存在同名集合则删除重建
#     )
#
#     # 等待索引构建完成
#     print("等待Milvus索引构建...")
#     time.sleep(5)
#
#     return vector_db.as_retriever(search_kwargs={"k": 5})
#
# from langchain_core.messages import AIMessage
# # 输出清理函数
# def clean_llm_output(msg: AIMessage):
#     """
#     清除JSON外的所有内容
#     """
#     text = msg.content
#     # 方法1：提取首个完整JSON对象
#     if '{' in text and '}' in text:
#         start = text.index('{')
#         end = text.index('}') + 1
#         return text[start:end]
#     # 方法2：正则提取
#     json_match = re.search(pattern=r'\{.*?\}', text)
#     return json_match.group(0) if json_match else text
#
# # 2. 意图识别组件（保持不变）
# class IntentRecognitionOutput(BaseModel):
#     """
#     意图识别的输出结构
#     """
#     primary_intents: list[str] = Field(description="主要意图列表")
#     entities: dict = Field(description="识别出的实体")
#     slots_to_fill: list[str] = Field(description="需要填充的槽位")
#     implicit_needs: list[str] = Field(description="隐含需求")
#
# def create_intent_recognition_chain():
#     """
#     创建意图识别链
#     """
#     prompt = ChatPromptTemplate.from_messages([
#         ("system", """你是一个旅游领域意图识别专家。分析用户查询，识别：
# 1. 主要意图（Primary Intents）- 从列表中选择：[PlanItinerary(行程规划), EstimateBudget(预算估算), FindAccommodation(查找住宿), FindActivities(查找活动), GetTravelTips(获取旅行建议)]
# 2. 实体（Entities）- 如目的地、日期等
# 3. 需要填充的槽位（Slots to Fill）
# 4. 隐含需求（Implicit Needs）
#
# 输出必须是严格JSON格式，使用以下结构：
# {"primary_intents": ["intent1", "intent2"], "entities": {"key": "value"}, "slots_to_fill": ["slot1", "slot2"], "implicit_needs": ["need1", "need2"]}
#
# 注意：只输出JSON，不要包含其他内容。"""),
#         ("user", "用户查询：{query}")
#     ])
#
#     model = ChatOpenAI(temperature=0, model="coder")
#     parser = JsonOutputParser(pydantic_object=IntentRecognitionOutput)
#
#     return prompt | model | clean_llm_output | parser
#
# # 3. 需求改写组件（保持不变）
# def create_query_rewriting_chain():
#     """
#     创建需求改写链
#     """
#     prompt = ChatPromptTemplate.from_messages([
#         ("system", """你是一个旅游助手，负责将模糊的用户查询改写为适合检索的查询。
# 根据意图识别结果，生成2-3个改写后的查询，要求：
# 1. 填补常见槽位（如天数=4天3晚，预算=中等）
# 2. 添加相关关键词（行程、预算、景点、住宿等）
# 3. 覆盖不同角度（行程模板 vs 预算明细）
# 4. 输出格式：{"rewritten_queries": ["query1", "query2", "query3"]}
#
# 只输出JSON，不要包含其他内容。"""),
#         ("user", """原始查询：{query}
# 意图分析结果：{intent_analysis}""")
#     ])
#
#     model = ChatOpenAI(temperature=0.3, model="coder")
#     parser = JsonOutputParser()
#
#     return prompt | model | clean_llm_output | parser
#
# # 4. 最终生成组件（优化）
# def create_generation_chain():
#     """
#     创建最终回答生成链
#     """
#     prompt = ChatPromptTemplate.from_messages([
#         ("system", """你是一个专业的海岛旅游规划助手。根据以下信息生成回答：
# - 用户原始问题：{query}
# - 意图分析结果：{intent_analysis}
# - 检索到的相关信息：{context}
#
# 回答要求：
# 1. 结构化行程：提供清晰的4天3晚行程安排（基于检索结果）
# 2. 详细预算：分项估算中等消费水平的费用（基于检索结果）
# 3. 处理模糊性：明确说明方法基于4天3晚和中等预算
# 4. 引导个性化：询问兴趣/预算/时间/同行人员等信息
# 5. 语气：专业、热情、易懂
# 6. 格式：使用Markdown格式组织内容
#
# 重要规则：
# - 只使用检索到的信息，不要编造数据！
# - 如果检索信息不足，明确告知用户
# - 在预算部分使用表格清晰展示"""),
#     ])
#
#     model = ChatOpenAI(temperature=0.2, model="coder")
#     return prompt | model | StrOutputParser()
#
# # 5. 完整RAG流程（优化）
# def travel_planning_rag(query: str):
#     """
#     完整的旅游规划RAG流程
#     """
#     # 步骤1：创建Milvus知识库检索器
#     retriever = create_milvus_knowledge_base()
#
#     # 步骤2：意图识别
#     intent_chain = create_intent_recognition_chain()
#     intent_analysis = intent_chain.invoke({"query": query})
#
#     # 步骤3：需求改写
#     rewrite_chain = create_query_rewriting_chain()
#     rewrite_result = rewrite_chain.invoke({
#         "query": query,
#         "intent_analysis": json.dumps(intent_analysis, ensure_ascii=False)
#     })
#     rewritten_queries = rewrite_result["rewritten_queries"]
#
#     # 步骤4：检索（使用改写后的查询）
#     all_contexts = []
#     for q in rewritten_queries:
#         contexts = retriever.invoke(q)
#         all_contexts.extend([ctx.page_content for ctx in contexts])
#
#     # 去重并限制上下文长度
#     unique_contexts = list(dict.fromkeys(all_contexts))[:5]
#     context_str = "\n\n---\n\n".join(unique_contexts)
#
#     # 步骤5：生成最终答案
#     generation_chain = create_generation_chain()
#     final_answer = generation_chain.invoke({
#         "query": query,
#         "intent_analysis": json.dumps(intent_analysis, indent=2, ensure_ascii=False),
#         "context": context_str
#     })
#
#     # 返回详细日志用于演示
#     return {
#         "original_query": query,
#         "intent_analysis": intent_analysis,
#         "rewritten_queries": rewritten_queries,
#         "retrieved_contexts": unique_contexts,
#         "final_answer": final_answer
#     }
#
# # 6. 主程序执行
# if __name__ == "__main__":
#     # 用户原始查询
#     user_query = "去海南岛玩几天怎么安排比较好？大概要花多少钱？"
#
#     print("=" * 88)
#     print(f"用户原始查询：{user_query}")
#     print("=" * 88)
#
#     # 执行完整RAG流程
#     result = travel_planning_rag(user_query)
#
#     # 打印详细过程
#     print("\n🔍 意图识别结果：")
#     print(json.dumps(result["intent_analysis"], indent=2, ensure_ascii=False))
#
#     print("\n📝 需求改写结果：")
#     for i, q in enumerate(result["rewritten_queries"], 1):
#         print(f"改写查询 ({i}): {q}")
#
#     print("\n📚 检索到的关键信息：")
#     for i, ctx in enumerate(result["retrieved_contexts"], 1):
#         print(f"\n上下文片段 ({i}):")
#         print(ctx.strip())
#
#     print("\n💬 最终生成的回答：")
#     print("-" * 50)
#     print(result["final_answer"])
#     print("-" * 50)