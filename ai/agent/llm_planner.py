from typing import TypedDict, List, Literal, Optional, Annotated

# 新增日志记录器配置（所有节点通用）
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),  # 控制台输出
        logging.FileHandler("agent_workflow.log", encoding='utf-8')  # 关键修改
    ]
)
logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    input: str                   # 原始用户输入
    clarified_input: str         # 用户补充后的完整输入
    plan_type: Optional[Literal["query", "analysis"]]  # 任务类型
    task_list: List[str]          # 任务步骤列表
    current_step: int             # 当前执行步骤索引
    query_result: str             # 查询结果
    analysis_result: str          # 分析结果
    report: str                   # 最终报告
    next_node: str                # 下一节点路由标识
    missing_info: str             # 缺失信息提示


from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
import os


# 1. 规划节点 - 动态生成任务列表
def planner_node(state: AgentState):
    logger.info(f"🚀 进入规划节点")

    # LLM任务规划提示词（含模糊检测）
    class PlanResponse(BaseModel):
        plan_type: Literal["query", "analysis", "ambiguous"]
        task_list: List[str] = []
        missing_info: str = ""

    planner_prompt = ChatPromptTemplate.from_template("""
    根据用户输入「{input}」进行任务规划：
    1. 若包含明确指标/时间/维度 → 返回任务计划，plan_type 为 "query" 或 "analysis"，并列出任务步骤
    2. 否则 → 返回缺失信息引导语，plan_type 为 "ambiguous"，并在 missing_info 字段说明需补充内容
    输出格式：请用JSON格式输出, {{"plan_type":"query", "task_list": ["query", "analysis", "report"], "missing_info": ""}}""")

    # 执行规划
    llm = ChatOpenAI(
        model="qwen-plus",  # DeepSeek的对话模型
        temperature=0,
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    planner_chain = planner_prompt | llm.with_structured_output(PlanResponse)
    response = planner_chain.invoke({"input": state.get("clarified_input") or state.get("input")})

    logger.debug(f"规划结果: plan_type={response.plan_type}, tasks={len(response.task_list)}")

    # 动态路由设置
    if response.plan_type == "ambiguous":
        return {"missing_info": response.missing_info, "next_node": "clarify"}
    else:
        return {
            "plan_type": response.plan_type,
            "task_list": response.task_list,
            "current_step": 0,
            "next_node": "dispatch"  # 进入任务分发
        }


# 2. 引导节点 - 获取用户补充信息
def clarify_node(state: AgentState):
    missing_info = state.get("missing_info", "请提供更多信息")
    logger.warning(f"⚠️ 需要用户补充信息: {missing_info}")

    # 交互式获取用户输入
    while True:
        try:
            user_input = input(">>> 请补充信息: ")
            if not user_input.strip():
                raise ValueError("输入不能为空")
            break
        except Exception as e:
            logger.error(f"输入错误: {str(e)}")

    logger.info(f"📝 用户补充内容: {user_input[:30]}...")
    return {
        "clarified_input": f"{state.get('input')} {user_input}".strip(),
        "next_node": "planner"
    }


# 3. 任务分发节点 - 动态路由
def dispatch_node(state: AgentState):
    current_step = state["current_step"]
    total_steps = len(state["task_list"])
    logger.info(f"🔄 任务分发 | 步骤 {current_step+1}/{total_steps}")

    if state["current_step"] >= len(state["task_list"]):
        return {"next_node": "final_step"}  # 任务完成

    current_task = state["task_list"][state["current_step"]]

    logger.debug(f"路由决策: {current_task}")

    # 根据任务描述路由
    if "query" == current_task:
        return {"next_node": "query"}
    elif "analysis" == current_task:
        return {"next_node": "analysis"}
    elif "report" == current_task:
        return {"next_node": "report"}
    else:
        return {"next_node": "fallback"}  # 异常处理


# 4. 查询节点 - 数据获取
def query_node(state: AgentState):

    task_desc = state["task_list"][state["current_step"]]
    logger.info(f"🔍 执行查询任务: {task_desc}")

    # 实际应接入数据库/API
    mock_data = {"销售额": "120万", "用户量": "45万 DAU"}
    result = next((v for k, v in mock_data.items() if k in task_desc), "未找到数据")

    return {
        "query_result": result,
        "current_step": state["current_step"] + 1,  # 步骤索引+1
        "next_node": "dispatch"  # 返回分发节点
    }


# 5. 分析节点 - 数据处理
def analysis_node(state: AgentState):
    task_desc = state["task_list"][state["current_step"]]
    logger.info(f"📊 执行分析任务: {task_desc}")

    # 实际应接入分析库
    analysis = f"基于「{state['query_result']}」的分析：\n"
    analysis += "1. 发现异常波动点\n2. 关键因素相关系数0.85"

    return {
        "analysis_result": analysis,
        "current_step": state["current_step"] + 1,
        "next_node": "dispatch"
    }


# 6. 报告节点 - 结果生成
def report_node(state: AgentState):
    logger.info("📝 生成最终报告")
    report = f"# 分析报告\n## 关键数据\n{state['query_result']}\n"
    report += f"## 深度洞察\n{state['analysis_result']}\n"
    report += "## 建议\n1. 优化产品功能\n2. 调整营销策略"

    return {
        "report": report,
        "current_step": state["current_step"] + 1,
        "next_node": "dispatch"
    }

# 结束节点
def final_node(state: AgentState):
    print("流程终止")
    return {
        "next_node": "final_step"
    }

# 结束节点
def fallback(state: AgentState):
    print("任务异常, 请检查输入或任务描述")
    return {
        "next_node": "fallback"
    }

# 7. 新增可视化函数（调试用）
def visualize_workflow():
    """生成工作流Mermaid图"""
    mermaid = """
    graph TD
        planner[规划节点] -->|需补充| clarify(交互节点)
        planner -->|任务就绪| dispatch[分发节点]
        dispatch -->|查询| query[查询节点]
        dispatch -->|分析| analysis[分析节点]
        dispatch -->|报告| report[报告节点]
        dispatch -->|完成| final_step[结束节点]
        query --> dispatch
        analysis --> dispatch
        report --> dispatch
        clarify --> planner
        final_step --> END([结束])
    """
    logger.debug("工作流结构:\n" + mermaid)
    return mermaid

# 初始化状态图
workflow = StateGraph(AgentState)

# 注册节点
workflow.add_node("planner", planner_node)
workflow.add_node("clarify", clarify_node)
workflow.add_node("dispatch", dispatch_node)
workflow.add_node("query", query_node)
workflow.add_node("analysis", analysis_node)
workflow.add_node("report", report_node)
workflow.add_node("fallback", fallback)
workflow.add_node("final_step", final_node)  # 添加真实节点


# 设置入口
workflow.set_entry_point("planner")

# 条件路由规则
workflow.add_conditional_edges(
    "planner",
    lambda s: s.get("next_node", "dispatch"),
    {"dispatch": "dispatch", "clarify": "clarify"}
)

workflow.add_conditional_edges(
    "dispatch",
    lambda s: s["next_node"],
    {
        "query": "query",
        "analysis": "analysis",
        "report": "report",
        "final_step": "final_step",
        "fallback": "clarify",
    }
)

# 固定流转路径
workflow.add_edge("clarify", "planner")
workflow.add_edge("query", "dispatch")
workflow.add_edge("analysis", "dispatch")
workflow.add_edge("report", "dispatch")
workflow.add_edge("final_step", END)         # 连接到END标识符

# 编译工作流
app = workflow.compile()


def run_workflow(question):
    """交互式工作流执行器"""
    # 获取用户初始输入
    question = input("💬 请输入您的问题: ")
    logger.info(f"🚩 启动工作流 | 初始问题: {question}")

    # 可视化工作流（可选）
    visualize_workflow()

    # 初始化状态
    state = {"input": question}

    # 执行流程
    while "next_node" not in state or state["next_node"] != "final_step":
        state = app.invoke(state)

    # 输出结果
    if state["plan_type"] == "query":
        print(f"📊 查询结果: {state['query_result']}")
    else:
        print(f"📈 分析报告:\n{state['report']}")


# 测试执行
run_workflow("分析Q3销售额下降原因")