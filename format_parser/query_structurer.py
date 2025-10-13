from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import OutputFixingParser
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, model_validator
from typing import List, Optional, Union, Dict, Any
from enum import Enum

# ========== 基础枚举类型定义 ==========
class FilterOperator(str, Enum):
    EQUALS = "equals"
    GT = "gt"
    LT = "lt"
    IN = "in"
    LIKE = "like"
    NOTEQUALS = "notEquals"


class TimeDimensionGranularity(str, Enum):
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    YEAR = "year"


class StatDateName(str, Enum):
    """统计日期的字段名字"""
    CALC_DATE = "calc_date"


class OrderDirection(str, Enum):
    ASC = "asc"
    DESC = "desc"


# ========== 过滤条件模型 ==========
class FilterCondition(BaseModel):
    member: str = Field(..., description="要过滤的字段名，如 t1.c1")
    operator: FilterOperator = Field(..., description="过滤操作符")
    values: List[Any] = Field(..., description="过滤值的列表")


class LogicalFilter(BaseModel):
    """支持嵌套的逻辑过滤条件（and/or组合）"""
    or_: Optional[List[Union[FilterCondition, 'LogicalFilter']]] = Field(None, alias="or")
    and_: Optional[List[Union[FilterCondition, 'LogicalFilter']]] = Field(None, alias="and")

    class Config:
        use_enum_values = True
        extra = "allow"
        validate_by_name = True  # 允许按字段名访问

    def model_dump(self, *args, **kwargs):
        return super().model_dump(*args, by_alias=True, exclude_none=True, **kwargs)


# 自引用模型需手动重建
LogicalFilter.model_rebuild()


# ========== 时间维度模型 ==========
class TimeDimension(BaseModel):
    dimension: StatDateName
    granularity: TimeDimensionGranularity
    dateRange: List[Any]  # 通常为[str, str]格式，如["2025-01-01", "2025-01-31"]


# ========== 计算表达式模型 ==========
class Expression(BaseModel):
    sql: str = Field(..., description="自定义计算逻辑，如'if(num>0, num/plan, -1) as rate'")


# ========== 关联查询模型 ==========
class Join(BaseModel):
    """关联查询模型，支持任意层级的嵌套查询作为关联表"""
    condition: str = Field(..., description="关联条件，如 't1.id = t2.user_id'")
    source: 'Query' = Field(..., description="左表查询（支持任意复杂查询）")
    relationship: str = Field(..., description="关联类型：inner/left/right")
    target: 'Query' = Field(..., description="右表查询（支持任意复杂查询）")


# ========== 核心查询模型（支持多层嵌套） ==========
class Query(BaseModel):
    """
    核心查询模型，支持：
    1. 多层子查询（通过subQuery嵌套）
    2. 关联查询（通过join字段）
    3. 简单查询（可省略subQuery和join）
    """
    alias: str = Field(..., description="查询别名，用于子查询或关联查询的表引用")
    measures: List[str] = Field(..., description="指标列表，如['sum(nbev) as nbev_total']")
    dimensions: List[str] = Field(default_factory=list, description="维度列表，如['region', 'account_code']")
    filters: List[LogicalFilter] = Field(default_factory=list, description="过滤条件列表")
    timeDimensions: List[TimeDimension] = Field(default_factory=list, description="时间维度过滤")

    # 复杂查询相关字段（均为可选）
    joins: Optional[Join] = Field(None, description="关联查询配置，可在任意层级查询中存在")
    subQuery: Optional['Query'] = Field(None, description="子查询，可无限嵌套")

    # 分页与排序
    limit: int = 0
    offset: int = 0
    order: Dict[str, OrderDirection] = Field(default_factory=dict, description="排序配置，如{'nbev_total': 'desc'}")
    total: bool = False  # 是否需要返回总数


# 手动重建自引用模型（关键步骤，确保嵌套生效）
Join.model_rebuild()
Query.model_rebuild()

import os
def generate_query_dsl(user_query: str) -> Query:
    """通过大模型生成符合DSL规范的查询结构"""
    # 初始化LLM
    llm = ChatOpenAI(
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    model="deepseek-ai/DeepSeek-V3.1",
    base_url="https://api.siliconflow.cn/v1/",
    temperature=0.1)

    # 创建解析器（支持自动修复格式错误）
    pydantic_parser = PydanticOutputParser(pydantic_object=Query)
    fixing_parser = OutputFixingParser.from_llm(
        llm=llm,
        parser=pydantic_parser
    )

    # 构建提示模板
    prompt = PromptTemplate(
        template="""你是专业的查询DSL生成器，请根据用户问题生成符合格式的查询结构。
要求：
1. 简单查询可省略subQuery和joins字段
2. 复杂查询通过subQuery实现多层嵌套
3. 关联查询通过joins字段实现，source和target可为任意查询结构
4. 必须严格遵循以下格式要求，不添加任何额外说明：

{format_instructions}

用户问题：{user_query}""",
        input_variables=["user_query"],
        partial_variables={"format_instructions": pydantic_parser.get_format_instructions()}
    )

    # 执行生成链
    chain = prompt | llm | fixing_parser
    return chain.invoke({"user_query": user_query})



# 示例用法
# ========== 示例用法 ==========
if __name__ == "__main__":
    # 复杂查询示例（包含多层子查询和关联）
    user_question = """
    生成一个查询：
    1. 最内层查询（alias: inner_query）：
       - 指标：num_metric0（个险代理人承保NBEV）
       - 维度：account_code, region
       - 过滤：region='深圳'，时间范围2025-01-01至2025-01-31
       - 数据源：sx_pdm_safe_doris_1509_agg_prd_sal_ims_chat_bi_hr_perf_m（别名num表）

    2. 中间层查询（alias: middle_query）：
       - 关联内层查询与规划值表（alias: den表）：
         - 关联条件：num表.account_code = den表.account_code
         - 关联类型：inner join
         - 规划值表过滤：region='深圳'、account_code='AC0021300'、计算类型='YTD'
       - 计算达成率：num_metric0 / den.plan_value（空值处理为-2147483648）
       - 指标：num_metric0, den.plan_value, 达成率(metric0_rate)

    3. 最外层查询（alias: outer_query）：
       - 基于中间层结果，按account_code分组
       - 指标：sum(num_metric0) as total_nbev, avg(metric0_rate) as avg_rate
       - 排序：total_nbev降序
    """

    # 生成DSL
    try:
        query_dsl = generate_query_dsl(user_question)
        print("生成的查询DSL：")
        print(query_dsl.model_dump_json(indent=2, by_alias=True, exclude_none=True))
    except Exception as e:
        print(f"生成失败：{str(e)}")

