# plan.py（确保与工作流参数匹配的核心代码）
from settings import Settings
from typing import List, Union
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

# 1. 定义 Step 和 Plan 模型（保持不变）
class Step(BaseModel):
    step: int = Field(description="步骤编号，从1开始")
    description: str = Field(description="步骤的详细描述")

class Plan(BaseModel):
    """Plan to follow in future"""
    steps: List[Step] = Field(
        description="按顺序排列的步骤列表，每个步骤包含step编号和description描述"
    )

# 2. 修复提示模板：使用 {messages} 变量（与工作流传递的参数一致）
# 注意：JSON 中的大括号需用双大括号 {{ }} 转义
planner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """For the given objective, come up with a simple step by step plan.
This plan should involve individual tasks that if executed correctly will yield the correct answer. 
Do not add any superfluous steps. The result of the final step should be the final answer.
Make sure that each step has all the information needed - do not skip steps.

You MUST return your plan in the following JSON format:
{{
  "steps": [
    {{
      "step": 1,
      "description": "第一个步骤的详细描述（例如：确定2024年澳大利亚网球公开赛男子单打冠军是谁）"
    }},
    {{
      "step": 2,
      "description": "第二个步骤的详细描述（例如：查找该冠军的家乡信息）"
    }}
  ]
}}

Ensure each step has both "step" (integer) and "description" (string) fields.
Do not include any other fields or formatting outside of this JSON structure.
最后用中文回答""",
        ),
        # 使用 {messages} 变量（与工作流传递的 {"messages": [...]} 匹配）
        ("placeholder", "{messages}"),
    ]
)

# 3. 计划生成器（保持不变，但依赖上面的模板变量）
planner = planner_prompt | ChatOpenAI(
    model=Settings.LLM_MODEL,
    temperature=Settings.TEMPERATURE,
    api_key=Settings.OPENAI_API_KEY,
    base_url=Settings.OPENAI_BASE_URL,
).with_structured_output(Plan)

# 4. 其他模型（Response/Act/replanner 保持不变，但确保模板变量正确）
class Response(BaseModel):
    """Response to user."""
    response: str = Field(description="直接回复用户的内容")

class Act(BaseModel):
    """Action to perform."""
    action: Union[Response, Plan] = Field(
        description="要执行的操作。如果要回复用户，使用Response；如果需要进一步使用工具，使用Plan。"
    )

# 修复 replanner 模板（确保变量与 state 结构匹配，且 JSON 大括号转义）
replanner_prompt = ChatPromptTemplate.from_template(
    """For the given objective, update the step-by-step plan.
Only keep steps that still need to be done (remove completed steps).
If no steps are left, respond directly to the user.

Your objective was this:
{input}

Your original plan was this (each step is "step编号: 描述"):
{plan}

You have currently done these steps (任务: 结果):
{past_steps}

You MUST return your response in one of the following JSON formats (escape curly braces correctly):

If returning a new plan (only remaining steps):
{{
  "action": {{
    "steps": [
      {{
        "step": 1,
        "description": "剩余步骤1的描述"
      }}
    ]
  }}
}}

If responding directly to the user:
{{
  "action": {{
    "response": "直接回复用户的内容（包含最终答案）"
  }}
}}"""
)

replanner = replanner_prompt | ChatOpenAI(
    model=Settings.LLM_MODEL,
    temperature=Settings.TEMPERATURE,
    api_key=Settings.OPENAI_API_KEY,
    base_url=Settings.OPENAI_BASE_URL,
).with_structured_output(Act)