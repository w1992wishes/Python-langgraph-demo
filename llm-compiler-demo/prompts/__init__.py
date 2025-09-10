from langchain import hub
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)


def get_llm_compiler_prompt() -> ChatPromptTemplate:
    """获取LLM Compiler提示词（优先从hub拉取，失败则手动构建）"""
    try:
        return hub.pull("wfh/llm-compiler")
    except Exception:
        return build_llm_compiler_prompt()


def get_llm_compiler_joiner_prompt() -> ChatPromptTemplate:
    """获取LLM Compiler Joiner提示词（优先从hub拉取，失败则手动构建）"""
    try:
        prompt = hub.pull("wfh/llm-compiler-joiner")
        return prompt.partial(examples="")
    except Exception:
        return build_llm_compiler_joiner_prompt()


def build_llm_compiler_prompt() -> ChatPromptTemplate:
    """构建LLM Compiler提示词模板（与hub模板结构一致）"""
    # 核心系统指令模板
    core_system_template = """Given a user query, create a plan to solve it with the utmost parallelizability. Each plan should comprise an action from the following {num_tools} types:
{tool_descriptions}
{num_tools}. join(): Collects and combines results from prior actions.
- An LLM agent is called upon invoking join() to either finalize the user query or wait until the plans are executed.
- join should always be the last action in the plan, and will be called in two scenarios:
(a) if the answer can be determined by gathering the outputs from tasks to generate the final response.
(b) if the answer cannot be determined in the planning phase before you execute the plans. Guidelines:
- Each action described above contains input/output types and description.
- You must strictly adhere to the input and output types for each action.
- The action descriptions contain the guidelines. You MUST strictly follow those guidelines when you use the actions.
- Each action in the plan should strictly be one of the above types. Follow the Python conventions for each action.
- Each action MUST have a unique ID, which is strictly increasing.
- Inputs for actions can either be constants or outputs from preceding actions. In the latter case, use the format $id to denote the ID of the previous action whose output will be the input.
- Always call join as the last action in the plan. Say '<END_OF_PLAN>' after you call join
- Ensure the plan maximizes parallelizability.
- Only use the provided action types. If a query cannot be addressed using these, invoke the join action for the next steps.
- Never introduce new actions other than the ones provided."""

    # 格式约束模板
    format_constraint_template = """Remember, ONLY respond with the task list in the correct format! E.g.:
idx. tool(arg_name=args)"""

    # 构建系统消息模板
    core_system_prompt = PromptTemplate(
        template=core_system_template,
        input_variables=["num_tools", "tool_descriptions"],
    )
    core_system_msg = SystemMessagePromptTemplate(prompt=core_system_prompt)

    format_constraint_prompt = PromptTemplate(
        template=format_constraint_template,
        input_variables=[],
    )
    format_constraint_msg = SystemMessagePromptTemplate(prompt=format_constraint_prompt)

    # 组合模板
    return ChatPromptTemplate.from_messages([
        core_system_msg,
        MessagesPlaceholder(variable_name="messages"),
        format_constraint_msg
    ])


def build_llm_compiler_joiner_prompt() -> ChatPromptTemplate:
    """构建LLM Compiler Joiner提示词模板（与hub模板结构一致）"""
    # 核心系统指令模板
    core_system_template = """Solve a question answering task. Here are some guidelines:
 - In the Assistant Scratchpad, you will be given results of a plan you have executed to answer the user's question.
 - Thought needs to reason about the question based on the Observations in 1-2 sentences.
 - Ignore irrelevant action results.
 - If the required information is present, give a concise but complete and helpful answer to the user's question.
 - If you are unable to give a satisfactory finishing answer, replan to get the required information. Respond in the following format:

Thought: <reason about the task results and whether you have sufficient information to answer the question>
Action: <action to take>
Available actions:
 (1) Finish(the final answer to return to the user): returns the answer and finishes the task.
 (2) Replan(the reasoning and other information that will help you plan again. Can be a line of any length): instructs why we must replan
 """

    # 决策约束模板
    decision_constraint_template = """Using the above previous actions, decide whether to replan or finish. If all the required information is present. You may finish. If you have made many attempts to find the information without success, admit so and respond with whatever information you have gathered so the user can work well with you.
"""

    # 构建系统消息模板
    core_system_prompt = PromptTemplate(
        template=core_system_template,
        input_variables=[],
    )
    core_system_msg = SystemMessagePromptTemplate(prompt=core_system_prompt)

    decision_constraint_prompt = PromptTemplate(
        template=decision_constraint_template,
        input_variables=["examples"],
    )
    decision_constraint_msg = SystemMessagePromptTemplate(prompt=decision_constraint_prompt)

    # 组合模板
    return ChatPromptTemplate.from_messages([
        core_system_msg,
        MessagesPlaceholder(variable_name="messages"),
        decision_constraint_msg
    ])