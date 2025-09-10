import re
import time
from concurrent.futures import ThreadPoolExecutor, wait
from typing import Any, Dict, Iterable, List, Optional, Union

from langchain_core.messages import BaseMessage, FunctionMessage, SystemMessage
from langchain_core.runnables import (
    chain as as_runnable, RunnableConfig
)
from typing_extensions import TypedDict

from parsers import Task


def _get_observations(messages: List[BaseMessage]) -> Dict[int, Any]:
    """从消息中提取历史任务执行结果（保留原有逻辑）"""
    results = {}
    for msg in messages[::-1]:
        if isinstance(msg, FunctionMessage):
            results[int(msg.additional_kwargs["idx"])] = msg.content
    return results


class SchedulerInput(TypedDict):
    """调度器输入数据结构（保留原有逻辑）"""
    messages: List[BaseMessage]
    tasks: Iterable[Task]


def _resolve_arg(arg: Union[str, Any], observations: Dict[int, Any]) -> Union[str, Any]:
    """解析参数中的依赖（替换$1/${1}为历史结果）"""
    def replace_match(match):
        idx = int(match.group(1))
        return str(observations.get(idx, match.group(0)))

    if isinstance(arg, str):
        return re.sub(ID_PATTERN, replace_match, arg)
    elif isinstance(arg, list):
        return [_resolve_arg(a, observations) for a in arg]
    else:
        return str(arg)


def _execute_task(task: Task, observations: Dict[int, Any], config: Optional[RunnableConfig] = None) -> str:
    """执行单个任务（保留原有逻辑）"""
    tool_to_use = task["tool"]
    if isinstance(tool_to_use, str):
        return tool_to_use  # join工具无需执行

    args = task["args"]
    # 解析参数（处理依赖）
    try:
        if isinstance(args, str):
            resolved_args = _resolve_arg(args, observations)
        elif isinstance(args, dict):
            resolved_args = {
                key: _resolve_arg(val, observations) for key, val in args.items()
            }
        else:
            resolved_args = args
    except Exception as e:
        return (
            f"ERROR(Failed to call {tool_to_use.name} with args {args}.)"
            f" Args could not be resolved. Error: {repr(e)}"
        )

    # 执行工具
    try:
        return str(tool_to_use.invoke(resolved_args, config))
    except Exception as e:
        return (
            f"ERROR(Failed to call {tool_to_use.name} with args {args}."
            f" Args resolved to {resolved_args}. Error: {repr(e)})"
        )


@as_runnable
def schedule_task(task_inputs: Dict[str, Any], config: Optional[RunnableConfig] = None) -> None:
    """调度单个任务（添加结果到观测字典）"""
    task: Task = task_inputs["task"]
    observations: Dict[int, Any] = task_inputs["observations"]
    try:
        observation = _execute_task(task, observations, config)
    except Exception:
        import traceback
        observation = "".join(traceback.format_exception())
    observations[task["idx"]] = observation


def schedule_pending_task(
    task: Task,
    observations: Dict[int, Any],
    retry_after: float = 0.2,
    config: Optional[RunnableConfig] = None
) -> None:
    """调度待执行任务（等待依赖满足）"""
    while True:
        deps = task["dependencies"]
        # 检查依赖是否全部满足
        if deps and any([dep not in observations for dep in deps]):
            time.sleep(retry_after)
            continue
        # 执行任务
        schedule_task.invoke({"task": task, "observations": observations}, config)
        break


@as_runnable
def schedule_tasks(scheduler_input: SchedulerInput) -> List[FunctionMessage]:
    """Group the tasks into a DAG schedule."""
    # For streaming, we are making a few simplifying assumption:
    # 1. The LLM does not create cyclic dependencies
    # 2. That the LLM will not generate tasks with future deps
    # If this ceases to be a good assumption, you can either
    # adjust to do a proper topological sort (not-stream)
    # or use a more complicated data structure
    tasks = scheduler_input["tasks"]
    args_for_tasks = {}
    messages = scheduler_input["messages"]
    # If we are re-planning, we may have calls that depend on previous
    # plans. Start with those.
    observations = _get_observations(messages)
    task_names = {}
    originals = set(observations)
    # ^^ We assume each task inserts a different key above to
    # avoid race conditions...
    futures = []
    retry_after = 0.25  # Retry every quarter second
    with ThreadPoolExecutor() as executor:
        for task in tasks:
            print(f"---------- [task]{task} --------------")
            deps = task["dependencies"]
            task_names[task["idx"]] = (
                task["tool"] if isinstance(task["tool"], str) else task["tool"].name
            )
            args_for_tasks[task["idx"]] = task["args"]
            if (
                # Depends on other tasks
                deps and (any([dep not in observations for dep in deps]))
            ):
                futures.append(
                    executor.submit(
                        schedule_pending_task, task, observations, retry_after
                    )
                )
            else:
                # No deps or all deps satisfied
                # can schedule now
                schedule_task.invoke(dict(task=task, observations=observations))
                # futures.append(executor.submit(schedule_task.invoke, dict(task=task, observations=observations)))

        # All tasks have been submitted or enqueued
        # Wait for them to complete
        wait(futures)
    # Convert observations to new tool messages to add to the state
    new_observations = {
        k: (task_names[k], args_for_tasks[k], observations[k])
        for k in sorted(observations.keys() - originals)
    }
    tool_messages = [
        FunctionMessage(
            name=name,
            content=str(obs),
            additional_kwargs={"idx": k, "args": task_args},
            tool_call_id=k,
        )
        for k, (name, task_args, obs) in new_observations.items()
    ]
    return tool_messages


@as_runnable
def plan_and_schedule(state: Dict[str, List[BaseMessage]], planner, config: Optional[RunnableConfig] = None) -> Dict[
    str, List[BaseMessage]]:
    """规划+调度一体化（保留原有逻辑，补充输出验证）"""
    messages = state["messages"]
    # 流式获取规划任务
    tasks = planner.stream(messages, config)
    try:
        # 确保任务迭代器可重复使用
        tasks = itertools.chain([next(tasks)], tasks)
    except StopIteration:
        tasks = iter([])
        # 补充：任务为空时添加日志
        print("警告：planner未生成任何任务")

    # 调度任务并获取结果
    scheduled_tasks = schedule_tasks.invoke(
        {
            "messages": messages,
            "tasks": tasks,
        },
        config
    )

    # 验证输出，确保返回标准State结构
    if not isinstance(scheduled_tasks, list):
        scheduled_tasks = [SystemMessage(content=f"schedule_tasks返回非列表结果：{str(scheduled_tasks)}")]

    return {"messages": scheduled_tasks}

# 导入必要模块（避免循环导入）
import itertools
ID_PATTERN = r"\$\{?(\d+)\}?"  # 与parsers保持一致