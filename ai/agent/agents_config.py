agents_config = [
    {
        "id": "metric_query_agent",
        "name": "指标查询Agent",
        "capabilities": ["指标数据查询"],
        "parameters": "自然语言查询（需包含：地区、时间范围、指标类型；如「近期广东分公司监管全量案件数量（各统计口径）及环比值」）",
        "examples": "查询近期广东分公司监管全量案件数量（各统计口径）及环比值"
    },
    {
        "id": "alert_analysis_agent",
        "name": "预警分析Agent",
        "capabilities": ["指标数据分析", "风险预警"],
        "parameters": '{"data": [前序指标查询结果（如{task1_result}）]}',  # 引用前序任务结果
        "examples": '{"data": ["{task1_result}"]}'
    },
    {
        "id": "report_generation_agent",
        "name": "报告生成Agent",
        "capabilities": ["报告整理", "图表展示"],
        "parameters": "基于分析结论的报告需求（如「整合{task2_result}和{task3_result}（若存在）生成监管案件分析报告」）",
        "examples": "基于{task2_result}和{task3_result}（若存在）生成广东分公司监管案件分析报告"
    }
]