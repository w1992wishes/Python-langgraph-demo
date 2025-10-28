import asyncio
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


# 定义ETL作业数据源模型
class ETLDataSource(BaseModel):
    type: str = Field(..., description="数据源类型，如mysql、postgresql、csv、api等")
    connection_info: Dict[str, Any] = Field(..., description="数据源连接信息")
    query: Optional[str] = Field(None, description="数据源查询语句，如果是数据库类型")
    path: Optional[str] = Field(None, description="数据源路径，如果是文件类型")


# 定义ETL作业转换步骤模型
class ETLTransformStep(BaseModel):
    name: str = Field(..., description="转换步骤名称")
    type: str = Field(..., description="转换类型，如filter、map、join、aggregate等")
    config: Dict[str, Any] = Field(..., description="转换配置参数")
    description: Optional[str] = Field(None, description="转换步骤描述")


# 定义ETL作业目标模型
class ETLDestination(BaseModel):
    type: str = Field(..., description="目标类型，如mysql、postgresql、csv、json等")
    connection_info: Dict[str, Any] = Field(..., description="目标连接信息")
    table_name: Optional[str] = Field(None, description="目标表名，如果是数据库类型")
    path: Optional[str] = Field(None, description="目标路径，如果是文件类型")
    mode: str = Field("overwrite", description="写入模式，如overwrite、append等")


# 定义ETL作业模型
class ETLJob(BaseModel):
    name: str = Field(..., description="ETL作业名称")
    description: str = Field(..., description="ETL作业描述")
    data_source: ETLDataSource = Field(..., description="ETL作业数据源")
    transform_steps: List[ETLTransformStep] = Field(default_factory=list, description="ETL作业转换步骤列表")
    destination: ETLDestination = Field(..., description="ETL作业目标")
    schedule: Optional[str] = Field(None, description="ETL作业调度表达式，如cron表达式")
    code: Optional[str] = Field(None, description="生成的ETL代码")


# ETL作业数据存储
class ETLJobStore:
    def __init__(self):
        self.jobs = {}
    
    async def add_job(self, name: str, description: str, data_source: Dict[str, Any],
                     transform_steps: List[Dict[str, Any]], destination: Dict[str, Any],
                     schedule: Optional[str] = None, code: Optional[str] = None) -> Dict[str, Any]:
        """异步添加新ETL作业"""
        # 模拟异步操作
        await asyncio.sleep(0.1)
        
        if name in self.jobs:
            return {"status": "error", "message": f"ETL作业 '{name}' 已存在"}
        
        self.jobs[name] = {
            "name": name,
            "description": description,
            "data_source": data_source,
            "transform_steps": transform_steps,
            "destination": destination,
            "schedule": schedule,
            "code": code
        }
        return {"status": "success", "message": f"ETL作业 '{name}' 已成功创建"}
    
    async def update_job(self, name: str, description: Optional[str] = None,
                        data_source: Optional[Dict[str, Any]] = None,
                        transform_steps: Optional[List[Dict[str, Any]]] = None,
                        destination: Optional[Dict[str, Any]] = None,
                        schedule: Optional[str] = None, code: Optional[str] = None) -> Dict[str, Any]:
        """异步更新现有ETL作业"""
        # 模拟异步操作
        await asyncio.sleep(0.1)
        
        if name not in self.jobs:
            return {"status": "error", "message": f"ETL作业 '{name}' 不存在"}
        
        if description is not None:
            self.jobs[name]["description"] = description
        if data_source is not None:
            self.jobs[name]["data_source"] = data_source
        if transform_steps is not None:
            self.jobs[name]["transform_steps"] = transform_steps
        if destination is not None:
            self.jobs[name]["destination"] = destination
        if schedule is not None:
            self.jobs[name]["schedule"] = schedule
        if code is not None:
            self.jobs[name]["code"] = code
        
        return {"status": "success", "message": f"ETL作业 '{name}' 已成功更新"}
    
    async def delete_job(self, name: str) -> Dict[str, Any]:
        """异步删除ETL作业"""
        # 模拟异步操作
        await asyncio.sleep(0.1)
        
        if name not in self.jobs:
            return {"status": "error", "message": f"ETL作业 '{name}' 不存在"}
        
        del self.jobs[name]
        return {"status": "success", "message": f"ETL作业 '{name}' 已成功删除"}
    
    async def get_job(self, name: str) -> Dict[str, Any]:
        """异步获取单个ETL作业详情"""
        # 模拟异步操作
        await asyncio.sleep(0.1)
        
        if name not in self.jobs:
            return {"status": "error", "message": f"ETL作业 '{name}' 不存在"}
        
        return {"status": "success", "data": self.jobs[name]}
    
    async def list_jobs(self) -> Dict[str, Any]:
        """异步列出所有ETL作业"""
        # 模拟异步操作
        await asyncio.sleep(0.1)
        return {"status": "success", "data": list(self.jobs.values())}
    
    async def generate_etl_code(self, name: str, language: str = "python") -> Dict[str, Any]:
        """根据ETL作业配置生成代码"""
        # 模拟异步操作
        await asyncio.sleep(0.1)
        
        if name not in self.jobs:
            return {"status": "error", "message": f"ETL作业 '{name}' 不存在"}
        
        job = self.jobs[name]
        
        # 这里简化处理，实际应该根据不同的数据源、转换步骤和目标生成相应的代码
        if language == "python":
            generated_code = f"""
# ETL Job: {job['name']}
# Description: {job['description']}

import pandas as pd

# 1. 数据提取
# 从 {job['data_source']['type']} 提取数据
"""
            
            # 根据数据源类型添加不同的提取代码
            if job['data_source']['type'] == 'mysql':
                generated_code += f"""
import mysql.connector

# 数据库连接信息
connection_info = {job['data_source']['connection_info']}

# 连接数据库
conn = mysql.connector.connect(**connection_info)

# 执行查询
query = "{job['data_source'].get('query', 'SELECT * FROM your_table')}"
df = pd.read_sql(query, conn)

# 关闭连接
conn.close()
"""
            elif job['data_source']['type'] == 'csv':
                generated_code += f"""
# 从CSV文件读取数据
file_path = "{job['data_source'].get('path', 'data.csv')}"
df = pd.read_csv(file_path)
"""
            else:
                generated_code += f"""
# 从 {job['data_source']['type']} 读取数据的代码
# 连接信息: {job['data_source']['connection_info']}
df = pd.DataFrame()  # 示例数据框
"""
            
            # 2. 添加转换步骤
            generated_code += "\n# 2. 数据转换\n"
            for i, step in enumerate(job['transform_steps'], 1):
                generated_code += f"# 转换步骤 {i}: {step['name']} ({step['type']})\n"
                generated_code += f"# 配置: {step['config']}\n"
                
                # 根据转换类型添加不同的转换代码
                if step['type'] == 'filter':
                    conditions = step['config'].get('conditions', [])
                    for condition in conditions:
                        column = condition.get('column')
                        operator = condition.get('operator')
                        value = condition.get('value')
                        if column and operator and value is not None:
                            generated_code += f"df = df[df['{column}'] {operator} {value}]\n"
                elif step['type'] == 'map':
                    mappings = step['config'].get('mappings', {})
                    for column, function in mappings.items():
                        generated_code += f"df['{column}'] = df['{column}'].{function}\n"
                elif step['type'] == 'aggregate':
                    group_by = step['config'].get('group_by', [])
                    aggregations = step['config'].get('aggregations', {})
                    if group_by and aggregations:
                        agg_str = ', '.join([f"'{col}': '{agg}'" for col, agg in aggregations.items()])
                        group_str = "', '".join(group_by)
                        generated_code += f"df = df.groupby(['{group_str}']).agg({{{agg_str}}}).reset_index()\n"
                else:
                    generated_code += f"# 实现 {step['type']} 转换的代码\n"
                generated_code += "\n"
            
            # 3. 添加数据加载
            generated_code += "# 3. 数据加载\n"
            if job['destination']['type'] == 'mysql':
                generated_code += f"""
# 写入MySQL数据库
import mysql.connector
from sqlalchemy import create_engine

# 创建数据库引擎
user = "{job['destination']['connection_info'].get('user', 'root')}"
password = "{job['destination']['connection_info'].get('password', '')}"
host = "{job['destination']['connection_info'].get('host', 'localhost')}"
database = "{job['destination']['connection_info'].get('database', 'default_db')}"
table_name = "{job['destination'].get('table_name', 'result_table')}"

engine = create_engine(f'mysql+mysqlconnector://{user}:{password}@{host}/{database}')

# 写入数据
df.to_sql(table_name, engine, if_exists='{job['destination']['mode']}', index=False)
"""
            elif job['destination']['type'] == 'csv':
                generated_code += f"""
# 写入CSV文件
output_path = "{job['destination'].get('path', 'output.csv')}"
df.to_csv(output_path, index=False)
"""
            else:
                generated_code += f"""
# 写入 {job['destination']['type']} 的代码
print("数据处理完成")
print(df.head())
"""
            
            # 更新作业中的代码
            self.jobs[name]['code'] = generated_code
            return {"status": "success", "message": "ETL代码生成成功", "data": {"code": generated_code}}
        
        return {"status": "error", "message": f"不支持的语言类型: {language}"}


# 初始化ETL作业存储实例
etl_job_store = ETLJobStore()