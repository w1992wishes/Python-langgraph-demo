import asyncio
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


# 定义指标数据模型
class Metric(BaseModel):
    name: str = Field(..., description="指标名称")
    description: str = Field(..., description="指标描述")
    value: float = Field(..., description="指标数值")
    unit: str = Field(..., description="指标单位")


# 模拟指标数据存储
class MetricsStore:
    def __init__(self):
        self.metrics = {}
    
    async def add_metric(self, name: str, description: str, value: float, unit: str) -> Dict[str, Any]:
        """异步添加新指标"""
        # 模拟异步操作
        await asyncio.sleep(0.1)
        self.metrics[name] = {
            "name": name,
            "description": description,
            "value": value,
            "unit": unit
        }
        return {"status": "success", "message": f"指标 '{name}' 已成功添加"}
    
    async def update_metric(self, name: str, description: Optional[str] = None, 
                     value: Optional[float] = None, unit: Optional[str] = None) -> Dict[str, Any]:
        """异步更新现有指标"""
        # 模拟异步操作
        await asyncio.sleep(0.1)
        if name not in self.metrics:
            return {"status": "error", "message": f"指标 '{name}' 不存在"}
        
        if description is not None:
            self.metrics[name]["description"] = description
        if value is not None:
            self.metrics[name]["value"] = value
        if unit is not None:
            self.metrics[name]["unit"] = unit
        
        return {"status": "success", "message": f"指标 '{name}' 已成功更新"}
    
    async def delete_metric(self, name: str) -> Dict[str, Any]:
        """异步删除指标"""
        # 模拟异步操作
        await asyncio.sleep(0.1)
        if name not in self.metrics:
            return {"status": "error", "message": f"指标 '{name}' 不存在"}
        
        del self.metrics[name]
        return {"status": "success", "message": f"指标 '{name}' 已成功删除"}
    
    async def get_metric(self, name: str) -> Dict[str, Any]:
        """异步获取单个指标详情"""
        # 模拟异步操作
        await asyncio.sleep(0.1)
        if name not in self.metrics:
            return {"status": "error", "message": f"指标 '{name}' 不存在"}
        
        return {"status": "success", "data": self.metrics[name]}
    
    async def list_metrics(self) -> Dict[str, Any]:
        """异步列出所有指标"""
        # 模拟异步操作
        await asyncio.sleep(0.1)
        return {"status": "success", "data": list(self.metrics.values())}


# 初始化指标存储实例
metrics_store = MetricsStore()