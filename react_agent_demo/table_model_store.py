import asyncio
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


# 定义表字段模型
class TableField(BaseModel):
    name: str = Field(..., description="字段名称")
    type: str = Field(..., description="字段类型，如string、number、boolean等")
    description: Optional[str] = Field(None, description="字段描述")
    required: bool = Field(False, description="是否必填字段")


# 定义表模型
class TableModel(BaseModel):
    name: str = Field(..., description="表名称")
    description: str = Field(..., description="表描述")
    fields: List[TableField] = Field(..., description="表字段列表")
    primary_key: Optional[str] = Field(None, description="主键字段名称")


# 表模型数据存储
class TableModelStore:
    def __init__(self):
        self.tables = {}
    
    async def add_table(self, name: str, description: str, fields: List[Dict[str, Any]], 
                       primary_key: Optional[str] = None) -> Dict[str, Any]:
        """异步添加新表模型"""
        # 模拟异步操作
        await asyncio.sleep(0.1)
        
        if name in self.tables:
            return {"status": "error", "message": f"表 '{name}' 已存在"}
        
        # 验证主键是否在字段列表中
        if primary_key:
            if not any(field["name"] == primary_key for field in fields):
                return {"status": "error", "message": f"主键 '{primary_key}' 不在字段列表中"}
        
        # 验证字段名是否唯一
        field_names = [field["name"] for field in fields]
        if len(field_names) != len(set(field_names)):
            return {"status": "error", "message": "字段名称必须唯一"}
        
        self.tables[name] = {
            "name": name,
            "description": description,
            "fields": fields,
            "primary_key": primary_key
        }
        return {"status": "success", "message": f"表 '{name}' 已成功创建"}
    
    async def update_table(self, name: str, description: Optional[str] = None,
                         fields: Optional[List[Dict[str, Any]]] = None,
                         primary_key: Optional[str] = None) -> Dict[str, Any]:
        """异步更新现有表模型"""
        # 模拟异步操作
        await asyncio.sleep(0.1)
        
        if name not in self.tables:
            return {"status": "error", "message": f"表 '{name}' 不存在"}
        
        # 验证更新的字段和主键
        if fields:
            # 验证字段名是否唯一
            field_names = [field["name"] for field in fields]
            if len(field_names) != len(set(field_names)):
                return {"status": "error", "message": "字段名称必须唯一"}
            self.tables[name]["fields"] = fields
        
        if description is not None:
            self.tables[name]["description"] = description
        
        if primary_key is not None:
            # 验证主键是否在字段列表中
            if not any(field["name"] == primary_key for field in (fields or self.tables[name]["fields"])):
                return {"status": "error", "message": f"主键 '{primary_key}' 不在字段列表中"}
            self.tables[name]["primary_key"] = primary_key
        
        return {"status": "success", "message": f"表 '{name}' 已成功更新"}
    
    async def delete_table(self, name: str) -> Dict[str, Any]:
        """异步删除表模型"""
        # 模拟异步操作
        await asyncio.sleep(0.1)
        
        if name not in self.tables:
            return {"status": "error", "message": f"表 '{name}' 不存在"}
        
        del self.tables[name]
        return {"status": "success", "message": f"表 '{name}' 已成功删除"}
    
    async def get_table(self, name: str) -> Dict[str, Any]:
        """异步获取单个表模型详情"""
        # 模拟异步操作
        await asyncio.sleep(0.1)
        
        if name not in self.tables:
            return {"status": "error", "message": f"表 '{name}' 不存在"}
        
        return {"status": "success", "data": self.tables[name]}
    
    async def list_tables(self) -> Dict[str, Any]:
        """异步列出所有表模型"""
        # 模拟异步操作
        await asyncio.sleep(0.1)
        return {"status": "success", "data": list(self.tables.values())}


# 初始化表模型存储实例
table_model_store = TableModelStore()