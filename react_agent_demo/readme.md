# 智能指标React Agent助手

这是一个使用LangChain框架和Python语言开发的智能指标助手，可以通过自然语言处理用户请求，并调用相应的工具来管理指标数据。

## 功能特性

- **自然语言交互**：接受用户的自然语言输入，理解用户意图
- **指标管理**：支持指标的添加、更新、删除和查询操作
- **对话记忆**：使用ConversationBufferMemory保持对话上下文
- **异步操作**：所有操作都使用异步方式实现，提高性能
- **API服务**：提供RESTful API接口，方便集成到其他系统

## 技术栈

- Python 3.8+
- LangChain
- LangGraph
- FastAPI
- Uvicorn
- OpenAI API (或DeepSeek API)

## 项目结构

```
react_agent_demo/
├── agent_utils.py       # Agent相关的工具和配置
├── api_server.py        # API服务实现
├── main.py              # 交互式命令行入口
├── metrics_store.py     # 指标存储核心功能
├── test_api.py          # API测试脚本
├── requirements.txt     # 项目依赖
└── .env                 # 环境变量配置
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

编辑 `.env` 文件，添加您的API密钥：

```
SILICONFLOW_API_KEY=your_siliconflow_api_key_here
```

### 3. 运行方式

#### 方式一：交互式命令行

```bash
python main.py
```

#### 方式二：启动API服务

```bash
python api_server.py
```

服务启动后，可访问以下地址：
- API文档：http://localhost:8000/docs
- ReDoc文档：http://localhost:8000/redoc
- 健康检查：http://localhost:8000/health

## API接口说明

### 1. 对话接口

**POST /chat**

请求体：
```json
{
  "message": "您的自然语言请求",
  "session_id": "可选的会话ID"
}
```

响应：
```json
{
  "response": "助手的回复内容",
  "session_id": "会话ID",
  "status": "success"
}
```

### 2. 指标管理接口

**GET /metrics**
- 功能：获取所有指标列表

**GET /metrics/{metric_name}**
- 功能：获取单个指标详情
- 参数：metric_name - 指标名称

### 3. 健康检查

**POST /health**
- 功能：检查服务健康状态

## 使用示例

### 命令行交互示例

```
欢迎使用智能指标助手！我可以帮助您管理各种指标。
您可以告诉我添加、更新、删除或查询指标。输入'退出'结束对话。

请输入您的请求: 添加一个新指标，名称是用户活跃度，描述是每日活跃用户比例，值是65.8，单位是%

助手: 指标 '用户活跃度' 已成功添加

请输入您的请求: 查看所有指标

助手: 当前系统中的指标有：
1. 用户活跃度：每日活跃用户比例，值为65.8%

请输入您的请求: 更新用户活跃度的值为70.2%

助手: 指标 '用户活跃度' 已成功更新
```

### API调用示例

使用curl调用对话接口：

```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "添加一个CPU使用率指标，值为75.5%"}'
```

### 测试API脚本

项目提供了一个异步测试脚本，可以全面测试API服务的各个端点：

1. 确保API服务正在运行：
```bash
python api_server.py
```

2. 在另一个终端中运行测试脚本：
```bash
python test_api.py
```

测试脚本会自动测试以下功能：
- 健康检查端点
- 聊天功能（添加、查询指标）
- 列出所有指标
- 获取单个指标详情
- 错误处理（访问不存在的指标）

测试完成后会显示详细的测试结果和统计信息。

## 注意事项

1. 请确保配置正确的API密钥，否则无法正常调用语言模型
2. 在生产环境中部署时，请修改CORS配置，限制允许的来源域名
3. 当前版本使用内存存储指标数据，重启服务后数据会丢失
4. 如需在生产环境中使用，建议添加更完善的错误处理和日志记录