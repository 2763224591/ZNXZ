# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from typing import List, Dict, Any, Optional, Tuple
import threading
import time
import os
from fastapi import FastAPI, HTTPException
from starlette.responses import FileResponse
from pydantic import BaseModel

import uvicorn

# 大模型接入
from llm_api import LlmApi
# 数据库类
from DatabaseManager import DatabaseManager
# 配置接入
from SystemConfig import SystemConfig
# 知识库接入
from KnowledgeBaseManager import KnowledgeBaseManager

from QueryProcessor import QueryProcessor


# ================================
# 9. 主系统类
# ================================
class SoftwareEngineeringAssistant:
    """软件工程课程小助手主系统"""

    def __init__(self, llm, neo4j_settings: dict = None):
        # 创建配置管理器并注入Neo4j设置
        self.config = SystemConfig(neo4j_settings)
        self.db_manager = DatabaseManager(self.config)
        self.kb_manager = KnowledgeBaseManager(self.config, self.db_manager, llm)
        self.query_processor = QueryProcessor(self.db_manager, llm)

        print(f"软件工程课程小助手初始化完成")

    def update_knowledge_base(self, paths: List[str]) -> Dict[str, Any]:
        """更新知识库接口"""
        print("开始更新知识库...")
        results = self.kb_manager.sync_knowledge_base(paths)
        return results

    def ask(self, question: str, chat_history: Optional[List[Tuple[str, str]]] = None) -> str:
        """提问接口"""
        return self.query_processor.query(question, chat_history)

    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        try:
            # 检查数据库连接
            result = self.db_manager.graph.query("RETURN 1 as test")
            db_status = "正常" if result else "异常"

            # 统计文档数量
            doc_count = self.db_manager.graph.query("MATCH (d:Document) RETURN count(d) as count")[0]['count']

            # 统计实体数量
            entity_count = self.db_manager.graph.query("MATCH (e:__Entity__) RETURN count(e) as count")[0]['count']

            # 统计文件记录数量
            file_count = self.db_manager.graph.query("MATCH (f:FileRecord) RETURN count(f) as count")[0]['count']

            return {
                "database_status": db_status,
                "document_count": doc_count,
                "entity_count": entity_count,
                "file_count": file_count,
                "course": "软件工程"
            }
        except Exception as e:
            return {
                "database_status": f"错误: {e}",
                "document_count": 0,
                "entity_count": 0,
                "file_count": 0,
                "course": "软件工程"
            }


# ================================
# 初始化与环境配置
# ================================
def get_neo4j_settings():
    """从环境变量获取Neo4j配置"""
    return {
        "url": os.getenv("NEO4J_URL", "bolt://localhost:7687"),
        "username": os.getenv("NEO4J_USERNAME", "neo4j"),
        "password": os.getenv("NEO4J_PASSWORD", "12345678")
    }

# Press the green button in the gutter to run the script.
llm_api = LlmApi()
print("=== 使用统一的LlmApi API ===")

# 从环境变量获取Neo4j配置
Neo4jSetting = get_neo4j_settings()
print(f"Neo4j配置: {Neo4jSetting['url']}")

# 初始化时传入配置
assistant = SoftwareEngineeringAssistant(llm_api, Neo4jSetting)

# 等待数据库连接
max_retries = 30
retry_count = 0
while retry_count < max_retries:
    try:
        status = assistant.get_system_status()
        if "错误" not in status["database_status"]:
            print("数据库连接成功!")
            print("系统状态:", status)
            break
    except Exception as e:
        print(f"等待数据库连接... ({retry_count + 1}/{max_retries})")
        time.sleep(2)
        retry_count += 1

if retry_count >= max_retries:
    print("数据库连接失败，但继续启动服务...")

# ================================
# 任务状态管理
# ================================
# 使用一个字典来全局管理更新任务的状态
# 包含 status: 'idle', 'running', 'finished', 'error'
#      result: 存储任务完成或失败的结果
update_task_status = {
    "status": "idle",
    "result": None
}
# 线程锁，用于确保对 `update_task_status` 的访问是线程安全的
status_lock = threading.Lock()


def run_knowledge_base_update():
    """在后台线程中运行的知识库更新函数"""
    global update_task_status
    try:
        # 调用实际的更新逻辑
        results = assistant.update_knowledge_base(["course_data/"])

        # 线程安全地更新状态
        with status_lock:
            update_task_status["status"] = "finished"
            # TODO 输出应该不匹配
            update_task_status["result"] = results
            print("知识库更新成功完成。")

    except Exception as e:
        # 线程安全地更新状态
        with status_lock:
            update_task_status["status"] = "error"
            update_task_status["result"] = {"error": str(e)}
            print(f"知识库更新时发生错误: {e}")


# ================================
# App配置
# ================================

# 添加请求模型
class AskRequest(BaseModel):
    question: str
    chat_history: Optional[List[Tuple[str, str]]] = []


class StatusResponse(BaseModel):
    database_status: str
    document_count: int
    entity_count: int
    file_count: int
    course: str


class UpdateResponse(BaseModel):
    message: str  # 异步动作，只能做到回复消息


# 新增：更新任务状态的响应模型
class UpdateStatusResponse(BaseModel):
    status: str
    result: Optional[Dict[str, Any]] = None


class AskResponse(BaseModel):
    回答: str


# app创建
app = FastAPI(title="软件工程课程小助手", version="1.0.0")


@app.post("/ask", response_model=AskResponse)
async def handle_ask(request: AskRequest):
    """处理用户提问的API端点"""
    # 检查更新任务状态，如果正在运行则禁止提问
    with status_lock:
        if update_task_status["status"] == "running":
            raise HTTPException(status_code=409, detail="系统正在更新知识库，请稍后再试。")

    if not request.question:
        raise HTTPException(status_code=400, detail="Question is required")

    try:
        response = assistant.ask(request.question, request.chat_history)
        print(response)
        print(request.chat_history)
        return AskResponse(回答=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status", response_model=StatusResponse)
async def handle_status():
    """获取系统状态的API端点"""
    try:
        status = assistant.get_system_status()
        return StatusResponse(**status)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/update", response_model=UpdateResponse)  # 修改为POST，因为此操作会改变服务器状态
async def handle_update():
    """更新知识库的API端点"""
    global update_task_status
    with status_lock:
        # 检查是否已有任务在运行
        if update_task_status["status"] == "running":
            return UpdateResponse(message="更新任务已在进行中，请勿重复发起。")

        # 如果上一次任务已完成或出错，可以重置状态并发起新任务
        print("接收到更新请求，开始后台更新任务。")
        update_task_status["status"] = "running"
        update_task_status["result"] = None

        # 创建并启动后台线程
        update_thread = threading.Thread(target=run_knowledge_base_update)
        update_thread.start()

    return UpdateResponse(message="知识库更新任务已在后台启动。请通过 /update-status 接口查询进度。")

@app.get("/update-status", response_model=UpdateStatusResponse)
async def handle_update_status():
    """获取知识库更新任务状态的API端点"""
    with status_lock:
        # 直接返回当前的全局状态
        # 如果任务完成，可以将状态重置为idle，以便下次触发
        if update_task_status["status"] in ["finished", "error"]:
            response = UpdateStatusResponse(**update_task_status)
            # 重置状态，以便可以发起下一次更新
            update_task_status["status"] = "idle"
            update_task_status["result"] = None
            return response

        return UpdateStatusResponse(**update_task_status)


@app.get("/web", response_class=FileResponse, summary="提供数据可视化主页")
async def serve_visualization_page():
    """
    当用户访问服务器的根URL时，返回 web.html 文件。
    这是正确的服务器部署方式。
    """
    file_name = "web.html"

    # 检查文件是否存在
    if not os.path.exists(file_name):
        raise HTTPException(status_code=404, detail=f"错误: {file_name} 文件未找到。")

    return FileResponse(file_name)

def start_fastapi_app():
    """启动FastAPI应用"""
    # 在Docker环境中绑定到所有接口
    host = "0.0.0.0" if os.getenv("DOCKER_ENV") else "localhost"
    uvicorn.run(app, host=host, port=9000)
    print(f"Web服务已启动，监听 {host}:9000")


if __name__ == "__main__":
    # 更新知识库 注意设定目录
    # results = assistant.update_knowledge_base(["course_data/"])

    # 启动Web服务（在新线程中）
    threading.Thread(target=start_fastapi_app, daemon=True).start()
    print("服务已启动。请通过API接口进行操作。")
    print("  - POST /update: 启动知识库更新")
    print("  - GET /update-status: 查看更新状态")
    print("  - POST /ask: 进行提问")
    print("  - GET /status: 查看系统基本状态")
    print("  - GET /web: 访问可视化界面")

    # 保持主线程运行
    try:
        while True:
            time.sleep(1)  # 降低CPU占用
    except KeyboardInterrupt:
        print("\n服务已停止")