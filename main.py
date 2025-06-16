# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from typing import List, Dict, Any, Optional, Tuple
from flask import Flask, request, jsonify
import threading

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

    def __init__(self,llm,neo4j_settings: dict = None):
        # 创建配置管理器并注入Neo4j设置
        self.config = SystemConfig(neo4j_settings)
        self.db_manager = DatabaseManager(self.config)
        self.kb_manager = KnowledgeBaseManager(self.config, self.db_manager, llm)
        self.query_processor = QueryProcessor(self.db_manager, llm)

        print(f"软件工程课程小助手初始化完成")

    def update_knowledge_base(self, paths: List[str]) -> Dict[str, bool]:
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
# 使用示例
# ================================
# Press the green button in the gutter to run the script.
llm_api = LlmApi()
print("=== 使用统一的LlmApi API ===")
# 全局大模型实例
app = Flask(__name__)
Neo4jSetting = {
    "url": "bolt://localhost:7687",
    "username": "neo4j",
    "password": "12345678"
}
# 初始化时传入配置
assistant = SoftwareEngineeringAssistant(llm_api, Neo4jSetting)
# 检查系统状态
status = assistant.get_system_status()
print("系统状态:", status)

@app.route('/ask', methods=['POST'])
def handle_ask():
    """处理用户提问的API端点"""
    data = request.json
    question = data.get('question', '')
    chat_history = data.get('chat_history', [])
    print(question,"|",chat_history)
    if not question:
        return jsonify({"error": "Question is required"}), 400

    try:
        response = assistant.ask(question, chat_history)
        print(response, "/n", chat_history)
        return jsonify({"回答": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/status', methods=['GET'])
def handle_status():
    """获取系统状态的API端点"""
    try:
        status = assistant.get_system_status()
        return jsonify(status)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/update', methods=['GET'])
def handle_update():
    """更新知识库的API端点"""
    results = assistant.update_knowledge_base(["course_data/"])
    print(results)
    return jsonify(results)


def start_flask_app():
    """启动Flask应用"""
    app.run(host='localhost', port=9000, threaded=True)
    print("Web服务已启动，监听端口 9000")


if __name__ == "__main__":

    # 更新知识库 注意设定目录
    results = assistant.update_knowledge_base(["course_data/"])
    # # 提问示例
    # response = assistant.ask("软件工程的定义是什么？")
    # print("回答:", response)
    # response = assistant.ask("什么是Multi-Head Attention？")
    # print("回答:", response)
    # response = assistant.ask("作业1的智能仓储机器人系统状态图如何构建？")
    # print("回答:", response)
    # response = assistant.ask("请详细的介绍Waterfall Model模型。")
    # print("回答:", response)
    # response = assistant.ask("练习.json中的第3题答案是什么？为什么?")
    # print("回答:", response)
    # 启动Web服务（在新线程中）
    threading.Thread(target=start_flask_app, daemon=True).start()

    # 保持主线程运行
    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("\n服务已停止")
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
