import os
from typing import Dict

# ================================
# 1. 配置管理类
# ================================
class SystemConfig:
    """系统配置管理，支持外部注入Neo4j配置"""

    def __init__(self, neo4j_settings: dict = None):
        """初始化配置，优先使用传入的Neo4j设置"""
        self.setup_environment(neo4j_settings)

    def setup_environment(self, neo4j_settings: dict = None):
        """设置环境变量，优先使用传入的配置"""
        try:
            # 直接使用传入的配置
            os.environ["NEO4J_URL"] = neo4j_settings.get("url")
            os.environ["NEO4J_USERNAME"] = neo4j_settings.get("username")
            os.environ["NEO4J_PASSWORD"] = neo4j_settings.get("password")
        except Exception as e:
            print(f"配置读取失败：{e}")
            os.environ.setdefault("NEO4J_URL", "bolt://localhost:7687")
            os.environ.setdefault("NEO4J_USERNAME", "neo4j")
            os.environ.setdefault("NEO4J_PASSWORD", "12345678")

        # HuggingFace配置
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    @property
    def neo4j_config(self) -> Dict[str, str]:
        return {
            "url": os.getenv("NEO4J_URL"),
            "username": os.getenv("NEO4J_USERNAME"),
            "password": os.getenv("NEO4J_PASSWORD")
        }
