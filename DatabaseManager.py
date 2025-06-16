from typing import List
# 数据库和向量存储
from langchain_neo4j import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from text2vec import SentenceModel

# 配置接入
from SystemConfig import SystemConfig

# ================================
# 6. 数据库管理类
# ================================
class Text2VecEmbedding:
    def __init__(self, model_name_or_path):
        self.model = SentenceModel(model_name_or_path)

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(text).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(texts)
        return embeddings.tolist()
class DatabaseManager:
    """数据库管理类"""

    def __init__(self, config: SystemConfig):
        self.config = config
        self.graph = Neo4jGraph(**config.neo4j_config)
        self.embeddings = Text2VecEmbedding('models--shibing624--text2vec-base-chinese/')
        self.vector_index = None
        self._setup_database()


    def _setup_database(self):
        """设置数据库索引"""
        try:
            # 创建全文索引
            self.graph.query(
                "CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]"
            )

            # 创建向量索引
            index_query = """
            CREATE VECTOR INDEX vector_index IF NOT EXISTS
            FOR (d:Document) ON (d.embedding)
            OPTIONS {indexConfig: {
                `vector.dimensions`: 768,
                `vector.similarity_function`: 'cosine'
            }}
            """
            self.graph.query(index_query)

            # 修复：创建FileRecord节点的属性约束
            self.graph.query(
                "CREATE CONSTRAINT file_record_unique IF NOT EXISTS FOR (f:FileRecord) REQUIRE (f.file_path, f.file_hash) IS UNIQUE"
            )

            print("数据库索引设置完成")

        except Exception as e:
            print(f"数据库设置警告: {e}")

    def get_vector_index(self):
        """获取向量索引"""
        if self.vector_index is None:
            self.vector_index = Neo4jVector.from_existing_graph(
                embedding=self.embeddings,
                search_type="vector",
                node_label="Document",
                text_node_properties=["text"],
                embedding_node_property="embedding",
                **self.config.neo4j_config
            )
        return self.vector_index

    def check_file_exists(self, file_path: str, file_hash: str) -> bool:
        """检查文件是否已存在于数据库中"""
        try:
            result = self.graph.query(
                """
                MATCH (f:FileRecord) 
                WHERE f.file_path = $file_path AND f.file_hash = $file_hash 
                RETURN f LIMIT 1
                """,
                {"file_path": file_path, "file_hash": file_hash}
            )
            return len(result) > 0
        except Exception as e:
            print(f"检查文件存在性时出错: {e}")
            return False

    def add_file_record(self, file_path: str, file_hash: str):
        """添加文件记录"""
        try:
            self.graph.query(
                """
                MERGE (f:FileRecord {file_path: $file_path})
                SET f.file_hash = $file_hash, 
                    f.updated_at = datetime(),
                    f.created_at = COALESCE(f.created_at, datetime())
                """,
                {"file_path": file_path, "file_hash": file_hash}
            )
        except Exception as e:
            print(f"添加文件记录时出错: {e}")

    def remove_old_file_data(self, file_path: str):
        """删除旧的文件数据"""
        try:
            # 删除相关的Document节点
            self.graph.query(
                """
                MATCH (d:Document) 
                WHERE d.source = $file_path
                DETACH DELETE d
                """,
                {"file_path": file_path}
            )
            print(f"已删除文件 {file_path} 的旧数据")
        except Exception as e:
            print(f"删除旧文件数据时出错: {e}")

    def remove_file_record(self, file_path: str):
        """
        清理指定文件在数据库中的所有记录
        包括 FileRecord 节点和相关的 Document 节点
        """
        try:
            # 先删除相关的 Document 节点
            self.remove_old_file_data(file_path)

            # 删除 FileRecord 节点
            self.graph.query(
                """
                MATCH (f:FileRecord {file_path: $file_path})
                DETACH DELETE f
                """,
                {"file_path": file_path}
            )
            print(f"已删除文件记录: {file_path}")

        except Exception as e:
            print(f"删除文件记录时出错: {e}")

    def get_all_file_records(self) -> List[str]:
        """
        获取数据库中所有文件记录的文件路径
        返回格式: ["path/to/file1", "path/to/file2", ...]
        """
        try:
            result = self.graph.query(
                """
                MATCH (f:FileRecord)
                RETURN f.file_path AS file_path
                """
            )

            # 提取文件路径列表
            file_paths = [record["file_path"] for record in result]
            print(f"获取到 {len(file_paths)} 条文件记录")
            return file_paths
        except Exception as e:
            print(f"获取文件记录时出错: {e}")
            return []
