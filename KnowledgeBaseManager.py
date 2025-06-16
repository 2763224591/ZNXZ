from langchain.text_splitter import TokenTextSplitter
import os
from typing import List, Dict

# 数据库管理类
from DatabaseManager import DatabaseManager
# 配置接入
from SystemConfig import SystemConfig
# 数据处理类
from DataProcessor import DataProcessorFactory
# 图转换器
from GraphTransformer import GraphTransformer


# ================================
# 7. 知识库管理类
# ================================
class KnowledgeBaseManager:
    """知识库管理类"""

    def __init__(self, config: SystemConfig, db_manager: DatabaseManager,llm_api):
        self.config = config
        self.db_manager = db_manager
        self.data_factory = DataProcessorFactory()
        self.text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=24)
        self.graph_transformer = GraphTransformer(llm_api)

    def process_file(self, file_path: str) -> bool:
        """处理单个文件"""
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            return False

        # 获取处理器
        processor = self.data_factory.get_processor(file_path)
        if not processor:
            print(f"不支持的文件类型: {file_path}")
            return False

        try:
            # 计算文件哈希
            file_hash = processor.get_file_hash(file_path)

            # 检查是否需要更新
            if self.db_manager.check_file_exists(file_path, file_hash):
                print(f"文件未变更，跳过: {file_path}")
                return True

            print(f"处理文件: {file_path}")

            # 删除旧数据
            self.db_manager.remove_old_file_data(file_path)

            # 加载文档
            raw_documents = processor.load_documents(file_path)
            print(f"加载文件: {file_path}成功")

            # 文档分块
            if processor.custom_chunking:
                documents = raw_documents
                print(f"跳过通用分块，使用处理器自定义分块: {file_path}")
            else:
                documents = self.text_splitter.split_documents(raw_documents)
                print(f"文件通用分块: {file_path}成功")

            # 添加源文件信息
            for doc in documents:
                doc.metadata['source'] = file_path
                doc.metadata['course'] = '软件工程'

            # 构建知识图谱
            graph_documents = self.graph_transformer.convert_to_graph_documents(documents)

            # 添加到图数据库
            self.db_manager.graph.add_graph_documents(
                graph_documents,
                baseEntityLabel=True,
                include_source=True
            )

            # 更新文件记录
            self.db_manager.add_file_record(file_path, file_hash)

            print(f"文件处理完成: {file_path}")
            return True

        except Exception as e:
            print(f"处理文件失败 {file_path}: {e}")
            return False

    def process_directory(self, directory_path: str) -> Dict[str, bool]:
        """处理目录中的所有文件"""
        results = {}

        if not os.path.exists(directory_path):
            print(f"目录不存在: {directory_path}")
            return results

        for root, dirs, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                results[file_path] = self.process_file(file_path)

        return results

    def check_and_remove_deleted_files(self) -> List[str]:
        """检查并移除已删除文件的知识库数据

        Args:
            base_paths: 基础路径列表，如果为None则检查所有记录的文件

        Returns:
            被删除的文件路径列表
        """
        try:
            # 获取数据库中所有文件记录
            recorded_files = self.db_manager.get_all_file_records()
            deleted_files = []
            for file_path in recorded_files:
                # 检查文件是否仍然存在
                if not os.path.exists(file_path):
                    print(f"检测到已删除文件: {file_path}")
                    # 从知识库中移除相关数据
                    self.db_manager.remove_file_record(file_path)

            if deleted_files:
                print(f"共清理了 {len(deleted_files)} 个已删除文件的数据")
            return deleted_files

        except Exception as e:
            print(f"检查删除文件时出错: {e}")
            return []

    def sync_knowledge_base(self, paths: List[str]) -> Dict[str, any]:
        """同步知识库 - 包括添加新文件、更新修改文件、删除已删除文件

        Args:
            paths: 要同步的路径列表

        Returns:
            同步结果统计
        """
        sync_results = {
            'processed_files': {},
            'deleted_files': [],
            'total_processed': 0,
            'errors': []
        }

        try:
            # 1. 首先检查并清理已删除的文件
            print("=== 开始清理已删除文件 ===")
            deleted_files = self.check_and_remove_deleted_files()
            sync_results['deleted_files'] = deleted_files

            # 2. 处理现有文件（新增和更新）
            print("=== 开始处理文件 ===")
            processed_results = self.update_knowledge_base(paths)
            sync_results['processed_files'] = processed_results
            sync_results['total_processed'] = sum(1 for success in processed_results.values() if success)

            print(f"=== 同步完成 ===")
            print(f"处理文件: {sync_results['total_processed']}/{len(processed_results)}")
            print(f"清理删除文件: {sync_results['deleted_files']}")

        except Exception as e:
            error_msg = f"同步知识库时出错: {e}"
            print(error_msg)
            sync_results['errors'].append(error_msg)
            raise

        return sync_results

    def update_knowledge_base(self, paths: List[str]) -> Dict[str, bool]:
        """更新知识库接口"""
        results = {}

        for path in paths:
            if os.path.isfile(path):
                results[path] = self.process_file(path)
            elif os.path.isdir(path):
                results.update(self.process_directory(path))
            else:
                print(f"路径不存在: {path}")
                results[path] = False

        return results
