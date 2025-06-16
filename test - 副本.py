# 软件工程课程小助手系统
# ================================

import os
import time
import hashlib
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from langchain_core.runnables import (
    RunnableBranch, RunnableLambda, RunnableParallel, RunnablePassthrough,Runnable
)
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.messages import AIMessage, HumanMessage,BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import TokenTextSplitter
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.runnables.config import RunnableConfig
# 文档加载器
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, CSVLoader, UnstructuredWordDocumentLoader
)
# 数据库和向量存储
from langchain_neo4j import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_neo4j.vectorstores.neo4j_vector import remove_lucene_chars
# 大模型接入
from llm_api import LlmApi

# ================================
# 1. 配置管理类
# ================================
class SystemConfig:
    """系统配置管理"""

    def __init__(self, model_type: str = "custom_api"):
        """
        初始化系统配置
        Args:
            model_type: 固定为 "custom_api"
        """
        self.model_type = "custom_api"
        self.setup_environment()

    def setup_environment(self):
        """设置环境变量"""
        # Neo4j配置
        os.environ["NEO4J_URL"] = "bolt://localhost:7687"
        os.environ["NEO4J_USERNAME"] = "neo4j"
        os.environ["NEO4J_PASSWORD"] = "12345678"

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
# ================================
# 2. 大模型封装类
# ================================
class CustomLLMWrapper(Runnable):
    """自定义LLM包装器，兼容LangChain接口"""

    def __init__(self, temperature: float = 0.0):
        self.llm_api = LlmApi()
        self.temperature = temperature

    def invoke(self, input: Any, config: Optional[RunnableConfig] = None) -> Any:
        """兼容LangChain的invoke接口"""
        # 处理不同的输入格式
        if isinstance(input, str):
            prompt = input
        elif isinstance(input, ChatPromptValue):
            # 处理ChatPromptValue类型
            formatted_messages = input.to_messages()
            prompt = self._format_messages(formatted_messages)
        elif isinstance(input, list) and all(isinstance(msg, BaseMessage) for msg in input):
            # 消息列表格式
            prompt = self._format_messages(input)
        else:
            # 其他格式，尝试转换为字符串
            prompt = str(input)

        # 调用API
        response = self.llm_api.znxz(prompt)

        # 返回兼容格式
        return AIMessage(content=response)

    async def ainvoke(self, input: Any, config: Optional[RunnableConfig] = None) -> Any:
        """异步调用 - 可选实现"""
        return self.invoke(input, config)

    def _format_messages(self, messages):
        """格式化消息列表为单个prompt"""
        formatted = []
        for msg in messages:
            if hasattr(msg, 'content'):
                content = msg.content
                if hasattr(msg, 'type'):
                    if msg.type == 'system':
                        formatted.append(f"系统: {content}")
                    elif msg.type == 'human':
                        formatted.append(f"用户: {content}")
                    elif msg.type == 'ai':
                        formatted.append(f"助手: {content}")
                    else:
                        formatted.append(content)
                else:
                    formatted.append(content)
            else:
                formatted.append(str(msg))

        return "\n".join(formatted)

    def with_structured_output(self, schema):
        """支持结构化输出"""
        return StructuredOutputWrapper(self, schema)

    @property
    def InputType(self) -> type:
        """输入类型"""
        return Any

    @property
    def OutputType(self) -> type:
        """输出类型"""
        return AIMessage

class StructuredOutputWrapper(Runnable):
    """结构化输出包装器"""

    def __init__(self, llm, schema):
        self.llm = llm
        self.schema = schema

    def invoke(self, inputs: Dict[str, Any], config=None, **kwargs) -> Any:
        """实现Runnable接口，调用结构化输出"""
        # 构建结构化提示
        if isinstance(inputs, dict) and 'question' in inputs:
            question = inputs['question']
            prompt = f"""请从以下文本中提取实体信息："{question}"

 实体类型包括：
   学术人物（学者、研究者）
   研究机构（大学、实验室、学会）
   公式概念（知识，公式）
   专业术语（首次出现的核心概念）
   文献记录（书籍，论文名称，作业，课程）
   以JSON格式返回：
{{"names": ["实体1", "实体2", ...]}}

如果没有找到相关实体，返回空列表。"""

            response = self.llm.invoke(prompt)
            content = response.content  # 直接访问content属性

            # 尝试解析JSON
            try:
                import json
                import re

                # 提取JSON部分
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    data = json.loads(json_str)
                    return EntityResult(data.get('names', []))
                else:
                    # 如果没有找到JSON，尝试简单解析
                    lines = content.split('\n')
                    entities = []
                    for line in lines:
                        if '：' in line or ':' in line:
                            parts = line.split('：' if '：' in line else ':')
                            if len(parts) > 1:
                                entities.extend([e.strip() for e in parts[1].split('、') if e.strip()])
                    return EntityResult(entities)
            except:
                return EntityResult([])

        return EntityResult([])

    async def ainvoke(self, inputs: Dict[str, Any], config=None, **kwargs) -> Any:
        """异步调用 - 可选实现"""
        return self.invoke(inputs, config, **kwargs)

    def stream(self, inputs: Dict[str, Any], config=None, **kwargs):
        """流式调用 - 可选实现"""
        result = self.invoke(inputs, config, **kwargs)
        yield result

    async def astream(self, inputs: Dict[str, Any], config=None, **kwargs):
        """异步流式调用 - 可选实现"""
        result = await self.ainvoke(inputs, config, **kwargs)
        yield result

    @property
    def InputType(self) -> type:
        """输入类型"""
        return dict

    @property
    def OutputType(self) -> type:
        """输出类型"""
        return EntityResult

class EntityResult:
    """实体结果类"""

    def __init__(self, names):
        self.names = names

class LLMManager:
    """大模型管理类"""

    def __init__(self, config: SystemConfig):
        self.config = config
        self._llm_instances = {}

    def get_llm(self, temperature: float = None):
        """获取大模型实例"""
        if temperature is None:
            temperature = 0

        key = f"custom_api_{temperature}"

        if key not in self._llm_instances:
            self._llm_instances[key] = CustomLLMWrapper(temperature=temperature)

        return self._llm_instances[key]

    def get_graph_transformer(self, model_name: str = None):
        """获取图转换器"""
        # 由于使用自定义API，需要创建兼容的图转换器
        return CustomGraphTransformer(self.get_llm(model_name))

class CustomGraphTransformer:
    """自定义图转换器"""

    def __init__(self, llm):
        self.llm = llm

    def convert_to_graph_documents(self, documents):
        """转换文档为图文档格式"""

        graph_documents = []

        for doc in documents:
            # 构建图转换提示
            prompt = f"""请分析以下软件工程相关文档内容，提取其中的实体和关系：

文档内容：
{doc.page_content}

请识别以下类型的实体和关系：
1. 概念实体：软件工程概念、方法、技术等
2. 关系：概念之间的关联、层次、依赖等

以简洁的方式描述主要的实体和关系。"""

            try:
                response = self.llm.invoke(prompt)
                # 创建简化的图文档
                graph_doc = SimpleGraphDocument(
                    source_document=doc,
                    entities=[],  # 简化处理
                    relationships=[]  # 简化处理
                )
                graph_documents.append(graph_doc)
            except Exception as e:
                print(f"图转换出错: {e}")
                # 创建简单的图文档
                graph_doc = SimpleGraphDocument(
                    source_document=doc,
                    entities=[],
                    relationships=[]
                )
                graph_documents.append(graph_doc)

        return graph_documents


class SimpleGraphDocument:
    """简化的图文档类"""

    def __init__(self, source_document, entities, relationships):
        self.source = source_document
        self.entities = entities
        self.relationships = relationships
        # 保持原有属性以兼容现有代码
        self.page_content = source_document.page_content
        self.metadata = source_document.metadata

# ================================
# 3. 数据处理抽象基类
# ================================
class DataProcessor(ABC):
    """数据处理器抽象基类，支持扩展"""

    @abstractmethod
    def can_process(self, file_path: str) -> bool:
        """判断是否可以处理该文件类型"""
        pass

    @abstractmethod
    def load_documents(self, file_path: str) -> List[Any]:
        """加载文档"""
        pass

    @abstractmethod
    def get_file_hash(self, file_path: str) -> str:
        """获取文件哈希值用于增量更新"""
        pass


# ================================
# 4. 具体数据处理器实现
# ================================
class PDFProcessor(DataProcessor):
    """PDF文档处理器"""

    def can_process(self, file_path: str) -> bool:
        """判断是否可以处理PDF文件"""
        if file_path.lower().endswith('.pdf'):
            print(f"检测到PDF文件: {os.path.basename(file_path)}")
            return True
        else:
            print(f"不支持的文件类型: {os.path.basename(file_path)}")
            return False

    def load_documents(self, file_path: str) -> List[Any]:
        """加载并解析PDF文档内容"""
        print(f"开始处理PDF文档: {os.path.basename(file_path)}")
        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            return documents
            # 结果显示
        except Exception as e:
            print(f"处理失败")
            raise
    def get_file_hash(self, file_path: str) -> str:
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
class TextProcessor(DataProcessor):
    """文本文档处理器"""

    def can_process(self, file_path: str) -> bool:
        """判断是否可以处理文本文件"""
        if file_path.lower().endswith(('.txt', '.md')):
            print(f"检测到文本文件: {os.path.basename(file_path)}")
            return True
        else:
            print(f"不支持的文件类型: {os.path.basename(file_path)}")
            return False

    def load_documents(self, file_path: str) -> List[Any]:
        loader = TextLoader(file_path, encoding='utf-8')
        return loader.load()

    def get_file_hash(self, file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8') as f:
            return hashlib.md5(f.read().encode()).hexdigest()


class WordProcessor(DataProcessor):
    """Word文档处理器"""

    def can_process(self, file_path: str) -> bool:
        return file_path.lower().endswith(('.doc', '.docx'))

    def load_documents(self, file_path: str) -> List[Any]:
        try:
            loader = UnstructuredWordDocumentLoader(file_path)
            return loader.load()

        except ImportError as e:
            # 检查错误信息中是否包含docx
            if "docx" in str(e).lower():
                print("\033[91m[错误] 缺少处理Word文档所需的依赖库: python-docx\033[0m")
                print("""请通过以下命令安装: pip install python-docx"
                        python命令行
                        >>> import nltk
                        >>> nltk.download('punkt')
                        >>> nltk.download('averaged_perceptron_tagger')
                      """)
            else:
                print(f"\033[91m[错误] 导入失败: {e}\033[0m")
            raise

    def get_file_hash(self, file_path: str) -> str:
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()


class CSVProcessor(DataProcessor):
    """CSV文档处理器"""

    def can_process(self, file_path: str) -> bool:
        return file_path.lower().endswith('.csv')

    def load_documents(self, file_path: str) -> List[Any]:
        loader = CSVLoader(file_path)
        return loader.load()

    def get_file_hash(self, file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8') as f:
            return hashlib.md5(f.read().encode()).hexdigest()


# ================================
# 5. 数据处理工厂类
# ================================
class DataProcessorFactory:
    """数据处理器工厂类"""

    def __init__(self):
        self.processors = [
            PDFProcessor(),
            TextProcessor(),
            WordProcessor(),
            CSVProcessor()
        ]

    def get_processor(self, file_path: str) -> Optional[DataProcessor]:
        """根据文件类型获取对应的处理器"""
        for processor in self.processors:
            if processor.can_process(file_path):
                return processor
        return None

    def register_processor(self, processor: DataProcessor):
        """注册新的数据处理器"""
        self.processors.append(processor)


# ================================
# 6. 数据库管理类
# ================================
class DatabaseManager:
    """数据库管理类"""

    def __init__(self, config: SystemConfig):
        self.config = config
        self.graph = Neo4jGraph(**config.neo4j_config)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
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
                `vector.dimensions`: 384,
                `vector.similarity_function`: 'cosine'
            }}
            """
            self.graph.query(index_query)

            # 修复：创建FileRecord节点的属性约束
            self.graph.query(
                "CREATE CONSTRAINT file_record_unique IF NOT EXISTS FOR (f:FileRecord) REQUIRE (f.file_path, f.file_hash) IS UNIQUE"
            )

            print("数据库索引设置完成")
            time.sleep(2)

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


# ================================
# 7. 知识库管理类
# ================================
class KnowledgeBaseManager:
    """知识库管理类"""

    def __init__(self, config: SystemConfig, llm_manager: LLMManager, db_manager: DatabaseManager):
        self.config = config
        self.llm_manager = llm_manager
        self.db_manager = db_manager
        self.data_factory = DataProcessorFactory()
        self.text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=24)

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
            documents = self.text_splitter.split_documents(raw_documents)
            print(f"文件分块: {file_path}成功")
            # 添加源文件信息
            for doc in documents:
                doc.metadata['source'] = file_path
                doc.metadata['course'] = '软件工程'

            # 构建知识图谱
            graph_transformer = self.llm_manager.get_graph_transformer()
            graph_documents = graph_transformer.convert_to_graph_documents(documents)

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


# ================================
# 8. 查询处理类
# ================================
class QueryProcessor:
    """查询处理类"""

    def __init__(self, llm_manager: LLMManager, db_manager: DatabaseManager):
        self.llm_manager = llm_manager
        self.db_manager = db_manager
        self.entity_chain = self._build_entity_chain()
        self.rag_chain = self._build_rag_chain()

    def _build_entity_chain(self):
        """构建实体提取链 - 简化版本"""

        def extract_entities(inputs):
            """直接提取实体的函数"""
            if isinstance(inputs, dict) and 'question' in inputs:
                question = inputs['question']
            else:
                question = str(inputs)

            prompt = f"""请从以下文本中提取实体信息："{question}"

    实体类型包括：
        学术人物（学者、研究者）
        研究机构（大学、实验室、学会）
        公式概念（知识，公式）
        专业术语（首次出现的核心概念）
        文献记录（书籍，论文名称，作业，课程）
        以JSON格式返回：
    {{"names": ["实体1", "实体2", ...]}}

    如果没有找到相关实体，返回空列表。"""

            try:
                llm = self.llm_manager.get_llm()
                response = llm.invoke(prompt)
                content = response.content

                # 尝试解析JSON
                import json
                import re

                # 提取JSON部分
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    data = json.loads(json_str)
                    return EntityResult(data.get('names', []))
                else:
                    # 如果没有找到JSON，尝试简单解析
                    lines = content.split('\n')
                    entities = []
                    for line in lines:
                        if '：' in line or ':' in line:
                            parts = line.split('：' if '：' in line else ':')
                            if len(parts) > 1:
                                entities.extend([e.strip() for e in parts[1].split('、') if e.strip()])
                    return EntityResult(entities)
            except Exception as e:
                print(f"实体提取出错: {e}")
                return EntityResult([])

        return RunnableLambda(extract_entities)

    def _generate_full_text_query(self, input_text: str) -> str:
        """生成全文查询"""
        full_text_query = ""
        words = [el for el in remove_lucene_chars(input_text).split() if el]
        for word in words[:-1]:
            full_text_query += f" {word}~2 AND"
        full_text_query += f" {words[-1]}~2"
        return full_text_query.strip()

    def _structured_retriever(self, question: str) -> str:
        """结构化检索器"""
        result = ""
        try:
            entities = self.entity_chain.invoke({"question": question})
            for entity in entities.names:
                response = self.db_manager.graph.query(
                    """CALL db.index.fulltext.queryNodes('entity', $query, {limit:2})
                    YIELD node, score
                    CALL (node) {
                      WITH node
                      MATCH (node)-[r:!MENTIONS]->(neighbor)
                      RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
                      UNION ALL
                      WITH node
                      MATCH (node)<-[r:!MENTIONS]-(neighbor)
                      RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS output
                    }
                    WITH output
                    RETURN output LIMIT 50
                    """,
                    {"query": self._generate_full_text_query(entity)},
                )
                result += "\n".join([el['output'] for el in response])
        except Exception as e:
            print(f"结构化检索出错: {e}")

        return result

    def _retriever(self, question: str) -> str:
        """混合检索器"""
        # 结构化数据检索
        try:
            structured_data = self._structured_retriever(question)
        except Exception as e:
            print(f"结构化检索出错: {e}")
            structured_data = ""

        # 向量检索
        try:
            vector_index = self.db_manager.get_vector_index()
            unstructured_data = [el.page_content for el in vector_index.similarity_search(question)]
        except Exception as e:
            print(f"向量检索出错: {e}")
            # 备用检索
            try:
                docs = self.db_manager.graph.query(
                    "MATCH (d:Document) WHERE d.text CONTAINS $query RETURN d.text as text LIMIT 3",
                    {"query": question.split()[0] if question.split() else ""}
                )
                unstructured_data = [doc['text'] for doc in docs]
            except Exception as e2:
                print(f"备用检索也失败: {e2}")
                unstructured_data = ["No relevant documents found."]

        final_data = f"""Structured data:
{structured_data}
Unstructured data:
{"#Document ".join(unstructured_data)}
        """
        return final_data

    def _format_chat_history(self, chat_history: List[Tuple[str, str]]) -> List:
        """格式化聊天历史"""
        buffer = []
        for human, ai in chat_history:
            buffer.append(HumanMessage(content=human))
            buffer.append(AIMessage(content=ai))
        return buffer

    def _build_rag_chain(self):
        """构建RAG链"""
        # 问题浓缩模板
        condense_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question,
in its original language.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""

        condense_prompt = PromptTemplate.from_template(condense_template)

        # 搜索查询分支
        search_query = RunnableBranch(
            (
                RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
                    run_name="HasChatHistoryCheck"
                ),
                RunnablePassthrough.assign(
                    chat_history=lambda x: self._format_chat_history(x["chat_history"])
                )
                | condense_prompt
                | self.llm_manager.get_llm()
                | StrOutputParser(),
            ),
            RunnableLambda(lambda x: x["question"]),
        )

        # 回答模板
        answer_template = """尝试根据以下与软件工程相关的内容回答问题:
{context}

问题: {question}
使用自然语言并保持简洁.专注于软件工程概念和实践.如果内容为空，请说明为空并自行补充.
回答:"""

        answer_prompt = ChatPromptTemplate.from_template(answer_template)

        # RAG链
        chain = (
                RunnableParallel({
                    "context": search_query | self._retriever,
                    "question": RunnablePassthrough(),
                })
                | answer_prompt
                | self.llm_manager.get_llm()
                | StrOutputParser()
        )

        return chain

    def query(self, question: str, chat_history: Optional[List[Tuple[str, str]]] = None) -> str:
        """执行查询"""
        input_data = {"question": question}
        if chat_history:
            input_data["chat_history"] = chat_history

        return self.rag_chain.invoke(input_data)


# ================================
# 9. 主系统类
# ================================
class SoftwareEngineeringAssistant:
    """软件工程课程小助手主系统"""

    def __init__(self):
        """
        初始化系统
        Args:
            model_type: 固定为 "custom_api"
            model_name: 忽略此参数，使用deepseek-chat
        """
        self.config = SystemConfig(model_type="custom_api")
        self.llm_manager = LLMManager(self.config)
        self.db_manager = DatabaseManager(self.config)
        self.kb_manager = KnowledgeBaseManager(self.config, self.llm_manager, self.db_manager)
        self.query_processor = QueryProcessor(self.llm_manager, self.db_manager)

        print(f"软件工程课程小助手初始化完成")
        print(f"使用模型: DeepSeek-V3")
        print(f"模型配置: {self.config.model_config}")

    def update_knowledge_base(self, paths: List[str]) -> Dict[str, bool]:
        """更新知识库接口"""
        print("开始更新知识库...")
        results = self.kb_manager.update_knowledge_base(paths)
        success_count = sum(1 for r in results.values() if r)
        total_count = len(results)
        print(f"知识库更新完成: {success_count}/{total_count} 个文件处理成功")
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
                "file_count": file_count
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
# 10. 使用示例
# ================================
if __name__ == "__main__":
    if __name__ == "__main__":
        print("=== 使用统一的DeepSeek-V3 API ===")
        assistant = SoftwareEngineeringAssistant()

        # 检查系统状态
        status = assistant.get_system_status()
        print("系统状态:", status)

        # 更新知识库（示例）
        results = assistant.update_knowledge_base(["course_data/"])

        # 提问示例
        response = assistant.ask("什么是软件工程？")
        print("回答:", response)