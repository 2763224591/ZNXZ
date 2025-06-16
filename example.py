# 1. 修复Pydantic导入
from pydantic import BaseModel, Field

# 2. 修复文档加载器导入
from langchain_community.document_loaders import WikipediaLoader, PyPDFLoader

# 3. 修复Neo4jGraph导入
try:
    from langchain_neo4j import Neo4jGraph
except ImportError:
    from langchain_community.graphs import Neo4jGraph

# 4. 其他标准导入
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
    ConfigurableField
)
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from typing import Tuple, List, Optional
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
import os
from langchain.text_splitter import TokenTextSplitter
from langchain_experimental.graph_transformers import LLMGraphTransformer
from neo4j import GraphDatabase
from yfiles_jupyter_graphs import GraphWidget
from langchain_community.vectorstores import Neo4jVector
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_neo4j.vectorstores.neo4j_vector import remove_lucene_chars

try:
    import google.colab
    from google.colab import output

    output.enable_custom_widget_manager()
except:
    pass

# 环境变量配置
os.environ["GOOGLE_API_KEY"] = "AIzaSyBqYQIt5quyBnQ6RXWp9cR1btoRFd3JdBo"
os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "12345678"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# 初始化Neo4j连接
graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD")
)

# 加载PDF文档
pdf_path = "course_data/Attention.pdf"
loader = PyPDFLoader(pdf_path)
raw_documents = loader.load()

# 文档分块
text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=24)
documents = text_splitter.split_documents(raw_documents[:3])

# 初始化Gemini模型
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0
)

# 构建知识图谱
llm_transformer = LLMGraphTransformer(llm=llm)
graph_documents = llm_transformer.convert_to_graph_documents(documents)
graph.add_graph_documents(
    graph_documents,
    baseEntityLabel=True,
    include_source=True
)

# 可视化图谱函数
default_cypher = "MATCH (s)-[r:!MENTIONS]->(t) RETURN s,r,t LIMIT 50"


def showGraph(cypher: str = default_cypher):
    driver = GraphDatabase.driver(
        uri=os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"])
    )
    session = driver.session()
    widget = GraphWidget(graph=session.run(cypher).graph())
    widget.node_label_mapping = 'id'
    return widget


# 初始化免费嵌入模型
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 创建向量索引 - 使用简化的配置避免复杂查询问题
vector_index = Neo4jVector.from_existing_graph(
    embedding=embeddings,
    search_type="vector",  # 先使用简单的向量搜索，避免混合搜索的复杂性
    node_label="Document",
    text_node_properties=["text"],
    embedding_node_property="embedding",
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
)

# 创建全文索引
graph.query(
    "CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]")


# 实体提取模型
class Entities(BaseModel):
    names: List[str] = Field(
        ...,
        description="All the person, organization, or business entities that appear in the text",
    )


# 实体提取链
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are extracting organization and person entities from the text."),
    ("human", "Use the given format to extract information from the following input: {question}"),
])
entity_chain = prompt | llm.with_structured_output(Entities)


# 生成全文查询
def generate_full_text_query(input: str) -> str:
    full_text_query = ""
    words = [el for el in remove_lucene_chars(input).split() if el]
    for word in words[:-1]:
        full_text_query += f" {word}~2 AND"
    full_text_query += f" {words[-1]}~2"
    return full_text_query.strip()


# 结构化检索器 - 修复CALL子查询的弃用警告
def structured_retriever(question: str) -> str:
    result = ""
    entities = entity_chain.invoke({"question": question})
    for entity in entities.names:
        # 修复CALL子查询语法，添加变量作用域
        response = graph.query(
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
            {"query": generate_full_text_query(entity)},
        )
        result += "\n".join([el['output'] for el in response])
    return result


# 混合检索器 - 添加错误处理
def retriever(question: str):
    try:
        structured_data = structured_retriever(question)
    except Exception as e:
        print(f"结构化检索出错: {e}")
        structured_data = ""

    try:
        unstructured_data = [el.page_content for el in vector_index.similarity_search(question)]
    except Exception as e:
        print(f"向量检索出错: {e}")
        # 如果向量检索失败，尝试直接从图数据库获取文档
        try:
            docs = graph.query(
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


# 问题浓缩模板
_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question,
in its original language.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)


# 格式化聊天历史
def _format_chat_history(chat_history: List[Tuple[str, str]]) -> List:
    buffer = []
    for human, ai in chat_history:
        buffer.append(HumanMessage(content=human))
        buffer.append(AIMessage(content=ai))
    return buffer


# 搜索查询分支
_search_query = RunnableBranch(
    (
        RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
            run_name="HasChatHistoryCheck"
        ),
        RunnablePassthrough.assign(
            chat_history=lambda x: _format_chat_history(x["chat_history"])
        )
        | CONDENSE_QUESTION_PROMPT
        | llm
        | StrOutputParser(),
    ),
    RunnableLambda(lambda x: x["question"]),
)

# 最终提示模板
template = """Answer the question based only on the following context:
{context}

Question: {question}
Use natural language and be concise.
Answer:"""
prompt = ChatPromptTemplate.from_template(template)

# RAG链
chain = (
        RunnableParallel({
            "context": _search_query | retriever,
            "question": RunnablePassthrough(),
        })
        | prompt
        | llm
        | StrOutputParser()
)

# 在测试查询前添加向量索引检查和创建
print("检查向量索引状态...")

# 检查是否已存在向量索引
try:
    # 尝试创建向量索引（如果不存在）
    index_query = """
    CREATE VECTOR INDEX vector_index IF NOT EXISTS
    FOR (d:Document) ON (d.embedding)
    OPTIONS {indexConfig: {
        `vector.dimensions`: 384,
        `vector.similarity_function`: 'cosine'
    }}
    """
    graph.query(index_query)
    print("向量索引创建/检查完成")
except Exception as e:
    print(f"向量索引创建警告: {e}")

# 等待索引建立完成
import time

time.sleep(2)

# 测试查询
response = chain.invoke({"question": "Which component in the Transformer uses multi-head attention?"})
print("最终答案:", response)