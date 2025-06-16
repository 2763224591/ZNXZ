import json
import re
from typing import List,Optional, Tuple
from langchain_core.runnables import (
    RunnableBranch, RunnableLambda, RunnableParallel, RunnablePassthrough
)
from langchain_core.messages import AIMessage, HumanMessage
from langchain_neo4j.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_core.prompts import PromptTemplate
# 数据库类
from DatabaseManager import DatabaseManager

# ================================
# 8. 查询处理类
# ================================
class QueryProcessor:
    """查询处理类"""

    def __init__(self, db_manager: DatabaseManager,llm_api):
        self.db_manager = db_manager
        self.rag_chain = self._build_rag_chain()
        self.llm=llm_api
    def _extract_entities(self, question: str) -> List[str]:
        """提取实体"""
        prompt = f"""请从以下文本中提取实体信息："{question}"

实体类型包括：
    学术人物
    研究机构
    公式概念
    专业术语
    文献记录（书籍，论文，作业，课程，文件，练习题目）
    以JSON格式返回：
{{"names": ["实体1", "实体2", ...]}}

如果没有找到相关实体，返回空列表。"""

        try:
            response = self.llm.znxz(prompt)

            # 尝试解析JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                data = json.loads(json_str)
                return data.get('names', [])
            else:
                # 如果没有找到JSON，尝试简单解析
                lines = response.split('\n')
                entities = []
                for line in lines:
                    if '：' in line or ':' in line:
                        parts = line.split('：' if '：' in line else ':')
                        if len(parts) > 1:
                            entities.extend([e.strip() for e in parts[1].split('、') if e.strip()])
                return entities
        except Exception as e:
            print(f"实体提取出错: {e}")
            return []

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
            entities = self._extract_entities(question)
            for entity in entities:
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
        condense_template = """根据以下对话历史和一个后续问题，请将后续问题重新表述为一个独立的完整问题。如果对话历史和后续问题和关联很小，以后续问题为准：
对话历史：
{chat_history}
后续问题: {question}
独立完整的问题:"""

        condense_prompt = PromptTemplate.from_template(condense_template)

        # 优化后的浓缩问题函数
        def condense_question(inputs: dict):
            if not inputs.get("chat_history"):
                return inputs["question"]

            # 格式化聊天历史
            formatted_history = "\n".join(
                [f"{'用户' if isinstance(msg, HumanMessage) else '助理'}: {msg.content}"
                 for msg in self._format_chat_history(inputs["chat_history"])]
            )
            return self.llm.znxz(condense_prompt.format(
                chat_history=formatted_history,
                question=inputs["question"]
            ))

        # 搜索查询分支
        search_query = RunnableBranch(
            (
                RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
                    run_name="HasChatHistoryCheck"
                ),
                RunnableLambda(condense_question)
            ),
            RunnableLambda(lambda x: x["question"]),
        )

        # 回答生成函数
        def generate_answer(inputs):
            context = inputs["context"]
            question = inputs["question"]

            answer_template = f"""尝试根据以下与软件工程相关的内容回答问题:
{context}

问题: {question}
使用自然语言并保持简洁.专注于软件工程概念和实践.
如果查询到知识，请以“根据数据库资料得出：”开头并跟上回答，并适度补充，补充内容另一起行添加到“补充：”中。
如果内容为空，请说明“未从数据库中得到有用知识”并另起一行自行补充.
回答:"""

            return self.llm.znxz(answer_template)

        # RAG链
        chain = (
                RunnableParallel({
                    "context": search_query | RunnableLambda(self._retriever),
                    "question": RunnablePassthrough(),
                })
                | RunnableLambda(generate_answer)
        )

        return chain

    def query(self, question: str, chat_history: Optional[List[Tuple[str, str]]] = None) -> str:
        """执行查询"""
        input_data = {"question": question}
        if chat_history:
            input_data["chat_history"] = chat_history

        return self.rag_chain.invoke(input_data)

