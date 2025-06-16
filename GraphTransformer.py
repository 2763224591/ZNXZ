from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
import json
# ================================
# 2. 图转换器类
# ================================
class GraphTransformer:
    """图转换器"""

    def __init__(self, llm_api):
        self.llm = llm_api
    def _extract_graph_data(self, doc, llm):
        """根据文件类型选择模板提取实体和关系"""
        # 从元数据获取文件路径
        file_path = doc.metadata.get("source")
        print(file_path)
        # 根据文件后缀选择模板
        if file_path.endswith(".json"):
            prompt = f"""请从软件工程练习题json文档中提取实体和关系：
    文档元数据：{doc.metadata}
    文档内容：{doc.page_content}
    返回JSON格式：
    {{
        "entities": [
            {{"name": "实体值","type": "实体类型"}},
            ...
        ],
        "relationships": [
            {{"source": "源实体","target": "目标实体","type": "关系类型"}},
            ...
        ]
    }}
    提取规则
    1. 实体类型：只能使用以下预设类型：
       - `题目库`: 表示题目所属的题库分类
       - `题目编号`: 题目的顺序编号
       - `题目ID`: 题目的唯一标识符
       - `题目类型`: 题目的答题形式
       - `难度等级`: 题目的难易程度
       - `题目内容`: 题目本身的文本描述
       - `题目选项`: 选择题的选项，附带内容
       - `标准答案`: 题目的正确答案
       - `答案解释`: 对答案的解析说明
       - `相关标签`: 题目的关键词标签
    2. 关系类型：只能使用以下预设关系：
       - `编号`: 题目的编号或者ID是某个实体
       - `归属于`: 题目内容归属于题目库
       - `对应答案`: 题目内容对应标准答案
       - `有解释`: 标准答案有解释说明
       - `有关联`: 题目和相关标签知识有关联
       - `属于类型`: 题目属于某种题型
       - `难度为`: 题目被标记为某个难度
       - `选项为`: 题目为选择题，它的选项为某些实体
    3. 格式要求：
       - 只提取文档中明确存在的实体和关系
       - 不要添加任何文档中不存在的信息
       - 实体属性值必须完全复制文档原文
    """
        elif file_path.endswith(".xml"):
            prompt = f"""请从UML流程模型xml文档中提取实体和关系：
    文档元数据：{doc.metadata}
    文档内容：{doc.page_content}
    返回JSON格式：
    {{
        "entities": [
            {{"name": "阶段名称", "type": "阶段ID"}},
            {{"name": "描述内容", "type": "描述"}},
            {{"name": "模型名称", "type": "描述"}},
        ],
        "relationships": [
            {{"source": "源实体", "target": "目标实体", "type": "触发条件"}}
        ]
    }}
        提取规则
    1. 实体类型（只能使用以下预设类型）：
    - `模型名称`: 这些实体和关系总属于哪个模型
    - `阶段名称`: 流程中的步骤名称
    - `阶段ID`: 阶段的唯一标识符
    - `描述`: 阶段的详细说明
    - `主要产出`: 阶段的核心输出物
    

    2. 关系类型（只能使用以下预设关系）：
    - `包含`: 阶段所属的模型 → 阶段ID
    - `有描述`: 阶段ID → 描述
    - `有产出`: 阶段ID → 主要产出
    - `连接`: 源阶段ID → 目标阶段ID
    - `触发条件`: 连接关系 → 条件文本

    3. 格式要求：
    - 实体属性值必须完全复制文档原文
    - 不要添加文档中不存在的信息
    """
        else:
            prompt = f"""请从软件工程文档中提取实体和关系：
    文档元数据：{doc.metadata}
    文档内容：{doc.page_content}
    返回JSON格式：
    {{
        "entities": [
            {{"name": "实体名称", "type": "实体类型"}},
            ...
        ],
        "relationships": [
            {{"source": "源实体", "target": "目标实体", "type": "关系类型"}},
            ...
        ]
    }}

    实体类型为任何与软件工程以及课程学习相关的类型，包括：概念、方法、技术、工具、过程、阶段、角色、文档、标准、模式、原则、作业等；
    关系类型为任何与软件工程以及课程学习相关的类型，包括：包含、依赖、实现、使用、产生、遵循、属于、关联、继承、组成、应用等。
    """
        try:
            response = llm.znxz(prompt)
            # 解析JSON响应
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return data.get('entities', []), data.get('relationships', [])
        except Exception as e:
            print(f"解析LLM响应失败: {e}")
        return [], []
    def convert_to_graph_documents(self, documents):
        """转换文档为图文档格式"""
        graph_documents = []
        print(f"开始转换 {len(documents)} 个文档为知识图谱...")

        for i, doc in enumerate(documents):
            try:
                print(f"正在处理第{i+1}个...")
                # 使用LLM提取结构化信息
                entities, relationships = self._extract_graph_data(doc, self.llm)

                # 创建节点
                nodes = [Node(id=entity["name"], type=entity["type"]) for entity in entities]

                # 创建关系
                relations = []
                for rel in relationships:
                    source_node = Node(id=rel["source"], type="Entity")
                    target_node = Node(id=rel["target"], type="Entity")
                    relations.append(Relationship(
                        source=source_node,
                        target=target_node,
                        type=rel["type"]
                    ))

                # 创建图文档
                graph_doc = GraphDocument(
                    nodes=nodes,
                    relationships=relations,
                    source=doc
                )
                graph_documents.append(graph_doc)

            except Exception as e:
                print(f"处理第{i + 1}个文档时出错: {e}")
                # 创建空的图文档作为回退
                graph_doc = GraphDocument(nodes=[], relationships=[], source=doc)
                graph_documents.append(graph_doc)

        print(f"转换完成，共处理 {len(graph_documents)} 个文档")
        return graph_documents
    def _fallback_conversion(self, documents):
        """回退转换方法"""
        graph_documents = []
        for doc in documents:
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
        self.page_content = source_document.page_content
        self.metadata = source_document.metadata

