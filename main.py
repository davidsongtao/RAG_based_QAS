"""
实现基于RAG技术的问答机器人：
    - 装载本地知识库文件，存入向量数据库
    - 加载向量数据库，根据问题查询相关数据
    - 基于本地知识库中查询到的相关数据和问题，构建prompt
    - 使用自定义大语言模型，生成回答
"""
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core._api import LangChainDeprecationWarning
import warnings
from model import ChatGLM3

from langchain_core.prompts import PromptTemplate

# 忽略一些烦人的警告
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)


# DONE 1. 装载本地知识库文件，存入向量数据库
def refresh_local_knowledge_base(file_path=r"./local_knowledge/知识点.txt",
                                 model_name=r"./xiaobu-embedding-v2", db_path=r'./local_knowledge'):
    # 1.1 读取本地知识库文件，分割数据
    doc_loader = UnstructuredFileLoader(file_path)
    doc = doc_loader.load()

    # 1.2 将知识库文件中的数据转换为向量
    doc_spliter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    split_doc = doc_spliter.split_documents(doc)

    # 1.3 实例化向量数据库对象并保存到本地
    embedding = HuggingFaceEmbeddings(model_name=model_name)
    db = FAISS.from_documents(split_doc, embedding)
    db.save_local(db_path)


# DONE 2. 加载向量数据库，根据问题查询相关数据
def get_related_data(query, model_name=r"./xiaobu-embedding-v2", db_path=r'./local_knowledge'):

    # 刷新本地知识库
    # refresh_local_knowledge_base()

    # 加载本地知识库，根据问题查询相关数据
    related_data_list = []
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    db = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
    search_res = db.similarity_search(query=query, k=1)

    # 对查询到的数据进行处理
    for doc in search_res:
        related_data_list.append(doc.page_content.replace("\n\n", "\n"))
    related_data = "\n".join(related_data_list)
    return related_data


# DONE 3. 基于本地知识库中查询到的相关数据和问题，构建prompt
def define_prompt(query):
    related_data = get_related_data(query=query)

    template = """
    请根据以下提供的相关信息，及问题，给出专业及简短的回答，回答中不允许有私自编造的内容，所有回答都用中文答复。
    相关信息：
    {context}
    问题：
    {question}
    """
    input_variables = ['context', 'question']

    prompt = PromptTemplate(template=template, input_variables=input_variables)
    prompt = prompt.format_prompt(context=related_data, question=query)

    return prompt


# DONE 4. 使用自定义大语言模型，生成回答
def get_response(query):

    llm = ChatGLM3()
    prompt = define_prompt(query)
    response = llm.invoke(prompt)
    print(f">>>>> Question: {query}")
    print(f">>>>> Answer: {response}")


if __name__ == '__main__':

    # refresh_local_knowledge_base()
    # print("知识库更新完成！")
    while True:
        query = input(">>>>> 请输入您的问题：")
        get_response(query)
