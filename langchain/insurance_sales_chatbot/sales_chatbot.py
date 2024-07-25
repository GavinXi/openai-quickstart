import gradio as gr
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
import os
from langchain.prompts import PromptTemplate


def initialize_sales_bot(vector_store_dir: str = "real_insurance_sale"):
    os.environ["OPENAI_API_KEY"] = "sk-abcd"
    os.environ['OPENAI_BASE_URL'] = 'https://api.xiaoai.plus/v1'
    db = FAISS.load_local(vector_store_dir, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    # 构建提示
    template = """Please answer the question in the context in which it was asked, remembering that you are a 
    good insurance salesman, working for Comfort Pension Insurance Company(安逸养老), and that you would recommend your 
    company's products to other people, especially pension insurance, and that you would give your customers 
    a detailed description of them without revealing yourself to be an AI robot,If the client asks about non-insurance 
    areas, you can politely decline to answer and bring the client back to the insurance-related topic
    Keep the answer as concise as possible. 
    {context}
    Question: {question}
    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    global SALES_BOT
    SALES_BOT = RetrievalQA.from_chain_type(llm,
                                           retriever=db.as_retriever(search_type="similarity_score_threshold",
                                                                     search_kwargs={"score_threshold": 0.8}),
                                            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})
    # 返回向量数据库的检索结果
    SALES_BOT.return_source_documents = True
    return SALES_BOT


def sales_chat(message, history):
    print(f"[message]{message}")
    print(f"[history]{history}")
    # TODO: 从命令行参数中获取
    enable_chat = True

    ans = SALES_BOT({"query": message})
    # 如果检索出结果，或者开了大模型聊天模式
    # 返回 RetrievalQA combine_documents_chain 整合的结果
    if ans["source_documents"] or enable_chat:
        print(f"[result]{ans['result']}")
        print(f"[source_documents]{ans['source_documents']}")
        return ans["result"]
    # 否则输出套路话术
    else:
        return "这个问题暂时无法回答，如果需要更多了解我们的产品欢迎到我们公司咨询。是否有其他问题我可以回答？"


def launch_gradio():
    demo = gr.ChatInterface(
        fn=sales_chat,
        title="保险销售客服",
        # retry_btn=None,
        # undo_btn=None,
        chatbot=gr.Chatbot(height=600),
    )

    demo.launch(share=True, server_name="0.0.0.0")


if __name__ == "__main__":
    # 初始化销售机器人
    initialize_sales_bot()
    # 启动 Gradio 服务
    launch_gradio()
