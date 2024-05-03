import os
from typing import List

from langchain_core.documents import Document
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from doc_loader import load_from_dir, load_from_pdf, load_from_markdown
from doc_split import split_text, split_markdown
from llm_provider import create_llm
from embedding_provider import create_embeddings, create_vectorstores
from langchain.retrievers import ContextualCompressionRetriever
from BCEmbedding.tools.langchain import BCERerank

llm_ollama_url='http://127.0.0.1:11434'
llm_model='qwen:4b'
embed_ollama_url='http://127.0.0.1:11434'
embed_model='nomic-embed-text'
vector_db='VectorDB'
references_folder='./knowledge/'
chunk_size=512
chunk_overlap=64
top_k = 5
prompt_template = """
你是问答任务助手。使用以下检索到的上下文片段,用中文来回答问题。如果你不知道答案，就说你不知道，保持答案简洁。
"""
verbose = True

print('正在初始化参考资料...')
split_documents:List[Document] = []
if not os.path.exists(vector_db):
    txt = load_from_dir(references_folder)
    split_documents.extend(split_text(txt, chunk_size, chunk_overlap))

    md = load_from_markdown(references_folder)
    split_documents.extend(split_markdown(md, chunk_size, chunk_overlap))

    pdf = load_from_dir(references_folder, '**/*.pdf', PyPDFLoader)
    split_documents.extend(split_text(pdf, chunk_size, chunk_overlap))
else:
    print('找到已有数据库，跳过初始化')
print('参考资料初始化完成', split_documents)

print('正在向量化资料...')
embedding = create_embeddings(base_url=embed_ollama_url, model=embed_model)
vectorstore = create_vectorstores(split_documents, embedding, vector_db)
print('资料向量化完成', embed_ollama_url, embed_model)


print('正在初始化语言模型', llm_ollama_url, llm_model)
llm = create_llm(base_url=llm_ollama_url, model=llm_model)
prompt = ChatPromptTemplate.from_template(prompt_template)
retriever=vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": top_k})
# rerank
reranker_args = {'model': './models/rerank', 'top_n': top_k}
reranker = BCERerank(**reranker_args)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=reranker, base_retriever=retriever
)

'''
chain_type: 
- map_rerank: 打分排序
- refine 每一个文档块进行总结，并且逐步汇总成一个总结
- stuff 直接搜索所有文档块，只输出相关文档块，抛弃掉不相关的文档块
'''
qachain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type='stuff', 
    retriever=compression_retriever,
    verbose=verbose,
    chain_type_kwargs={
        "verbose": verbose,
        # "prompt": prompt
    })

if __name__ == "__main__":
    while True:
        query = input('您好，请问有什么能够帮到您的？')
        if query.lower() == '/bye':
            print('感谢您的使用，再见！')
            break
        # 什么是IO零拷贝
        response = qachain({"query": query})
        print(response['result'])
        