import os
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

def create_embeddings(base_url='http://127.0.0.1:11434', model='nomic-embed-text'):
    return OllamaEmbeddings(base_url=base_url, model=model)

def create_vectorstores(split_documents, embedding, persist_directory='VectorDB'):
    if not os.path.exists(persist_directory):
        return Chroma.from_documents(documents=split_documents, embedding=embedding, persist_directory=persist_directory)
    else:
        return Chroma(persist_directory=persist_directory, embedding_function=embedding)

if __name__ == "__main__":
    from doc_loader import load_from_dir, load_from_pdf
    from doc_split import split_text
    print('正在初始化参考资料...')
    txt = load_from_dir('.\knowledge')
    txt_s = split_text(txt)
    split_documents = txt_s + docs
    print('参考资料初始化完成')

    print('正在向量化资料...')
    embedding = create_embeddings()
    vectorstore = create_vectorstores(split_documents, embedding)
    print('资料向量化完成')

    print('test')
    matches = vectorstore.similarity_search('RQA的角色定位有哪些')
    print(matches)
    print('Done')