from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownTextSplitter

def split_text(documents, chunk_size=512, chunk_overlap=64):
    text_spliter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_spliter.split_documents(documents)

def split_markdown(documents, chunk_size=512, chunk_overlap=64):
    text_splitter = MarkdownTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(documents)

if __name__ == "__main__":
    from doc_loader import load_from_dir, load_from_pdf
    txt = load_from_dir('.\knowledge')
    print(txt)
    txt_s = split_text(txt)
    print(txt_s[:2])
