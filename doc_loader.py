from langchain_community.document_loaders import BSHTMLLoader, DirectoryLoader, TextLoader, UnstructuredMarkdownLoader,UnstructuredMarkdownLoader

def load_from_dir(folder_path: str, glob='**/*.txt', loader_cls=TextLoader):
    loader = DirectoryLoader(folder_path, glob, loader_cls=loader_cls)
    data = loader.load()
    return data

def load_from_markdown(folder_path: str):
    return load_from_dir(folder_path, '**/*.md', UnstructuredMarkdownLoader)

def load_from_pdf(folder_path: str):
    return load_from_dir(folder_path, '**/*.pdf', UnstructuredMarkdownLoader)

def load_from_url(url: str):
    loader = BSHTMLLoader(url)
    data = loader.load()
    return data

if __name__ == "__main__":
    # html = load_from_url('https://blog.csdn.net/HRG520JN/article/details/136934005')
    # print(html)
    # pdf = load_from_pdf('.\knowledge\1905.11142v1.pdf')
    # print(pdf)
    txt_references='./knowledge'
    txt = load_from_dir(txt_references)
    print('Text', txt)
    md = load_from_dir(txt_references, '**/*.md', UnstructuredMarkdownLoader)
    print('Markdown', md)