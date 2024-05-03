from langchain_community.llms import Ollama


def create_llm(base_url='http://127.0.0.1:11434', model='phi3'):
    return Ollama(base_url=base_url, model=model)

if __name__ == "__main__":
    llm = create_llm()
    response = llm('RQA的角色定位有哪些')
    print(response)