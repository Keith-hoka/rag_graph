from dotenv import load_dotenv

load_dotenv()  

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
resp = llm.invoke("what is rag?")
print(resp.content)