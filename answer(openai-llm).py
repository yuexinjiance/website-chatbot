import pickle
from langchain.chains import RetrievalQAWithSourcesChain
from langchain import OpenAI
from langchain.cache import InMemoryCache
import langchain

import os
os.environ["OPENAI_API_KEY"] = "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"
# 读取数据向量信息
with open("faiss_store_openai.pkl", "rb") as f:
    VectorStore = pickle.load(f)

# 配置LLM模型
llm=OpenAI(temperature=0,)
langchain.llm_cache = InMemoryCache()
chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=VectorStore.as_retriever())

# 提问并获取答案
result = chain({"question": "请告诉我两个校区图书馆的开放时间？请用中文回答我"}, return_only_outputs=True)

print(result)