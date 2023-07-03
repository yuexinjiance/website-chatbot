from langchain import PromptTemplate, HuggingFaceHub
from langchain.chains import RetrievalQAWithSourcesChain
import pickle
import os
os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'


with open("faiss_store_openai.pkl", "rb") as f:
    VectorStore = pickle.load(f)

template = """Question: {question}
Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])
llm = HuggingFaceHub(repo_id="google/flan-t5-xl", model_kwargs={"temperature":0, "max_length":64})
chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, question_prompt=prompt, retriever=VectorStore.as_retriever())
question = "如何进行续借的操作？请用中文回答我"
# 提问并获取答案
result = chain(question, return_only_outputs=True)

print(result)