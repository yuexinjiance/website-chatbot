from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
import pickle


# 配置数据源URL列表
urls = [
    'https://lib.zyufl.edu.cn/2018/0706/c1373a25206/page.htm',
    'https://lib.zyufl.edu.cn/1375/list.htm'
]
#
# 加载URL页面数据
loaders = UnstructuredURLLoader(urls=urls)
data = loaders.load()

# 分割URL页面数据
text_splitter = CharacterTextSplitter(separator='\n',
                                      chunk_size=500,
                                      chunk_overlap=80,
                                      )
docs = text_splitter.split_documents(data)

instructor_embeddings = HuggingFaceInstructEmbeddings(model_name = "hkunlp/instructor-xl", model_kwargs = {"device": "cuda"})
Embedding_store_path = f"embedding_store"

