from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("DocumentLoaders/Smart_AI_Interview_Platform_Detailed_Documentation.pdf")

docs = loader.load()

splitter = CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=0,
    separator=" "
)

result = splitter.split_documents(docs)

print(len(result))
print(result[0].page_content)
