from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("DocumentLoaders/Smart_AI_Interview_Platform_Detailed_Documentation.pdf")

docs = loader.load()

print(docs)
print(type(docs))
print(len(docs))
print(type(docs[0]))
print("Page Content:", docs[0].page_content)
print("Metadata:", docs[0].metadata)