from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path="DocumentLoaders\pdfs",
    glob="*.pdf",
    loader_cls=PyPDFLoader
)

# docs = loader.load()

# print(len(docs))
# print("page content:", docs[0].page_content)
# print("metadata:", docs[0].metadata)


docs = loader.lazy_load()
for doc in docs:
    print(doc.metadata)
