from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path="DocumentLoaders\IoT_hierarchical_data_jamshedpur.csv")

docs = loader.load()

print(len(docs))
print(docs[0])