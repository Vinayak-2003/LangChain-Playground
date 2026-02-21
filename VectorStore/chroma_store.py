from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from dotenv import load_dotenv
import os
load_dotenv()

os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")

doc1 = Document(
        page_content="Virat Kohli is one of the most successful and consistent batsmen in IPL history. Known for his aggressive batting style and fitness, he has led the Royal Challengers Bangalore in multiple seasons.",
        metadata={"team": "Royal Challengers Bangalore"}
    )
doc2 = Document(
        page_content="Rohit Sharma is the most successful captain in IPL history, leading Mumbai Indians to five titles. He's known for his calm demeanor and ability to play big innings under pressure.",
        metadata={"team": "Mumbai Indians"}
    )
doc3 = Document(
        page_content="MS Dhoni, famously known as Captain Cool, has led Chennai Super Kings to multiple IPL titles. His finishing skills, wicketkeeping, and leadership are legendary.",
        metadata={"team": "Chennai Super Kings"}
    )
doc4 = Document(
        page_content="Jasprit Bumrah is considered one of the best fast bowlers in T20 cricket. Playing for Mumbai Indians, he is known for his yorkers and death-over expertise.",
        metadata={"team": "Mumbai Indians"}
    )
doc5 = Document(
        page_content="Ravindra Jadeja is a dynamic all-rounder who contributes with both bat and ball. Representing Chennai Super Kings, his quick fielding and match-winning performances make him a key player.",
        metadata={"team": "Chennai Super Kings"}
    )

docs = [doc1, doc2, doc3, doc4, doc5]

vector_store = Chroma(
    embedding_function=GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001"),
    persist_directory="VectorStore/chroma_db",
    collection_name="sample"
)

store = vector_store.add_documents(docs)
"""
['176da374-e0f2-4b51-8700-02b5991af444', 'c67bc3e2-6b67-4a4c-90ff-879e98b089ff', '13d3377f-4c3a-4520-afae-0d21748a71fc', '3d28ac77-e4f2-466e-8642-8533b9e953d9', '92d9d3f1-7d46-4632-a443-1499db725701']
"""

store_values = vector_store.get(include=["embeddings", "documents", "metadatas"])
"""
{'ids': ['c06fb2c0-baec-4550-80df-682c2d030611', '524a4aef-9329-448c-aa26-34bbe665e9fc', '23d22cec-1f1d-48c8-af2e-841abe18d016', '7c76fced-73c7-4fc2-a5a2-88ea2014bf90', '2b60057d-3dbd-4621-9ec4-58490d0e13a1', 'd8e244a2-d13f-4581-bfa8-af5ce1b9f1fb', 'c8696f87-ac7b-459a-bbc9-82f7c8568632', 'b8fb954d-e70a-4296-9f03-02342dfa8f3a', '41c71cc8-aa14-467f-83cd-f4e6c65f8073', '834e5ab6-eff9-4c64-b5dd-807656910a47', '176da374-e0f2-4b51-8700-02b5991af444', 'c67bc3e2-6b67-4a4c-90ff-879e98b089ff', '13d3377f-4c3a-4520-afae-0d21748a71fc', '3d28ac77-e4f2-466e-8642-8533b9e953d9', '92d9d3f1-7d46-4632-a443-1499db725701'], 'embeddings': array([[-0.00996654,  0.02554117,  0.0242223 , ...,  0.01421537,
        -0.01549197, -0.00279169],
       [-0.01977077,  0.01123443,  0.01743519, ...,  0.00972693,
        -0.01764421, -0.00453035],
       [-0.01377287, -0.00339517,  0.01143404, ...,  0.01138755,
        -0.02144302,  0.00401155],
       ...,
       [-0.01377287, -0.00339517,  0.01143404, ...,  0.01138755,
        -0.02144302,  0.00401155],
       [-0.01237376, -0.00252263,  0.00491068, ..., -0.00136671,
         0.00647076, -0.0056542 ],
       [-0.01384098, -0.02732802,  0.0132105 , ...,  0.00964825,
        -0.0091735 , -0.0027933 ]], shape=(15, 3072)), 'documents': ['Virat Kohli is one of the most successful and consistent batsmen in IPL history. Known for his aggressive batting style and fitness, he has led the Royal Challengers Bangalore in multiple seasons.', "Rohit Sharma is the most successful captain in IPL history, leading Mumbai Indians to five titles. He's known for his calm demeanor and ability to play big innings under pressure.", 'MS Dhoni, famously known as Captain Cool, has led Chennai Super Kings to multiple IPL titles. His finishing skills, wicketkeeping, and leadership are legendary.', 'Jasprit Bumrah is considered one of the best fast bowlers in T20 cricket. Playing for Mumbai Indians, he is known for his yorkers and death-over expertise.', 'Ravindra Jadeja is a dynamic all-rounder who contributes with both bat and ball. Representing Chennai Super Kings, his quick fielding and match-winning performances make him a key player.', 'Virat Kohli is one of the most successful and consistent batsmen in IPL history. Known for his aggressive batting style and fitness, he has led the Royal Challengers Bangalore in multiple seasons.', "Rohit Sharma is the most successful captain in IPL history, leading Mumbai Indians to five titles. He's known for his calm demeanor and ability to play big innings under pressure.", 'MS Dhoni, famously known as Captain Cool, has led Chennai Super Kings to multiple IPL titles. His finishing skills, wicketkeeping, and leadership are legendary.', 'Jasprit Bumrah is considered one of the best fast bowlers in T20 cricket. Playing for Mumbai Indians, he is known for his yorkers and death-over expertise.', 'Ravindra Jadeja is a dynamic all-rounder who contributes with both bat and ball. Representing Chennai Super Kings, his quick fielding and match-winning performances make him a key player.', 'Virat Kohli is one of the most successful and consistent batsmen in IPL history. Known for his aggressive batting style and fitness, he has led the Royal Challengers Bangalore in multiple seasons.', "Rohit Sharma is the most successful captain in IPL history, leading Mumbai Indians to five titles. He's known for his calm demeanor and ability to play big innings under pressure.", 'MS Dhoni, famously known as Captain Cool, has led Chennai Super Kings to multiple IPL titles. His finishing skills, wicketkeeping, and leadership are legendary.', 'Jasprit Bumrah is considered one of the best fast bowlers in T20 cricket. Playing for Mumbai Indians, he is known for his yorkers and death-over expertise.', 'Ravindra Jadeja is a dynamic all-rounder who contributes with both bat and ball. Representing Chennai Super Kings, his quick fielding and match-winning performances make him a key player.'], 'uris': None, 'included': ['embeddings', 'documents', 'metadatas'], 'data': None, 'metadatas': [{'team': 'Royal Challengers Bangalore'}, {'team': 'Mumbai Indians'}, {'team': 'Chennai Super Kings'}, {'team': 'Mumbai Indians'}, {'team': 'Chennai Super Kings'}, {'team': 'Royal Challengers Bangalore'}, {'team': 'Mumbai Indians'}, {'team': 'Chennai Super Kings'}, {'team': 'Mumbai Indians'}, {'team': 'Chennai Super Kings'}, {'team': 'Royal Challengers Bangalore'}, {'team': 'Mumbai Indians'}, {'team': 'Chennai Super Kings'}, {'team': 'Mumbai Indians'}, {'team': 'Chennai Super Kings'}]}
"""

search_query = vector_store.similarity_search(
    query="Who is rohit sharma?",           # query
    k=2,                                            # number of search results
)

print(search_query)

search_query_score = vector_store.similarity_search_with_score(
    query="Who among these are a batsmen?",           # query
    k=2,                                            # number of search results
)

print(search_query_score)

# metadata filtering
search_metadata_query = vector_store.similarity_search_with_score(
    query="Who among these are a batsmen?",
    filter={"team": "Mumbai Indians"}
)

print(search_metadata_query)


# ------- update document -------
updated_doc1 = Document(
    page_content="Virat Kohli is one of the most successful and consistent batsmen in IPL history. Known for his aggressive batting style and fitness, he has led the Royal Challengers Bangalore in multiple seasons. He is also a former captain of the Indian national team.",
    metadata={"team": "Royal Challengers Bangalore"}
)
vector_store.update_document(document_id="176da374-e0f2-4b51-8700-02b5991af444", document=updated_doc1)


# ------- delete document -------
vector_store.delete(ids=["176da374-e0f2-4b51-8700-02b5991af444"])

