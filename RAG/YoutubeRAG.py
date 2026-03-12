from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")

################################################################################################################################
#####       INGESTION        #####
################################################################################################################################

# 1. document Ingestion
video_id = "Gfr50f6ZBvo"           # only the video id, not the full url
try:
    youtube_transcript_api = YouTubeTranscriptApi()
    fetched_transcripts = youtube_transcript_api.fetch(video_id, languages=["en"])
    """
    FetchedTranscriptSnippet(text='thinking through a problem are you', start=7314.96, duration=2.48)
    FetchedTranscriptSnippet(text='talking about a sheet of paper or the', start=7316.08, duration=3.119)
    FetchedTranscriptSnippet(text='patent pen is there some independent', start=7317.44, duration=4.0)
    FetchedTranscriptSnippet(text='structure yeah i like processes i still', start=7319.199, duration=4.48)
    FetchedTranscriptSnippet(text='like pencil and paper best for working', start=7321.44, duration=4.56)
    FetchedTranscriptSnippet(text="out things but um these days it's just", start=7323.679, duration=4.0)
    """
    
    transcript_list = fetched_transcripts.to_raw_data()
    """
    {'text': 'whole thing', 'start': 7548.96, 'duration': 4.08}
    {'text': "uh why are we humans here you've already", 'start': 7550.079, 'duration': 4.961}
    {'text': 'mentioned that perhaps the universe', 'start': 7553.04, 'duration': 3.52}
    {'text': 'created us', 'start': 7555.04, 'duration': 3.52}
    {'text': "is that why you think we're here", 'start': 7556.56, 'duration': 3.84}
    {'text': 'to understand how the universe yeah i', 'start': 7558.56, 'duration': 3.599}
    """

    transcript = " ".join(chunk["text"] for chunk in transcript_list)
    """
    or arrange this meeting the next day when you're thinking through a problem are you talking about a 
    sheet of paper or the patent pen is there some independent structure yeah i like processes i still 
    like pencil and paper best for working out things but um these days it's just so efficient to read 
    research papers just on the screen i still often print them out actually i still prefer to mark out 
    things and i find it goes into the brain quick better and sticks in the brain better when you're 
    you're still using physical pen and pencil and paper so you take notes with the i have lots of 
    nodes electronic ones and also um whole stacks of notebooks that um that i use at home yeah on some 
    of these most challenging next steps for example stuff none of us know about that you're working on 
    you're thinking there's some deep thinking required there right like what what is the right problem what
    """

except TranscriptsDisabled:
    print("No transcripts available for this video.")

except Exception as e:
    print(f"An error occurred: {e}")

# 2. text splitting
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript])
print(len(chunks))  # number of chunks created - 168

# 3. Embedding
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# 4. Vector Store Creation
vector_store = FAISS.from_documents(documents=chunks, embedding=embeddings)

# print(vector_store.index_to_docstore_id)
"""
{0: '00c29307-d739-4327-a20d-7f789a738627', 1: '418f41d0-6e7d-4546-8c8e-e7e0a8ad6027', 2: '99990e8c-25e1-4e03-b2ec-908189e14af0', 3: '17a5eae0-e66b-4321-8035-068cb1c7e449', 4: 'dd6107fd-40ec-46c7-9149-f20ec9457f39', 5: 'f9b15351-5195-4220-8565-539cfa9d339f', 6: 'e4f2569a-af3f-4538-8198-3fedfd256206', 7: 'afcbf195-c11e-4815-b6c9-ab1bc17a34fc', 8: 'd4f29ea9-72b0-4d23-9f75-5669f9430b96', 9: 'e9868618-ca5e-43fd-af79-d8c65af40356', 10: '8be74139-31d6-4d15-bded-e95f5dfa2826', 11: '781f1857-b35c-4657-90b1-cdb10f5efbfd', 12: 'a32f157f-d328-4871-86b3-c2d61f1db484' ...................... .. 167: 'd9c8b1e7-5a0c-4c3e-9f1b-2a0c8e7f1a3e'}
"""


################################################################################################################################
#####      RETRIEVAL        #####
################################################################################################################################

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

output = retriever.invoke("What is deepmind?")
"""
[Document(id='2190cb5d-cc17-4052-8401-c95f36b75965', metadata={}, page_content='i used to discuss um uh 
uh what were the sort of founding tenets of deep mind and it was very various things one was um algorithmic 
advances so deep learning you know jeff hinton and cohen just had just sort of invented that in academia 
but no one in industry knew about it uh we love reinforcement learning we thought that could be scaled up 
but also understanding about the human brain had advanced um quite a lot uh in the decade prior with 
fmri machines and other things so we could get some good hints about architectures and algorithms and 
and sort of um representations maybe that the brain uses so as at a systems level not at a implementation 
level um and then the other big things were compute and gpus right so we could see a compute was going to 
be really useful and it got to a place where it became commoditized mostly through the games industry 
and and that could be taken advantage of and then the final thing was also mathematical and theoretical 
definitions of intelligence so'), Document(id='32dba65b-60f4-47af-9629-b593b995f7fb', metadata={}, 
page_content="used of ai is in deep mind from the beginning which is using games as ............... 
"""

################################################################################################################################
#####      AUGUMENTATION        #####
################################################################################################################################

prompt = PromptTemplate(
    template="""
    You are a helpful youtube assistant
    Answer ONLY from the provided context.
    If the context is insufficient just say you don't know.

    Context: {context}
    Question: {question}
    """,
    input_variables=["context", "question"]
)

question = "Is the topic of aliens discussed in the video? If yes then what?"
retrieved_docs = retriever.invoke(question)

context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

final_prompt = prompt.invoke({
    "context": context_text,
    "question": question
})
print(final_prompt)



################################################################################################################################
#####      GENERATION        #####
################################################################################################################################

llm = ChatGoogleGenerativeAI(model="gemini-pro")
answer = llm.invoke(final_prompt)
print(answer.content)