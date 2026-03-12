from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")
print(os.getenv("GEMINI_API_KEY"))

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

################################################################################################################################
#####      RETRIEVAL        #####
################################################################################################################################

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

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

################################################################################################################################
#####      GENERATION        #####
################################################################################################################################

llm = ChatGoogleGenerativeAI(model="gemini-pro")

################################################################################################################################
#####      BUILDING CHAIN        #####
################################################################################################################################
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()

def format_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)

parallel_chain = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough()
})

question = "Is the topic of aliens discussed in the video? If yes then what?"
main_chain = parallel_chain | prompt | llm | parser
response = main_chain.invoke(question)
print(response)