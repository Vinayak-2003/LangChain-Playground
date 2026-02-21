from langchain_text_splitters import CharacterTextSplitter

text = """
Artificial intelligence is transforming the way we interact with technology in everyday life. From personalized recommendations on streaming platforms to intelligent chatbots that assist with customer support, AI systems are becoming more accurate and efficient. As advancements continue in machine learning and natural language processing, the integration of AI into healthcare, education, and business is expected to grow even further, making digital experiences smarter and more seamless.
"""

splitter = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    separator=" "
)

result = splitter.split_text(text)
print(result)