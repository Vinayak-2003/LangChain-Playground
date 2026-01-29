from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import load_prompt

load_dotenv()

# Initialize the HuggingFace LLM endpoint
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-0528",
    task="text-generation"
)

# Create the ChatHuggingFace model
model = ChatHuggingFace(llm=llm)

# Streamlit UI
st.header("Research Tool")

# Static Prompt - Input prompt from user
# user_input = st.text_input("Enter your prompt: ")


# Dynamic Prompt - Select research paper, explanation style, and length
paper_input = st.selectbox("Select Research Paper Name", ["Attention Is All You Need", 
                            "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", 
                            "GPT-3: Language Models are Few-Shot Learners",
                            "RoBERTa: A Robustly Optimized BERT Pretraining Approach"])

style_input = st.selectbox("Select Explanation Style", ["Beginner-friendly", "Technical", "Code-Oriented", 
                                            "Detailed", "Analogies", "Mathematical"])

input_length = st.selectbox("Select Explaination Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", 
                                                           "Long (Detailed explanation with examples)"])

# Load the prompt template
template = load_prompt("Prompts/template.json")

# Generate the prompt based on user selections
prompt = template.invoke({
    "paper_input": paper_input,
    "style_input": style_input,
    "length_input": input_length
})

if st.button('Summarize'):
    result = model.invoke(prompt)
    st.write(result.content)

"""
the above process is a 2 time invoke process where first we invoke the prompt template to generate 
the final prompt based on user inputs and then we invoke the LLM model with that generated prompt to 
get the final response.

we can use chain to combine these two invocations into a single step.
"""