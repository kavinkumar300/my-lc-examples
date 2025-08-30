from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_groq import ChatGroq  # or use OpenAI/Gemini depending on your LLM
import os
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize LLM (Groq's LLaMA3 as example)
llm = ChatGroq(
    temperature=0.9,
    groq_api_key=groq_api_key,
    model_name="llama3-70b-8192"
)

# Prompt template for bias shift
bias_prompt = PromptTemplate(
    input_variables=["fact"],
    template="""                          
You are tasked with reframing the same fact in three different biases/audiences.

Fact: "{fact}"

Output the following versions clearly:

Positive Spin: <positive biased version of the fact>
Negative Spin: <negative biased version of the fact>
Neutral News: <neutral objective news style version of the fact>
"""
)

# Create LLMChain
bias_chain = LLMChain(llm=llm, prompt=bias_prompt)

# Example input
fact = "Social media usage among teenagers increased 40% last year."

# Run chain
result = bias_chain.run(fact=fact)
print(result)
