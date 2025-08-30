from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

# Load API key
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize Groq LLM (LLaMA3 model)
llm = ChatGroq(
    temperature=0,
    groq_api_key=groq_api_key,
    model_name="llama3-70b-8192"
)

# Prompt template for educational simplification
simplify_prompt = PromptTemplate(
    input_variables=["fact"],
    template="""
You are tasked with simplifying the following fact for three different audiences:

Fact: "{fact}"

Output clearly:

For a 5-year-old: <very simple explanation with easy words>
For a high schooler: <scientifically accurate but age-appropriate explanation>
For a researcher: <detailed technical explanation with domain terms>
"""
)

# Create LLMChain
simplify_chain = LLMChain(llm=llm, prompt=simplify_prompt)

# Example input
fact = "Photosynthesis is the process by which plants convert sunlight into chemical energy."

# Run chain
result = simplify_chain.run(fact=fact)
print(result)
