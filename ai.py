import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_groq import ChatGroq

# Load API Key
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize Groq LLM
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama3-70b-8192"
)

# Define prompt template with more tones
template = """
You are a text rewriter. Rewrite the given content in the following tones:
1. Formal
2. Casual
3. Persuasive
4. Scientist
5. News Reporter

Content: "{text}"

Return the output clearly separated by tone labels (e.g., 'Formal:', 'Casual:', etc.).
"""

prompt = PromptTemplate(
    input_variables=["text"],
    template=template
)

# Build the chain
tone_chain = LLMChain(llm=llm, prompt=prompt)

# Example input
input_text = "Artificial Intelligence is changing the job market rapidly."

# Run chain
result = tone_chain.run({"text": input_text})

print(result)
