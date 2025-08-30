import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_groq import ChatGroq

# --- Load API key from .env ---
load_dotenv()

# Debug check
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("❌ GROQ_API_KEY not found. Make sure .env exists and has GROQ_API_KEY=...")

# --- configure Groq LLM ---
llm = ChatGroq(
    temperature=0,
    model="llama3-70b-8192",  # correct param name
    api_key=os.getenv("GROQ_API_KEY")
)

# --- prompt templates ---
scientist_template = PromptTemplate(
    input_variables=["abstract"],
    template=(
        "You are a scientist. Summarize the following research abstract concisely and precisely, "
        "using technical language and emphasizing methods, mechanisms, and implications for future research. "
        "Limit to one or two sentences.\n\nAbstract:\n{abstract}\n\nScientist Summary:"
    )
)

reporter_template = PromptTemplate(
    input_variables=["abstract"],
    template=(
        "You are a news reporter writing for a general audience. Summarize the following research abstract in "
        "a clear, engaging sentence that highlights the main result and why it matters to readers. Avoid jargon.\n\n"
        "Abstract:\n{abstract}\n\nNews Reporter Summary:"
    )
)

# --- build LLM chains ---
scientist_chain = LLMChain(llm=llm, prompt=scientist_template, output_key="scientist_summary")
reporter_chain = LLMChain(llm=llm, prompt=reporter_template, output_key="reporter_summary")

# --- function to run both chains ---
def generate_two_summaries(abstract: str) -> dict:
    """
    Generate two different summaries of a research abstract.
    """
    try:
        sci_out = scientist_chain.run({"abstract": abstract})
        rep_out = reporter_chain.run({"abstract": abstract})
        return {
            "Scientist Summary": sci_out.strip(),
            "News Reporter Summary": rep_out.strip()
        }
    except Exception as e:
        if "invalid_api_key" in str(e).lower() or "401" in str(e):
            raise RuntimeError(
                f"❌ Invalid API Key Error: {str(e)}\n"
                "Please check if your GROQ_API_KEY is valid and active.\n"
                "You can get a new API key from: https://console.groq.com/keys"
            )
        else:
            raise RuntimeError(f"Error generating summaries: {str(e)}")

# --- example usage ---
if __name__ == "__main__":
    sample_abstract = (
        "Quantum computing is emerging as a field that uses principles of quantum mechanics "
        "to perform calculations much faster than classical computers in certain tasks. "
        "Recent work demonstrates new error-mitigation techniques and variational algorithms "
        "that improve performance on optimization and simulation problems."
    )

    summaries = generate_two_summaries(sample_abstract)
    print("Scientist Summary:", summaries["Scientist Summary"])
    print("News Reporter Summary:", summaries["News Reporter Summary"])
