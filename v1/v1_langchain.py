from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI


# Load environment variables
load_dotenv()

# Create Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0.3,
)

# Create prompt template
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful Courier and Logistics Policy Assistant."
    ),
    (
        "human",
        "Answer the following question clearly:\n\n{question}"
    )
])

# Create output parser
parser = StrOutputParser()

# Create Runnable chain
chain = prompt | llm | parser


# Get user question
question = input("Ask your courier/logistics question: ")

# Run the chain
response = chain.invoke({
    "question": question
})

# Display answer
print("\nAssistant:")
print(response)