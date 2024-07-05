import os
import yaml
from dotenv import load_dotenv
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.schema.output_parser import StrOutputParser
from langchain.tools import BaseTool, StructuredTool, tool
from langchain_community.graphs import Neo4jGraph
# from utils import utils


# Question-Cypher pair examples
with open("prompts/cypher_examples.yaml", "r") as f:
    example_pairs = yaml.safe_load(f)
    
examples = example_pairs["examples"]

# LLM for choose the best similar examples
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

embedding_model = GoogleGenerativeAIEmbeddings(
    model= "models/text-embedding-004"
)

example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples = examples,
    embeddings = embedding_model,
    vectorstore_cls = FAISS,
    k = 1
)

# Load schema, prefix, suffix
with open("prompts/schema.txt", "r") as file:
    schema = file.read()

with open("prompts/cypher_instruct.yaml", "r") as file:
    instruct = yaml.safe_load(file)

example_prompt = PromptTemplate(
    input_variables = ["question_example", "cypher_example"],
    template = instruct["example_template"]
)

dynamic_prompt = FewShotPromptTemplate(
    example_selector = example_selector,
    example_prompt = example_prompt,
    prefix = instruct["prefix"],
    suffix = instruct["suffix"].format(schema=schema),
    input_variables = ["question"]
)


def generate_cypher(question: str) -> str:
    """Make Cypher query from given question."""
    load_dotenv()

    # Set up Neo4J & Gemini API
    os.environ["NEO4J_URI"] = os.getenv("NEO4J_URI")
    os.environ["NEO4J_USERNAME"] = os.getenv("NEO4J_USERNAME")
    os.environ["NEO4J_PASSWORD"] = os.getenv("NEO4J_PASSWORD")
    os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

    gemini_chat = ChatGoogleGenerativeAI(
        model= "gemini-1.5-flash-latest"
    )

    chat_messages = [
      SystemMessage(content= dynamic_prompt.format(question=question)),
    ]


    output_parser = StrOutputParser()
    cypher_statement = []
    chain = dynamic_prompt | gemini_chat | output_parser
    cypher_statement = chain.invoke({"question": question})
    cypher_statement = cypher_statement.replace("```", "").replace("cypher", "").strip()

    return cypher_statement

def run_cypher(question, cypher_statement: str) -> str:
    """Return result of Cypher query from Knowledge Graph."""
    knowledge_graph = Neo4jGraph()
    result = knowledge_graph.query(cypher_statement)
    print(f"\nCypher Result:\n{result}")

    gemini_chat = ChatGoogleGenerativeAI(
        model= "gemini-1.5-flash-latest"
    )

    answer_prompt = f"""
    Generate a concise and informative summary of the results in a polite and easy-to-understand manner based on question and Cypher query response.
    Question: {question}
    Response: {str(result)}

    Avoid repeat information.
    If response is empty, you should answer "Knowledge graph doesn't have enough information".
    Answer:
    """

    sys_answer_prompt = [
        SystemMessage(content= answer_prompt),
        HumanMessage(content="Provide information about question from knowledge graph")
    ]

    response = gemini_chat.invoke(sys_answer_prompt)
    answer = response.content
    return answer

def lookup_kg(question: str) -> str:
    """Based on question, make and run Cypher statements.
    question: str
        Raw question from user input
    """
    cypher_statement = generate_cypher(question)
    cypher_statement = cypher_statement.replace("cypher", "").replace("```", "").strip()
    print(f"\nQuery:\n {cypher_statement}")

    try:
        answer = run_cypher(question, cypher_statement)
    except:
        answer = "Knowledge graph doesn't have enough information\n"

    return answer


if __name__ == "__main__":
    question = "Have any company is recruiting Machine Learning jobs?"

    # Test few-shot template
    # print(dynamic_prompt.format(question = "What does the Software Engineer job usually require?"))

    # # Test generate Cypher
    # result = generate_cypher(question)

    # # Test return information from Cypher
    # final_result = run_cypher(result)
    # print(final_result)

    # Test lookup_kg tool
    kg_info = lookup_kg(question)
    print(kg_info)