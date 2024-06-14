import os
import yaml
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.graphs import Neo4jGraph
from langchain_core.prompts.prompt import PromptTemplate
from langchain.chains import GraphCypherQAChain
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

def config():
    # load_dotenv()

    # Set up Neo4J & Gemini API
    os.environ["NEO4J_URI"] = os.getenv("NEO4J_URI")
    os.environ["NEO4J_USERNAME"] = os.getenv("NEO4J_USERNAME")
    os.environ["NEO4J_PASSWORD"] = os.getenv("NEO4J_PASSWORD")
    os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

def load_prompt(filepath):
    with open(filepath, "r") as file:
        prompt = yaml.safe_load(file)

    return prompt

def init_():
    config()
    knowledge_graph = Neo4jGraph()
    llm_chat = ChatGoogleGenerativeAI(
        model= "gemini-1.5-flash-latest"
    )

    # Connect to Neo4J Knowledge Graph
    cypher_prompt = load_prompt("prompts/cypher_prompt.yaml")
    qa_prompt = load_prompt("prompts/qa_prompt.yaml")

    CYPHER_GENERATION_PROMPT = PromptTemplate(**cypher_prompt)
    QA_GENERATION_PROMPT = PromptTemplate(**qa_prompt)

    chain = GraphCypherQAChain.from_llm(
        llm_chat, graph=knowledge_graph, verbose=True,
        cypher_prompt= CYPHER_GENERATION_PROMPT,
        qa_prompt= QA_GENERATION_PROMPT
    )
    
    return chain

# Init GraphQA Chain     
chain = init_()

def get_llm_response(query):
    return chain.invoke({"query": query})["result"]
    

def llm_answer(message, history):
    # history_langchain_format = []
    #
    # for human, ai in history:
    #     history_langchain_format.append(HumanMessage(content= human))
    #     history_langchain_format.append(AIMessage(content= ai))
    #
    # history_langchain_format.append(HumanMessage(content= message["text"]))

    try:
        response = get_llm_response(message["text"])
    except Exception:
        response = "Exception"
    except Error:
        response = "Error"
    return response

# if __name__ == "__main__":
#     message = "Have any company recruiting jobs about Machine Learning and coresponding job titles?"
#     history = [("What's your name?", "My name is Gemini")]
#     resp = llm_answer(message, history)
#     print(resp)

