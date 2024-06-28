import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from tavily import TavilyClient
from langchain.tools import BaseTool, StructuredTool, tool

load_dotenv()

os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

def tavily_search(question: str) -> str:
    """
    useful for when you need to search relevant informations such as: jobs, companies from Web sites.
    """

    search_prompt = f"""
    Response to user question by search job descriptions include: job titles, company, required skill, education, etc related to job recruitment posts in Vietnam. 

    Query: {question}
    """

    tavily = TavilyClient(
        api_key = os.environ["TAVILY_API_KEY"],
    )

    response = tavily.search(
        query = question,
        include_raw_content = True,
        max_results = 5
    )

    search_results = ""
    for obj in response["results"]:
        search_results += f"""
- Page content: {obj["raw_content"]}
Source: {obj["url"]}
          
        """

    print(search_results)

    response_prompt = f"""
    Generate a concise and informative summary of the results in a polite and easy-to-understand manner based on question and Tavily search results.
    Returns URLs at the end of the summary for proof.

    Question: {question}
    Search Results: 
    {search_results}

    Answer:
    """

    # return context

def tavily_qna_search(question: str) -> str:
    tavily = TavilyClient(
        api_key=os.environ["TAVILY_API_KEY"],
    )

    response = tavily.qna_search(query=question)
    return response

if __name__ == "__main__":
    question = "Software Engineer job postings in Vietnam"

    result = tavily_search(question)
    print(result)