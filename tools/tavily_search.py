import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools import BaseTool, StructuredTool, tool

load_dotenv()

os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")


def tavily_search(question: str) -> str:
    """
    useful for when you need to search relevant informations such as: jobs, companies from Web sites.
    """

    # setup prompt
    # prompt = [{
    #     "role": "system",
    #     "content": f'You are an AI critical thinker research assistant. ' \
    #                f'Your sole purpose is to write well written, critically acclaimed,' \
    #                f'objective and structured reports on given text.'
    # }, {
    #     "role": "user",
    #     "content": f'Information: """{content}"""\n\n' \
    #                f'Using the above information, answer the following' \
    #                f'query: "{query}" in a detailed report --' \
    #                f'Please use MLA format and markdown syntax.'
    # }]

    tool_search = TavilySearchResults(
        max_results = 3,
        include_raw_content = True
    )

    # prompt_search = f"""You are an expert at finding information about the job,
    #     the company, and the skills required for that job.
    #     Try to find out what is relevant to the company, the job, and the skills required for that job.
    #     If the questions are not relevant, answer them in your own words.
    #
    #     Query: {question}
    # """

    # Search
    # for information on Web sites: Indeed, LinkedIn, TopCV
    # by
    # using
    # entity in user
    # question(Job
    # Titles, Company, Location, etc).
    # Using
    # search
    # pattern: site:indeed

    search_prompt = f"""
    Response to user question by search job descriptions include: job titles, company, required skill, education, etc related to job recruitment posts in Vietnam. 
    
    Query: {question}
    """

    result = tool_search.invoke({"query": search_prompt})

    # llm_chat = ChatGoogleGenerativeAI(
    #     model = "gemini-1.5-flash-latest",
    #     temperature = 0
    # )

    # content = []
    # for i in result:
    #     content.append(i['content'])

    # prompt = f"""
    #
    # You are a career consultant, based on the information you have  contents: {content},
    # consider yourself an expert to summarize summary details not too short the content and
    # highlight the content related to the company's job and the necessary skills and return must 1 URL
    #
    # You can add information you know about the question {question}
    # """

    # response_prompt = f"""
    # Generate a concise and informative summary of the results in a polite and easy-to-understand manner based on question and Tavily search results.
    # Returns URLs at the end of the summary for proof.
    #
    # Question: {question}
    # Search Results: {str(result)}
    #
    # Answer:
    # """

    # response = llm_chat.invoke(response_prompt)

    return result


if __name__ == "__main__":
    question = "Recruitment information for the position of Software Engineer?"
    result = tavily_search(question)
    print(result)