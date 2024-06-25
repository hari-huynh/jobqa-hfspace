from langchain.agents import Tool, AgentType, initialize_agent
from langchain.memory import ConversationBufferMemory
# from langchain.utilities import DuckDuckGoSearchAPIWrapper
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor
from langchain import hub
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.tools.render import render_text_description
import os
from tools.kg_search import lookup_kg
from dotenv import load_dotenv
from langchain.agents import Tool
from langchain_core.prompts import PromptTemplate

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")
llm = ChatGoogleGenerativeAI(
    model= "gemini-1.5-flash-latest",
    temperature = 0
)

# search = DuckDuckGoSearchAPIWrapper()
#
# search_tool = Tool(name="Current Search",
#                    func=search.run,
#                    description="Useful when you need to answer questions about detail jobs information or search a job."
#                    )

kg_query = Tool(
    name = 'Query Knowledge Graph',
    func = lookup_kg,
    description='Useful for when you need to answer questions about job posts.'
)


tools = [kg_query]
# memory = ConversationBufferMemory(memory_key="chat_history")
#
# agent_chain = initialize_agent(tools,
#                                llm,
#                                agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
#                                memory=memory,
#                                verbose=True)

# agent_prompt = hub.pull("hwchase17/react-chat")

with open("prompts/react_prompt_v2.txt", "r") as file:
    react_template = file.read()

react_prompt = PromptTemplate(
    input_variables = ["tools", "tool_names", "input", "agent_scratchpad", "chat_history"],
    template = react_template
)

prompt = react_prompt.partial(
    tools = render_text_description(tools),
    tool_names = ", ".join([t.name for t in tools]),
)

llm_with_stop = llm.bind(stop=["\nObservation"])

agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_log_to_str(x["intermediate_steps"]),
        "chat_history": lambda x: x["chat_history"],
    }
    | prompt
    | llm_with_stop
    | ReActSingleInputOutputParser()
)

memory = ConversationBufferMemory(memory_key="chat_history")

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, memory=memory)

# result = agent_executor.invoke({"input": "Have any company recruit Machine Learning jobs?"})
# print(result)

# result = agent_chain.run(input = "Have any company recruit Machine Learning jobs?")
# print(result)

# question = {
#     "input": "What did I just ask?"
# }
#
# result = agent_executor.invoke(question)
# print(result)

if __name__ == "__main__":
    while True:
        try:
            question = input("> ")
            result = agent_executor.invoke({
                "input": question
            })
        except:
            break
