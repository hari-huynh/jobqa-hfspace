from tools import kg_search
from tools.kg_search import lookup_kg
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.agents import Tool
from utils.utils import init_
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

kg_query = Tool(
    name = 'Query Knowledge Graph',
    func = lookup_kg,
    description='Useful for when you need to answer questions about job posts.'
)

tools = [kg_query]

with open("prompts/react_prompt.txt", "r") as file:
    react_template = file.read()

react_prompt = PromptTemplate(
    input_variables = ["tools", "tool_names", "input", "agent_scratchpad"],
    template = react_template
)

prompt = ChatPromptTemplate.from_messages([
    react_template,
    MessagesPlaceholder(variable_name = "chat_history")
])

_, llm = init_()

# Init ReAct agent
agent = create_react_agent(llm, tools, react_prompt)
agent_executor = AgentExecutor(
    agent = agent,
    tools = tools,
    verbose = True
)

message_history = ChatMessageHistory()

agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    lambda session_id : message_history,
    input_messages_key = "input",
    history_messages_key = "chat_history"
)

if __name__ == "__main__":
    # Test ReAct Agent
    question = {
        "input": "Have any company recruit Machine Learning jobs?"
    }
    result = agent_with_chat_history.invoke(
        question,
        config = {"configurable": {"session_id": "foo"}}
    )
    print(result)

    print("Answered!!!!!!!!")

    # Test memory
    question = {
        "input": "What did I just ask?"
    }
    result = agent_with_chat_history.invoke(
        question,
        config={"configurable": {"session_id": "foo"}}
    )
    print(result)

    x = input("> ")




