from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from react_agent_v2 import agent_executor
import chainlit as cl
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# agent_with_chat_history.invoke("Have any company recruit Machine Learning jobs?")


@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("runnable", agent_executor)

@cl.on_message
async def on_message(message: cl.Message):
    # runnable = cl.user_session.get("runnable")  # type: Runnable
    #
    # msg = cl.Message(content="")
    #
    # for chunk in await cl.make_async(runnable.stream)(
    #     {"input": message.content},
    #     config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    # ):
    #     await msg.stream_token(chunk)
    #
    # await msg.send()
#
#     # Get user input from the message
#     user_input = message.content
#
#     # Run the agent with user input and get the response
#     response = await cl.make_async(agent_executor)(user_input)
#
#     # Display the response to the user
#     cl.message(response)

    llm_chain = cl.user_session.get("runnable")

    response = await llm_chain.ainvoke({
        "input": message.content
        },
        callbacks = [cl.AsyncLangchainCallbackHandler()]
    )

    await cl.Message(response["output"].replace("`", "")).send()
