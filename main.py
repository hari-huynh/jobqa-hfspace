import gradio as gr
from utils import llm_answer

# demo = gr.ChatInterface(
#     fn = echo,
#     multimodal= True,    # Allow upload CV
#     examples = ["hello", "hola", "merhaba"],
#     title= "Echo Bot",
#     description= "Chatbot to answer about job postings & recommend jobs based on your CV.",
#     theme= None,
#     autofocus= True,
#     fill_height= False,
# )


demo = gr.ChatInterface(
    fn= llm_answer,
    examples=[],
    title="Jobs QA & Recommendations Chatbot",
    multimodal=True,
)

demo.launch()