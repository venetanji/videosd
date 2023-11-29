from langchain.llms import LlamaCpp
from langchain.chains import LLMChain
from langchain_experimental.chat_models import Llama2Chat
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,    
)
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from fastapi import FastAPI
from langserve import add_routes

app = FastAPI(
  title="LangChain Server",
  version="1.0",
  description="A simple api server using Langchain's Runnable interfaces",
)

n_gpu_layers = 40  # Metal set to 1 is enough.
n_batch = 256  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.

# Make sure the model path is correct for your system!
llm = LlamaCpp(
    model_path="/models/model.gguf",
    n_gpu_layers=0,
    n_batch=n_batch,
    n_ctx=256,
    f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
    verbose=True,
    max_new_tokens=30,
    temperature=5,
    top_k=60,
    top_p=1,
)

system = """
    I want you to act as a image prompt generator program.
    The user will provide you with a subject. You will add style keywords to the subject.
    If the user provides style keywords in input, ignore them, but use the same subject.
    You can use any art style, for example modern, ancient, primitive, western or eastern. 
    Feel free to add artists names at the end. 
    You will only answer with an image description, no chat. 
    Keep your response within 20 words and answer without using quotes.
"""


template_messages = [
    SystemMessage(content=system),
    HumanMessage(content='A landscape'),
    AIMessage(content='A painting of a landscape, a valley of a battlefield, caos around the mirror, realistic, well done, detailed, 8k'),
    # HumanMessage(content='A face'),
    # AIMessage(content='A digital illustration of a face, British, 20 years old, male, Caucasian, permed hair, cute smile, goddess, Makoto Shinkai style Studio Ghibli Genshin Impact'),
    HumanMessagePromptTemplate.from_template("{text}"),
]

prompt_template = ChatPromptTemplate.from_messages(template_messages)
chain = LLMChain(llm=Llama2Chat(llm=llm), prompt=prompt_template)


class ChainInput(BaseModel):
    """Input for the chat bot."""
    text: str = Field(..., description="User's latest query.")

add_routes(
    app,
    chain,
    path="/llama-chat",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)