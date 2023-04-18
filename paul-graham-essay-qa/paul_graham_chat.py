import asyncio
import pickle

import semantic_kernel as sk
from semantic_kernel.ai.open_ai import OpenAITextCompletion, OpenAITextEmbedding, OpenAIChatCompletion
from langchain.text_splitter import RecursiveCharacterTextSplitter

import streamlit as st

print("Loading text data...")
with open("../data/paul_graham_essay.txt", "r") as f:
    document = f.read()

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 1000,
    chunk_overlap  = 20,
    length_function = len,
)

print("Splitting the text data into chunks...")
texts = text_splitter.create_documents([document])
len(texts)

sk_prompt = """
ChatBot can have a conversation with you about any topic.
It can give explicit instructions or say 'I don't know' if
it does not have an answer.

Use this primary source information if relevant:
{{recall $user_input}}

Chat:
{{$chat_history}}
User: {{$user_input}}
ChatBot: """.strip()

print("Building the kernel...")
kernel = sk.Kernel()

api_key, org_id = sk.openai_settings_from_dot_env()
kernel.config.add_text_backend("dv", OpenAITextCompletion("text-davinci-003", api_key, org_id))
kernel.config.add_embedding_backend("ada", OpenAITextEmbedding("text-embedding-ada-002", api_key, org_id))

kernel.register_memory_store(memory_store=sk.memory.VolatileMemoryStore())
kernel.import_skill(sk.core_skills.TextMemorySkill())

chat_func = kernel.create_semantic_function(sk_prompt, max_tokens=200, temperature=0.8)

context = kernel.create_new_context()

context[sk.core_skills.TextMemorySkill.COLLECTION_PARAM] = "primarySource"
context[sk.core_skills.TextMemorySkill.RELEVANCE_PARAM] = 0.8

context["chat_history"] = ""

print("Creating memories...")
with open("kernel_storage_store.pkl", "rb") as f:
    kernel.memory._storage._store = pickle.load(f)

# UNCOMMENT THIS SECTION IF YOU WANT TO BUILD YOUR OWN EMBEDDINGS
# collection_name = "paul-graham-essays"
# print("Adding texts into the Semantic Kernel's memory")
# for idx, value in enumerate(texts):
#     asyncio.run(kernel.memory.save_reference_async(
#         collection=collection_name,
#         description=value.page_content,
#         text=value.page_content,
#         external_id=idx,
#         external_source_name=collection_name
#     ))
#     print("URL {} saved".format(idx))

print("Initializing Streamlit chat")
st.title("Chat with PaulG")
st.sidebar.header("Instructions")
st.sidebar.info(
    """
    This is a web application that allows you to interact with 
    the Paul Graham, founder of YCombinator.
    Enter a **query** in the **text box** and **press enter** to receive 
    a **response**
    """
    )

# Get user input
user_query = st.text_input("Enter question here, to exit enter :q", "What does Paul say you should look for in a cofounder?")
if user_query != ":q" or user_query != "":
    # Pass the query to the ChatGPT function
    context['user_input'] = user_query
    response = asyncio.run((kernel.run_on_vars_async(context.variables, chat_func)))
    st.write(f"{user_query}\n\n {response}")
