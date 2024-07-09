from zutils import get_huggingface_llm, get_vector_db
import torch
import chainlit as cl
from chainlit.types import AskFileResponse

from transformers import BitsAndBytesConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from langchain_huggingface.llms import HuggingFacePipeline
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader, TextLoader

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain import hub

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

LLM = get_huggingface_llm()

welcome_message = """Welcome to PDF Q&A. To start:
1. Upload PDF document.
2. Asj a question about the file.
"""

@cl.on_chat_start
async def on_chat_start():
    files = None
    while files is None:
        files = await cl.AskFileMessage(content=welcome_message,\
                                        accept=["text/plain", "application/pdf"],\
                                        max_size_mb=20,\
                                        timeout=180
                                        ).send()
        
        file = files[0]

        msg = cl.Message(content=f"Processing '{file.name}'...", disable_feedback=True)

        await msg.send()

        vector_db = await cl.make_async(get_vector_db)(file)

        message_history = ChatMessageHistory()
        memory = ConversationBufferMemory(memory_key="chat_history",\
                                            output_key="answer",\
                                            chat_memory=message_history,\
                                            return_messages=True)

        retriever = vector_db.as_retriever(search_type="mmr",\
                                           search_kwargs={"k":3})
        
        chain = ConversationalRetrievalChain.from_llm(llm=LLM,\
                                                        chain_type="stuff",\
                                                        retriever=retriever,\
                                                        memory=memory,
                                                        return_source_documents=True)
        
        msg.content = f"{file.name} processed. You can now ask questions."
        await msg.update()

        cl.user_session.set("chain", chain)

@cl.on_message
async def on_message(message: cl.Message):
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler()
    res = await chain.ainvoke(message.content, callbacks=[cb])
    answer = res["answer"]
    source_documents = res["source_documents"]
    text_elements = []

    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx}"
            text_elements.append(
                                cl.Text(content=source_doc.page_content,\
                                        name=source_name)
                                )
            source_names = [text_el.name for text_el in text_elements]

            if source_names:
                answer += f"\nSources:{','.join(source_names)}"
            else:
                answer += "\nNo source found."
    await cl.Message(content=answer, elements=text_elements).send()




