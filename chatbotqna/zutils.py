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
from langchain import hub

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
embedding = HuggingFaceEmbeddings()


def process_file(file: AskFileResponse):
    if file.type == "text/plain":
        Loader = TextLoader
    elif file.type == "application/pdf":
        Loader = PyPDFLoader

    loader = Loader(file.path)
    documents = loader.load()
    docs = text_splitter.split_documents(documents)
    for i, doc in enumerate(docs):
        doc.metadata["source"] = f"source_{i}"
    
    return docs

def get_vector_db(file: AskFileResponse):
    docs = process_file(file)
    cl.user_session.set("docs", docs)
    vector_db = Chroma.from_documents(documents=docs, embedding=embedding)
    return vector_db

def get_huggingface_llm(model_name: str="lmsys/vicuna-7b-v1.5", max_new_token: int=512):
    nf4_config = BitsAndBytesConfig(load_in_4bit=True,\
                                    bnb_4bit_quant_type="nf4",\
                                    bnb_4bit_use_double_quant=True,\
                                    bnb_4bit_compute_dtype=torch.bfloat16)
    
    model = AutoModelForCausalLM.from_pretrained(model_name,\
                                                 quantization_config=nf4_config,\
                                                low_cpu_mem_usage=True)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model_pipeline = pipeline("text-generation",\
                            model=model,\
                            tokenizer=tokenizer,\
                            max_new_tokens=max_new_token,\
                            pad_token_id=tokenizer.eos_token_id,\
                            device_map="auto")

    llm = HuggingFacePipeline(pipeline=model_pipeline)
    return llm

