import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents.base import Document
from langchain_ai21 import AI21Embeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough, RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

def load_and_process_pdfs(pdf_folder_path: str) -> list[Document]:
    documents: list = []
    try:
        for file in os.listdir(pdf_folder_path):
            if file.endswith('.pdf'):
                pdf_path = os.path.join(pdf_folder_path, file)
                loader = PyPDFLoader(pdf_path)
                documents.extend(loader.load())
    except Exception as e:
        print(f"Error loading PDFs: {e}")
    text_splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits: list[Document] = text_splitter.split_documents(documents)
    return splits

def initialize_vectorstore(splits: list[Document]) -> FAISS:
    embedding_model: AI21Embeddings = AI21Embeddings()
    try:
        return FAISS.from_documents(documents=splits, embedding=embedding_model)
    except Exception as e:
        print(f"Error initializing vector store: {e}")
        return None

def format_docs(documents: list[Document]) -> str:
    return "\n\n".join(document.page_content for document in documents)

splits: list[Document] = load_and_process_pdfs('documents')
if splits:
    vectorstore: FAISS = initialize_vectorstore(splits)
    if vectorstore:
        prompt_template: str = """You are a finance expert. You need to answer the question related to finance. 
        Given below is the context and question of the user.
        context = {context}
        question = {question}
        """
        
        prompt: ChatPromptTemplate = ChatPromptTemplate.from_template(template=prompt_template)
        
        llm: GoogleGenerativeAI = GoogleGenerativeAI(model="gemini-pro")
        
        rag_chain: RunnableSequence = (
            {"context": vectorstore.as_retriever() | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        continue_asking: bool = True
        while continue_asking:
            try:
                print(rag_chain.invoke(input("Enter your question: "))) 
                continue_asking = int(input("Do you want to ask another question?\n 0. No\n 1. Yes\n"))
            except Exception as e:
                print(f"An error occurred: {e}")
