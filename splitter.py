# Split documents into chunks
from langchain.text_splitter import RecursiveJsonSplitter, RecursiveCharacterTextSplitter


def split_documents(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000,
                                                   chunk_overlap=50,
                                                   length_function=len,
                                                   is_separator_regex=False,)
    text_chunks = text_splitter.split_documents(docs)
    return text_chunks
