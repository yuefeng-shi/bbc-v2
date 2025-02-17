from langchain_core.prompts import ChatPromptTemplate

from langchain_community.vectorstores import FAISS

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings



def retri_gen_QA(vectordb, keys, query, llm_type):
    
    if llm_type == 'gpt-3.5':
        llm = ChatOpenAI(
            model="gpt-3.5-turbo-1106",
            temperature=0,
            max_retries=2,
            api_key=keys,
        )
    elif llm_type == 'gpt-4':
        llm = ChatOpenAI(
            model="gpt-4-turbo",
            temperature=0,
            max_retries=2,
            api_key=keys,)
    else:
        raise ValueError('wrong type of llms')   
        
    retriever = vectordb.as_retriever()
    system_prompt =  (
    "Use the given context to answer the question. "
    "If you don't know the answer, say you don't know. "
    "Use five sentence maximum and keep the answer concise. "
    "Context: {context}"
)
    summarization_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    combine_docs_chain = create_stuff_documents_chain(llm, summarization_template)
    rag_chain = create_retrieval_chain(retriever, combine_docs_chain)
    res = rag_chain.invoke({"input": query})

    ref_list = []

    sim_res = vectordb.similarity_search_with_score(query=query, k=4)

    for _, score in sim_res:
        ref_list.append(score)


    # pdb.set_trace()
    print('Question:')
    print(res['input'])
    print ('Answer:')
    print (res['answer'])
    print('References:')
    for item in res['context']:
        print (item.metadata['source'][11:-4])
        print (item.metadata['date'])
    
    return res, ref_list


def retri_gen_QA_final(vectordb_dir, keys, query, llm_type):
    
    if llm_type == 'gpt-3.5':
        llm = ChatOpenAI(
            model="gpt-3.5-turbo-1106",
            temperature=0,
            max_retries=2,
            api_key=keys,
        )
    elif llm_type == 'gpt-4':
        llm = ChatOpenAI(
            model="gpt-4-turbo",
            temperature=0,
            max_retries=2,
            api_key=keys,)
    else:
        raise ValueError('wrong type of llms') 

    embeddings = OpenAIEmbeddings(openai_api_key=keys, model="text-embedding-3-small",  chunk_size=300)  
    
    vectordb = FAISS.load_local(vectordb_dir, embeddings, allow_dangerous_deserialization=True)
        
    retriever = vectordb.as_retriever()
    system_prompt =  (
    "Use the given context to answer the question. "
    "If you don't know the answer, say you don't know. "
    "Use five sentence maximum and keep the answer concise. "
    "Context: {context}"
)
    summarization_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    combine_docs_chain = create_stuff_documents_chain(llm, summarization_template)
    rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

    ref_list = []
    sim_res = vectordb.similarity_search_with_score(query=query, k=4)

    for _, score in sim_res:
        ref_list.append(score)

    res = rag_chain.invoke({"input": query})
    print(res)
    return res, sim_res
