import streamlit as st

from full_chain import retri_gen_QA, retri_gen_QA_final
from vector_store import create_vector_db
from local_loader import get_document_text
from splitter import split_documents
import base64

def get_image_base64(path):
    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()


st.set_page_config(page_title="AI Archive Explorer")
# st.title("AI Archive Explorer")
logo = get_image_base64("pic/cannon.png")
st.markdown(
     f"""
    <h1 style='display: inline; funt-size: 60px; color: orange'>
        <img src="data:image/png;base64,{logo}" 
             style='vertical-align: middle; width: 60px; margin-left: 10px;'>
        AI Archive Explorer
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #2F4F4F; 
    }
    .stWrite {
        color: #FFFDD0; 
    }
    .stSelectbox label {
        color: #FFFDD0; 
    }
    .stSelectbox select {
        color: #FFFDD0; 
    }
    .stButton button {
        color: #FFFDD0; 
    }
    .stForm {
        color: #FFFDD0;
    }
    .stForm label {
        color: #FFFDD0; 
    }

    
    .stSidebar {
        background-color: #2F4F4F; 
        color: #FFFDD0; 
    }
    .stSidebar .stMarkdown {
        color: #FFFDD0; 
    }
    .stSidebar label {
        color: #FFFDD0; 
    }
    .st-emotion-cache-h4xjwg {
        background-color: #2F4F4F !important; 
    }
    </style>
    """,
    unsafe_allow_html=True
)


openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")

st.write('Please choose the Large Lange Model.')
llm_type = st.selectbox('Select', ['GPT-3.5-Turbo', 'GPT-4-Turbo'])

st.write('Please choose the Data Source.')
data_type = st.selectbox('Select', ['Correspondence', 'Newspaper'])


with st.form("my_form"):
    text = st.text_area(
        "Enter text:",
    )
    submitted = st.form_submit_button("Submit")
    if not openai_api_key.startswith("sk-"):
        st.warning("Please enter your OpenAI API key!", icon="⚠")
    if llm_type not in ['GPT-3.5-Turbo', 'GPT-4-Turbo']:
        st.warning("Please select the LLM, or it cannot work", icon="⚠")
    if data_type not in ['Correspondence', 'Newspaper']:
        st.warning("Please select the data type, or it cannot work", icon="⚠")

    if submitted and openai_api_key.startswith("sk-"):
        type = ''    
        if llm_type == 'GPT-3.5-Turbo':
            type = 'gpt-3.5'
        else:
            type = 'gpt-4'
        
        if text == '':
            st.warning("Please input texts", icon="⚠")
        
        json_select = ''
        txt_select = ''
        emb_select = ''
        if data_type == 'Correspondence':
            json_select = 'filtered.json'
            txt_select = '../text_files'
            emb_select = 'faiss_emb'
        else:
            json_select = 'others.json'
            txt_select = '../text_others_files'
            emb_select = 'faiss_others_emb'

        # data = get_document_text(json_select, txt_select)
        # data_chunks = split_documents(data)
        # dbvector = create_vector_db(data, openai_api_key, emb_select)

    
        # res,_ = retri_gen_QA(dbvector, openai_api_key, text, llm_type=type)

        res,_ = retri_gen_QA_final(vectordb_dir=emb_select , keys= openai_api_key, query = text, llm_type=type)

        st.write('Generated Answers:')
        st.info(res['answer'])
        st.write('References:')
        for i, item in enumerate(res['context']):
            # st.info('Text ID and Date for Reference ' + str(i) + ': \n' + item.metadata['source'][11:-4] + ';' + item.metadata['date'])
            st.info('Text ID Date and Original file for Reference ' + str(i + 1) + ':')
            if data_type == 'Correspondence':
                st.write(item.metadata['source'][14:-4])
                st.write(item.metadata['date']) 
                st.write(item.metadata['full_text'])
            # st.info( item.metadata['source'][11:-4] )
            # st.info(item.metadata['date'])
            # st.info(item.metadata['full_text'])
            else:
                st.write(item.metadata['source'][21:-4])
                st.write(item.metadata['date']) 
                st.write(item.metadata['full_text']
                    )
