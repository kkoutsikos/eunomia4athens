import streamlit as st
from streamlit_chat import message
import os,tempfile
from pathlib import Path
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.llms.openai import OpenAIChat
# Define the path for generated embeddings
TMP_DIR = Path(__file__).resolve().parent.joinpath('data', 'tmp')
LOCAL_VECTOR_STORE_DIR = Path(__file__).resolve().parent.joinpath('data', 'vector_store')



# Set the title for the Streamlit app
st.title("Δημοτικος Βοηθος")



def load_documents():
    loader = DirectoryLoader(TMP_DIR.as_posix(), glob='**/*.pdf')
    documents = loader.load()
    return documents


def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
    texts = text_splitter.split_documents(documents)
    return texts


def embeddings_on_local_vectordb(texts):
    embeddings = HuggingFaceEmbeddings(model_name='lighteternal/stsb-xlm-r-greek-transfer')
    vectordb = FAISS.from_documents(texts, embedding=embeddings,
                                     persist_directory=LOCAL_VECTOR_STORE_DIR.as_posix())
    vectordb.persist()
    
    bm25_retriever = BM25Retriever.from_documents(texts)
    bm25_retriever.k = 4
    faiss_retriever = vectordb.as_retriever(search_kwargs={"k":4})
    retriever = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.75])
    return retriever

def query_llm(retriever, query):
    prompt_template = """""Θα σου δώσω συγκεκριμένες πληροφορίες [ΠΛΗΡΟΦΟΡΙΕΣ] και μια 
                        ερώτηση [ΕΡΩΤΗΣΗ]. Βάσει των πληροφοριών αυτών, θέλω μια αναλυτική και ακριβή απάντηση [ΑΠΑΝΤΗΣΗ]. Απαντάς σε έναν απλό πολίτη σαν εκπρόσωπος του Δήμου του. Εάν δεν διαθέτεις 
                        αρκετές πληροφορίες για μια συγκεκριμένη απάντηση, χρησιμοποίησε τη φράση: 'Δυστυχώς δεν έχω τις απαραίτητες 
                        πληροφορίες για να σου προσφέρω μια αξιόπιστη απάντηση. ' 
                        ΠΛΗΡΟΦΟΡΙΕΣ: \n{context}\n
                        ΕΡΩΤΗΣΗ: \n{question}\n
                        ΑΠΑΝΤΗΣΗ:"
                        """

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    # Create a conversational chain
    llm = ChatOpenAI(temperature=0.5, model='gpt-4', openai_api_key = st.secrets.openai_api_key)
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, chain_type_kwargs={"prompt" : prompt, "verbose": True}) 


    result = chain.run({'question': query, 'chat_history': st.session_state.messages})
    
    
    
    
    result = result['answer']
    st.session_state.messages.append((query, result))
    return result
# Create a file uploader in the sidebar

def input_fields():
    #
    with st.sidebar:
        #
        if "openai_api_key" in st.secrets:
            st.session_state.openai_api_key = st.secrets.openai_api_key
        else:
            st.session_state.openai_api_key = st.text_input("OpenAI API key", type="password")
        
    
    st.session_state.source_docs = st.file_uploader(label="Upload Documents", type="pdf", accept_multiple_files=True)
    
# Handle file upload
def process_documents():
    if not st.session_state.openai_api_key or not st.session_state.source_docs:
        st.warning(f"Please upload the documents and provide the missing fields.")
    else:
        try:
            for source_doc in st.session_state.source_docs:
                #
                with tempfile.NamedTemporaryFile(delete=False, dir=TMP_DIR.as_posix(), suffix='.pdf') as tmp_file:
                    tmp_file.write(source_doc.read())
                #
                documents = load_documents()
                #
                for _file in TMP_DIR.iterdir():
                    temp_file = TMP_DIR.joinpath(_file)
                    temp_file.unlink()
                #
                texts = split_documents(documents)
                #
                
                st.session_state.retriever = embeddings_on_local_vectordb(texts)
                
        except Exception as e:
            st.error(f"An error occurred: {e}")
            
def boot():
    #
    input_fields()
    #
    st.button("Ανεβαστε αρχεια", on_click=process_documents)
    #
    if "messages" not in st.session_state:
        st.session_state.messages = []
    #
    for message in st.session_state.messages:
        st.chat_message('πολιτης').write(message[0])
        st.chat_message('βοηθος').write(message[1])    
    #
    if query := st.chat_input():
        st.chat_message("πολιτης").write(query)
        response = query_llm(st.session_state.retriever, query)
        st.chat_message("βοηθος").write(response)

if __name__ == '__main__':
    #
    boot()            