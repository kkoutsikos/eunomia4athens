import streamlit as st
from streamlit_chat import message
import tempfile


from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings

# Define the path for generated embeddings
DB_FAISS_PATH = 'vectorstore/db_faiss'


# Load the model of choice
def load_llm():
    llm = ChatOpenAI(
        openai_api_key = st.secrets.openai_api_key,
        model="gpt-4",
        temperature=0.1,
    )
    return llm


st.title("Δημοτικος Βοηθος")

uploaded_file = st.sidebar.file_uploader("Upload File", type="pdf")


if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    # Load CSV data using CSVLoader
    loader = PyPDFLoader(file_path=tmp_file_path)
    data = loader.load()
    #Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
    texts = text_splitter.split_documents(data)
    # Create embeddings using Sentence Transformers
    embeddings = HuggingFaceEmbeddings(model_name='lighteternal/stsb-xlm-r-greek-transfer')

    
    
    # Create a FAISS vector store and save embeddings
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)

    # Load the language model
    llm = load_llm()
    # Create retriever
    bm25_retriever = BM25Retriever.from_documents(texts)
    bm25_retriever.k = 4
    faiss_retriever = db.as_retriever(search_kwargs={"k":4})
    ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.75])
    
    # Create Prompt 
    
    prompt_template = """""Θα σου δώσω συγκεκριμένες πληροφορίες [ΠΛΗΡΟΦΟΡΙΕΣ] και μια 
                        ερώτηση [ΕΡΩΤΗΣΗ]. Βάσει των πληροφοριών αυτών, θέλω μια αναλυτική και ακριβή απάντηση [ΑΠΑΝΤΗΣΗ]. Απαντάς σε έναν απλό πολίτη σαν εκπρόσωπος του Δήμου του. Εάν δεν διαθέτεις 
                        αρκετές πληροφορίες για μια συγκεκριμένη απάντηση, χρησιμοποίησε τη φράση: 'Δυστυχώς δεν έχω τις απαραίτητες 
                        πληροφορίες για να σου προσφέρω μια αξιόπιστη απάντηση. ' 
                        ΠΛΗΡΟΦΟΡΙΕΣ: \n{context}\n
                        ΕΡΩΤΗΣΗ: \n{question}\n
                        ΑΠΑΝΤΗΣΗ:"
                        """

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=ensemble_retriever)

    
    def conversational_chat(query):
        result = chain({"question": query, "chat_history": st.session_state['history']})
        st.session_state['history'].append((query, result["answer"]))
        return result["answer"]

    
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    
    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Καλημέρα, πως μπορώ να βοηθήσω?"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey ! 👋"]

    
    response_container = st.container()
    container = st.container()

    
    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Query:", placeholder="Ερώτηση στον βοηθό", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            output = conversational_chat(user_input)
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    
    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")
