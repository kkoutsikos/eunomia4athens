import streamlit as st
import streamlit as st
from streamlit_chat import message



from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings

st.title("Δημοτικός βοηθός")

@st.cache_data
def loadpdf():
    loader = PyPDFDirectoryLoader(r"data")
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    texts = text_splitter.split_documents(pages)
    return texts

@st.cache_data
def create_vectorstore(texts):
    embeddings = HuggingFaceEmbeddings(model_name='lighteternal/stsb-xlm-r-greek-transfer')
    db = FAISS.from_documents(texts, embeddings)
    
    return db



prompt_template = """""Θα σου δώσω συγκεκριμένες πληροφορίες [ΠΛΗΡΟΦΟΡΙΕΣ] και μια 
                        ερώτηση [ΕΡΩΤΗΣΗ]. Βάσει αυτών, θέλω μια αναλυτική και ακριβή απάντηση [ΑΠΑΝΤΗΣΗ]. Απαντάς σε έναν απλό πολίτη σαν εκπρόσωπος του Δήμου του. Εάν δεν διαθέτεις 
                        αρκετές πληροφορίες για μια συγκεκριμένη απάντηση, χρησιμοποίησε τη φράση: 'Δυστυχώς δεν έχω τις απαραίτητες 
                        πληροφορίες για να σου προσφέρω μια αξιόπιστη απάντηση. ' 
                        ΠΛΗΡΟΦΟΡΙΕΣ: \n{context}\n
                        ΕΡΩΤΗΣΗ: \n{question}\n
                        ΑΠΑΝΤΗΣΗ:"
                        """

prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

texts = loadpdf()
vectorstore = create_vectorstore(texts)
bm25_retriever = BM25Retriever.from_documents(texts)
bm25_retriever.k = 4
faiss_retriever = vectorstore.as_retriever(search_kwargs={"k":4})
ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.75])

llm = ChatOpenAI(temperature=0.5, model='gpt-4')
chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever= ensemble_retriever, chain_type_kwargs={"prompt" : prompt, "verbose": True}) 





user_input = st.text_input("Ρωτήστε τον βοηθό", "")

if st.button("Submit"):
    try:
        response = chain.run(user_input)
        st.write(response)
    except Exception as e:
        st.write(f"An error occurred: {e}")
