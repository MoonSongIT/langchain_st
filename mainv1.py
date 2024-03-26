import streamlit as st
import tiktoken
from loguru import logger

from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI
# from langchain_openai import ChatOpenAI

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import UnstructuredPowerPointLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS

# from streamlit_chat import message
from langchain_community.callbacks import get_openai_callback
from langchain.memory import StreamlitChatMessageHistory

def main():
    st.set_page_config(
    page_title="무었이든",
    page_icon=":volcano:")

    st.title("_무었이 불편 하실까? :red[QA Chat]_ :volcano:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    with st.sidebar:
        uploaded_files =  st.file_uploader("Upload your file",type=['pdf','docx'],accept_multiple_files=True)
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")

        # Streamlit 사이드바에 슬라이더 추가
        chunk_size = st.sidebar.slider("Chunk Size", min_value=100, max_value=2000, value=900, step=50)
        chunk_overlap = st.sidebar.slider("Chunk Overlap", min_value=50, max_value=500, value=100, step=10)
        # Streamlit 사이드바 콤보박스 추가
        device_option = st.sidebar.selectbox(
            "Choose the device for the model",
            options=['cpu', 'cuda', 'cuda:0', 'cuda:1'],  # 여기에 필요한 모든 옵션을 추가하세요.
            index=0  # 'cpu'를 기본값으로 설정
        )
        # Streamlit 사이드바  콤보박스 추가
        model_name = st.sidebar.selectbox(
            "Choose the model for OpenAI LLM API",
            options=['gpt-3.5-turbo', 'gpt-3', 'gpt-4','davinci-codex', 'curie'],  # 사용 가능한 모델 이름들
            index=0  # 'gpt-3.5-turbo'를 기본값으로 설정
        )

        process = st.button("Process")
    if process:
        if not openai_api_key:
            openai_api_key = st.secrets["OpenAI_Key"]
            if not openai_api_key:
                st.info("Please add your OpenAI API key to continue.")
                st.stop()
        files_text = get_text(uploaded_files)
        
        display_document_page(files_text)

        text_chunks = get_text_chunks(files_text, chunk_size, chunk_overlap)
        vetorestore = get_vectorstore(text_chunks,device_option)
     
        st.session_state.conversation = get_conversation_chain(vetorestore,openai_api_key,model_name) 

        st.session_state.processComplete = True

    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant", 
                                        "content": "안녕하세요! 주어진 문서에 대해 궁금하신 것이 있으면 언제든 물어봐주세요!"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    history = StreamlitChatMessageHistory(key="chat_messages")

    # Chat logic
    query = st.chat_input("질문을 입력해주세요.")
    if query:
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            chain = st.session_state.conversation

            with st.spinner("Thinking..."):
                result = chain({"question": query})
                with get_openai_callback() as cb:
                    st.session_state.chat_history = result['chat_history']
                response = result['answer']
                source_documents = result['source_documents']

                st.markdown(response)
                with st.expander("참고 문서 확인"):
                    st.markdown(source_documents[0].metadata['source'], help = source_documents[0].page_content)
                    st.markdown(source_documents[1].metadata['source'], help = source_documents[1].page_content)
                    st.markdown(source_documents[2].metadata['source'], help = source_documents[2].page_content)
                    


# Add assistant message to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
def display_document_page(documents):
    # 사용자가 볼 페이지 번호를 선택할 수 있게 합니다.
    # page_number = st.sidebar.selectbox(
    #     "Choose the page number:",
    #     options=range(len(documents)),  # 0부터 문서 수 - 1까지의 번호
    #     format_func=lambda x: f"Page {x + 1}"  # 페이지 번호를 좀 더 친절하게 표시
    # )
    
    # 선택된 페이지의 내용을 보여줍니다.
    for i in range(len(documents)):
        doc = str(documents[i])
        start = doc.find("page_content=") + len("page_content=") +2 
        end = doc.find("metadata=") -2
        extracted_content = doc[start:end]
        # st.write(extracted_content)
        extracted_content = extracted_content.replace('\n','<br>')
        st.markdown(extracted_content,unsafe_allow_html=True)

def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)

def get_text(docs):

    doc_list = []
    
    for doc in docs:
        file_name = doc.name  # doc 객체의 이름을 파일 이름으로 사용
        with open(file_name, "wb") as file:  # 파일을 doc.name으로 저장
            file.write(doc.getvalue())
            logger.info(f"Uploaded {file_name}")
        if '.pdf' in doc.name:
            loader = PyPDFLoader(file_name)
            documents = loader.load_and_split()
        elif '.docx' in doc.name:
            loader = Docx2txtLoader(file_name)
            documents = loader.load_and_split()
        elif '.pptx' in doc.name:
            loader = UnstructuredPowerPointLoader(file_name)
            documents = loader.load_and_split()

        doc_list.extend(documents)
    return doc_list


def get_text_chunks(text, chunk_size=900, chunk_overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=tiktoken_len
    )
    chunks = text_splitter.split_documents(text)
    return chunks


def get_vectorstore(text_chunks, device_option ='cpu'):
    embeddings = HuggingFaceEmbeddings(
                                        model_name="jhgan/ko-sroberta-multitask",
                                        model_kwargs={'device': device_option},
                                        encode_kwargs={'normalize_embeddings': True}
                                        )  
    vectordb = FAISS.from_documents(text_chunks, embeddings)
    return vectordb

def get_conversation_chain(vetorestore,openai_api_key,model_name='gpt-3.5-turbo'):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name = model_name ,temperature=0)
    conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm, 
            chain_type="stuff", 
            retriever=vetorestore.as_retriever(search_type = 'mmr', vervose = True), 
            memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
            get_chat_history=lambda h: h,
            return_source_documents=True,
            verbose = True
        )

    return conversation_chain



if __name__ == '__main__':
    main()