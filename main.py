from fastapi import FastAPI
from fastapi import Body, Form, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import json
import nest_asyncio
from pyngrok import ngrok
import uvicorn
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import os
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnableMap
from langchain.prompts import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import Document
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader

load_dotenv()
persist_directory = os.getenv('PERSIST_DIRECTORY')
collection_name = os.getenv('COLLECTION_NAME')
openai_api_key = os.getenv('OPENAI_API_KEY')

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(self.text + "▌")

def setup_chroma(persist_directory, collection_name, openai_api_key):
    # OpenAIEmbeddings 객체 생성
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    # Chroma 데이터베이스 설정
    db = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory
    )

    return db

def get_retriever(db):
    # Chroma 데이터베이스에서 retriever 가져오기
    retriever = db.as_retriever(search_kwargs={"k": 5})
    return retriever

def load_and_store_pdf(db, chunk_size=1000, chunk_overlap=200):
    # PDF 파일 로드
    loader = PyPDFDirectoryLoader('./')
    pages = loader.load_and_split()

    # 텍스트 분할
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    texts = text_splitter.split_documents(pages)

    # Chroma 데이터베이스에 텍스트 삽입
    db.add_documents(texts)

    print(f"Loaded and stored {len(texts)} chunks")

def upload_pdf(db, uploaded_file):
    if uploaded_file is not None:

        filepath = "./uploaded/" + uploaded_file.filename
        with open(filepath, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        
        loader = PyPDFDirectoryLoader(filepath)
        pages = loader.load_and_split()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=50
        )
        texts = text_splitter.split_documents(pages)

        # Chroma 데이터베이스에 텍스트 삽입
        db.add_documents(texts)

        print(f"Loaded and stored {len(texts)} chunks")

async def convert_pdf_to_text(uploaded_file):
    # PDF 파일을 텍스트로 변환하는 함수
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())
    
    loader = PyMuPDFLoader("temp.pdf")
    documents = loader.load()
    text = "\n".join(doc.page_content for doc in documents)
    return text


def get_file_title(file_path):
    base_name = os.path.basename(file_path)  # 파일 이름과 확장자 추출
    file_title = os.path.splitext(base_name)[0]  # 확장자 제거
    return file_title
     

db = setup_chroma(persist_directory, collection_name, openai_api_key)

# PDF 파일 로드 및 Chroma 데이터베이스에 삽입
# load_and_store_pdf(db)

# Retriever 가져오기
retriever = get_retriever(db)

# ChatOpenAI 모델 설정
chat_model = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo")

# 프롬프트 템플릿 설정
template = """
[ChatBot Prompt]
당신은 SafeStep이라는 법률 상담 챗봇 어시스턴트입니다. 사회초년생들이 근로 문제를 해결할 수 있도록 친절하고 전문적인 조언을 제공하는 것이 당신의 역할입니다. 사용자의 질문에 답변할 때는 다음 사항을 고려하세요.

1. 공감과 위로의 말을 전하세요.
2. 관련 법률과 공공기관의 가이드를 바탕으로 정확하고 신뢰할 수 있는 정보를 제공하세요.
3. 구체적이고 실행 가능한 해결 방안을 제시하세요.
4. 사용자의 문제 해결 의지를 격려하고, 필요한 지원과 조언을 아끼지 마세요.
5. SafeStep이 사용자의 든든한 조력자임을 강조하세요.

[Question]
{question}

[Context]
{context}

[Assistant Answer]
{question}에 대해 [관련 법률 및 조항]과 [공공기관 가이드]를 참고하여 안내 드리겠습니다.
문제 해결을 위해서는 다음 절차를 따라주시기 바랍니다:

1.
2.
3.

...

절차 진행 중 어려움이 있으시면, [관련 기관]에 도움을 요청하세요.
증거 자료 준비도 잊지 마시고, 특히 [중요 증거 자료]를 꼭 확보하시기 바랍니다.
더 궁금한 점이 있으시면 언제든 문의해 주세요. SafeStep이 항상 함께 하겠습니다.
"""

prompt = PromptTemplate(template=template, input_variables=["question", "context"])

inputs = RunnableMap({
    'context': lambda x: retriever.get_relevant_documents(x['question']),
    'question': lambda x: x['question']
})
chain = inputs | prompt | chat_model

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


@app.post("/ask")
async def ask_question(question: str = Form(...), file: UploadFile = File(None)):
    pdf_text = ""
    if file is not None:
        pdf_text = convert_pdf_to_text(file)
    combined_question = question + "\n" + pdf_text
    response = chain.invoke({'question': question})
    answer = response.content
    source = retriever.get_relevant_documents(question)

    return {
        "answer": answer,
        "source": source
    }

if __name__ == "__main__":
    # ngrok_tunnel = ngrok.connect(8000)
    # print('Public URL:', ngrok_tunnel.public_url)
    # nest_asyncio.apply()
    uvicorn.run(app, host="0.0.0.0", port=8000)


# # Streamlit 앱
# if 'messages' not in st.session_state:
#     st.session_state.messages = []

# # 제목과 소개
# st.title("SafeStep - 당신의 안전한 첫 걸음을 도와드립니다!")
# st.markdown("""
# **근로 시간 관련 문제로 어려움을 겪고 계신가요?**  
# 저희 SafeStep 챗봇이 도와드립니다! 주 52시간제, 초과 근무 수당 미지급, 포괄임금제 문제 등 다양한 근로 시간 관련 문제를 해결하기 위한 법적 조언과 가이드를 제공합니다.  
# **지금 바로 질문해보세요!** 우리의 AI 상담원이 신속하고 정확하게 답변해드립니다.
# """)

# # PDF 파일 업로드 섹션
# uploaded_file = st.file_uploader("문제를 겪고 있는 문서를 업로드하세요 📄", type="pdf")

# if uploaded_file is not None:
#     upload_pdf(db, uploaded_file)
#     st.success("파일이 성공적으로 업로드되었습니다. 🔥 이제 질문을 입력해보세요!")

# # 채팅 메시지 표시
# for message in st.session_state.messages:
#     st.chat_message(message['role']).markdown(message['content'])

# # 질문 입력 섹션
# if question := st.chat_input("메시지를 입력하세요 📝:"):
#     st.session_state.messages.append({"role": "human", "content": question})

#     with st.chat_message('human'):
#         st.markdown(question)

#     with st.chat_message('assistant'):
#         response_placeholder = st.empty()

#     response = chain.invoke({'question': question}, config={'callbacks': [StreamHandler(response_placeholder)]})
#     answer = response.content

#     st.session_state.messages.append({"role": "ai", "content": answer})
#     response_placeholder.markdown(answer)

#     docs = retriever.get_relevant_documents(question)
#     if docs:
#         with st.expander("🔍 참고 자료 확인"):
#             for i, doc in enumerate(docs, start=1):
#                 st.markdown(f"##### 출처 {i}")
#                 st.markdown(f"- {get_file_title(doc.metadata['source'])} / {doc.metadata['page']}p")
#                 st.markdown("##### 내용")
#                 st.markdown(doc.page_content)
#                 st.markdown("---")


# # 입력값에 대한 답변 생성
# while True:
#     query = input("Ask a question (or type 'exit' to quit): ")
#     if query.lower() == "exit":
#         break

#     response = chain.invoke({'question': query})
#     answer = response.content

#     print(answer)

#     docs = retriever.get_relevant_documents(query)

#     sources = "\nSources:\n"
#     for doc in docs:
#         sources += f"- {doc.metadata['source']} {doc.metadata['page']} page\n"
#         sources += f"  Content: {doc.page_content}\n"

#     print(sources)
