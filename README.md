# SafeStep

SafeStep은 사회초년생들이 근로 관련 문제를 해결할 수 있도록 돕는 법률 상담 서비스입니다. 주 52시간제 위반, 초과 근무 수당 미지급, 포괄임금제 오남용, 수습 기간 중 부당 해고, 임금 체불 문제 등 다양한 근로 문제에 대해 실시간 법적 조언과 가이드를 제공합니다.

## 주요 기능

- **실시간 법적 조언 제공**: 사용자가 질문을 입력하면, 챗봇이 관련 법적 조언과 절차에 대한 가이드를 실시간으로 제공합니다.
- **관련 법률 문서 제공**: 근로기준법, 고용노동부 가이드, 판례 등의 참고 자료를 제공합니다.
- **사용자 친화적인 인터페이스**: 언제 어디서나 접근 가능한 실시간 상담 챗봇을 통해 즉각적인 법적 지원을 제공합니다.

## 기술 스택

- **Backend**: FastAPI
- **Database**: ChromaDB (벡터 데이터베이스)
- **Embeddings**: OpenAIEmbeddings
- **LLM**: OpenAI GPT-3.5-turbo
- **Framework**: LangChain

## 설치 및 실행 방법

### 로컬 환경에서 실행하기

1. **저장소 클론**
   ```bash
   git clone https://github.com/yourusername/safestep.git
   cd safestep
   ```
2. **가상환경 설정 및 패키지 설치**
    ```
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```
3. **환경 변수 설정**
    ```
    PERSIST_DIRECTORY=./db
    COLLECTION_NAME=safestep
    OPENAI_API_KEY=your_openai_api_key
    ```
4. **서버 실행**
    ```
    uvicorn main:app --reload
    ```

## 사용 방법
1. **질문 입력**

- 웹 인터페이스에서 질문을 입력합니다. 예를 들어, "초과 근무 수당은 어떻게 계산되나요?"와 같은 질문을 입력할 수 있습니다.

2. **파일 첨부 (선택 사항)**
- 관련 법률 문서나 자료를 첨부할 수 있습니다. 첨부된 파일은 자동으로 처리되어 질문과 함께 분석됩니다.

3. **결과 확인**
- 실시간으로 제공되는 법적 조언과 관련 법률 문서를 확인합니다. 필요한 경우 관련 기관에 도움을 요청할 수 있는 절차도 안내됩니다.

## 주의사항
- ./content 디렉토리에 참고할 법률 문서 파일들이 있어야 합니다. 이는 근로기준법, 고용노동부 가이드, 판례 및 사례집, 노동청 자료 등입니다.
- 시스템에 sqlite가 설치되어 있어야 합니다. 설치하려면 다음 명령어를 사용하세요:
```bash
sudo apt-get install sqlite3
```


## 주요 클래스 및 함수 설명
- main.py
    - ask_question: 사용자 질문을 받아 처리하고, 관련 법률 문서와 함께 답변을 생성하여 반환합니다.
    - convert_pdf_to_text: 업로드된 PDF 파일을 텍스트로 변환합니다.
    - setup_chroma: ChromaDB를 설정하고 초기화합니다.
    - get_retriever: ChromaDB에서 관련 정보를 검색하는 함수입니다.

## 참고 자료
- 근로기준법
- 고용노동부 가이드
- 판례 및 사례집
- 노동청 자료
