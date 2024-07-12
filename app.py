import streamlit as st
import time
from PIL import Image
#from chatgpt_llm import graph_chain
import asyncio
import os
import sys
from operator import itemgetter
from typing import List, Tuple, Dict
import streamlit as st

from langchain.schema import HumanMessage, AIMessage, BaseMessage, Document
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    format_document,
)
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAI
# from langchain_community.vectorstores.faiss import FAISS
from langchain.vectorstores.chroma import Chroma
sys.modules['sqlite3'] = import('pysqlite3')
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from operator import itemgetter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.graph import END, StateGraph
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory

from typing_extensions import TypedDict

##### API 
api_key = st.secrets['OPENAI_API_KEY']
tavily_key = st.secrets['TAVILY_API_KEY']

##### Knowledge Base
embed_model = OpenAIEmbeddings(api_key=api_key)
# vector_index = FAISS.load_local("./carinfo/faiss_chatgpt.json", embeddings=embed_model, allow_dangerous_deserialization=True)
vector_index = Chroma(embedding_function=embed_model,
                      persist_directory="./renault_chroma",
                      collection_name="rag")
retriever = vector_index.as_retriever(search_type="similarity", search_kwargs={"k": 10})


_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""  # noqa: E501
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)


##### Agent Template
# 프롬프트 텍스트 파일 경로
template_gaingenie = "./templates/gaingenie.txt"
template_connecto  = "./templates/connecto.txt"
template_infomaster = "./templates/infomaster.txt"

# 파일에서 프롬프트 텍스트 읽어오기
def load_prompt_template(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        return file.read()

# 프롬프트 템플릿 불러오기
prompt_gaingenie = load_prompt_template(template_gaingenie)
prompt_connecto = load_prompt_template(template_connecto)
prompt_infomaster = load_prompt_template(template_infomaster)


DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")

def _combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)

def _format_chat_history(chat_history: List[Dict[str, str]]) -> List[BaseMessage]:
    if len(chat_history) > 6:
        chat_history = chat_history[-6:]
    buffer = []
    for entry in chat_history:
        if entry['role'] == 'user':
            buffer.append(HumanMessage(content=entry['content']))
        elif entry['role'] == 'assistant':
            buffer.append(AIMessage(content=entry['content']))
    return buffer

# User input
class ChatHistory(BaseModel):
    chat_history: List[Dict[str, str]] = Field(..., extra={"widget": {"type": "chat"}})
    question: str

_search_query = RunnableBranch(
    # If input includes chat_history, we condense it with the follow-up question
    (
        RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
            run_name="HasChatHistoryCheck"
        ),  # Condense follow-up question and chat into a standalone_question
        RunnablePassthrough.assign(
            chat_history=lambda x: _format_chat_history(x["chat_history"])
        )
        | CONDENSE_QUESTION_PROMPT
        | ChatOpenAI(temperature=0)
        | StrOutputParser(),
    ),
    # Else, we have no chat history, so just pass through the question
    RunnableLambda(itemgetter("question")),
)

_inputs = RunnableParallel(
    {
        "question": lambda x: x["question"],
        "chat_history": lambda x: _format_chat_history(x["chat_history"]),
        "context": _search_query | retriever,
    }
).with_types(input_type=ChatHistory)

###### LLM
# Data model (데이터 판단)
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


# LLM with function call
llm = ChatOpenAI(model="gpt-4o", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeDocuments)

# Prompt
system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader
# question = "르노 자동차"
# docs = retriever.get_relevant_documents(question)
# doc_txt = docs[1].page_content
# print(retrieval_grader.invoke({"question": question, "document": doc_txt}))

### Generate (문장 생성)

# Prompt
# prompt = ChatPromptTemplate.from_template(template_rag)
prompt_g= ChatPromptTemplate.from_messages(
    [
        ("system", prompt_gaingenie),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{question}"),
    ]
)
prompt_c= ChatPromptTemplate.from_messages(
    [
        ("system", prompt_connecto),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{question}"),
    ]
)
prompt_i= ChatPromptTemplate.from_messages(
    [
        ("system", prompt_infomaster),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{question}"),
    ]
)

# LLM
answer_llm = ChatOpenAI(model_name="gpt-4o", temperature=0.1, streaming=True)

# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# Chain
rag_chain_g = prompt_g | answer_llm | StrOutputParser()
rag_chain_c = prompt_c | answer_llm | StrOutputParser()
rag_chain_i = prompt_i | answer_llm | StrOutputParser()

branch = RunnableBranch(
    (lambda x: "게인지니" in x["agent"].lower(), rag_chain_g), # a list of (condition, runnable) pair
    (lambda x: "코넥토" in x["agent"].lower(), rag_chain_c), 
    (lambda x: "인포마스터" in x["agent"].lower(), rag_chain_i),
    rag_chain_c
)

# Run
# generation = rag_chain.invoke({"context": docs, "question": question})
# print(generation)

##### Question Re-writer (질문 생성기)

# LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Prompt
system = """You a question re-writer that converts an input question to a better version that is optimized \n 
    for web search. Most of question is about the car. Look at the input and try to reason about the underlying semantic intent / meaning. You havet to rewrite in Korean."""
re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "Here is the initial question: \n\n {question} \n Formulate an improved question.",
        ),
    ]
)

question_rewriter = re_write_prompt | llm | StrOutputParser()
# question_rewriter.invoke({"question": question})

##### Web Search (웹 검색)

web_search_tool = TavilySearchResults(k=3)

# Memory integration
# memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
memory = ConversationSummaryMemory(llm=OpenAI(temperature=0), memory_key="chat_history", return_messages=True)

##### Langgraph (그래프 구성)
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
        conversation: list of conversation history
    """

    question: str
    generation: str
    web_search: str
    agent: str
    documents: List[str]
    chat_history: List[tuple]

### Node(노드) 
def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    #print("---RETRIEVE---")
    question = state["question"]
    chat_history = state["chat_history"]

    #print(chat_history)

    # Retrieval
    result = _inputs.invoke({"question":question, "chat_history":chat_history})

    return {"documents": result['context'], "question": result['question'], "chat_history":result['chat_history']}


def generate(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    #print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    chat_history = state["chat_history"]
    agent = state["agent"]
    #print(chat_history)

    # RAG generation
    generation = branch.invoke({"context": _combine_documents(documents), "question": question, "chat_history": chat_history, "agent": agent})
    return {"documents": documents, "question": question, "generation": generation}


def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """

    #print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    # Score each doc
    filtered_docs = []
    web_search = "No"
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score.binary_score
        if grade == "yes":
            #print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            #print("---GRADE: DOCUMENT NOT RELEVANT---")
            web_search = "Yes"
            continue
    return {"documents": filtered_docs, "question": question, "web_search": web_search}


def transform_query(state):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """

    #print("---TRANSFORM QUERY---")
    question = state["question"]
    documents = state["documents"]

    # Re-write question
    better_question = question_rewriter.invoke({"question": question})
    #print(f"Better Question: {better_question}")
    return {"documents": documents, "question": better_question}


def web_search(state):
    """
    Web search based on the re-phrased question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """

    #print("---WEB SEARCH---")
    question = state["question"]
    documents = state["documents"]

    # Web search
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    documents.append(web_results)

    #print(f"Result: {documents}")

    return {"documents": documents, "question": question}


### Edges
def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    #print("---ASSESS GRADED DOCUMENTS---")
    question = state["question"]
    web_search = state["web_search"]
    filtered_documents = state["documents"]

    if web_search == "Yes":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        # print(
        #    "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
        # )
        return "transform_query"
    else:
        # We have relevant documents, so generate answer
        # print("---DECISION: GENERATE---")
        return "generate"
    

##### 그래프 완성

workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)  # generatae
workflow.add_node("transform_query", transform_query)  # transform_query
workflow.add_node("web_search_node", web_search)  # web search

# Build graph
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)
workflow.add_edge("transform_query", "web_search_node")
workflow.add_edge("web_search_node", "generate")
workflow.add_edge("generate", END)

# Compile
graph_chain = workflow.compile()

def image_to_base64(image):
    import base64
    from io import BytesIO

    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# 챗봇 아바타 이미지 불러오기
chatbot_avatar = Image.open("./images/salesguy.jpg")
chatbot_avatar_base64 = image_to_base64(chatbot_avatar)

# 초기 질문용 아바타 이미지 불러오기
chatbot_initial_avatar = Image.open("./images/logo.png")
chatbot_initial_avatar_base64 = image_to_base64(chatbot_initial_avatar)

async def typing_effect(message_placeholder):
    typing_text = "에이전트가 답변을 작성 중입니다"
    while st.session_state.awaiting_response:
        st.session_state.typing_index = (st.session_state.typing_index + 1) % 6
        message_placeholder.markdown(
            f"<div class='assistant-message'>"
            f"<img src='data:image/png;base64,{chatbot_avatar_base64}' class='assistant-avatar'>"
            f"<div><div class='assistant-name'>피카지니</div>"
            f"<div class='chatbox typing'>{typing_text}{'.' * st.session_state.typing_index}</div>"
            f"</div></div>",
            unsafe_allow_html=True
        )
        await asyncio.sleep(0.3)

async def fetch_answer_and_typing(inputs, message_placeholder):
    loop = asyncio.get_event_loop()
    future_result = loop.run_in_executor(None, graph_chain.invoke, inputs)
    
    typing_task = asyncio.create_task(typing_effect(message_placeholder))

    result = await future_result
    typing_task.cancel()

    try:
        await typing_task
    except asyncio.CancelledError:
        pass

    final_answer = result.get("generation", "Sorry, I couldn't find an answer.")
    st.session_state.final_answer = final_answer

    # 최종 답변을 placeholder에 스트리밍
    message_text = ""
    for char in final_answer:
        message_text += char
        message_placeholder.markdown(
            f"<div class='assistant-message'>"
            f"<img src='data:image/png;base64,{chatbot_avatar_base64}' class='assistant-avatar'>"
            f"<div><div class='assistant-name'>피카지니</div>"
            f"<div class='chatbox'>{message_text}</div>"
            f"</div></div>",
            unsafe_allow_html=True
        )
        await asyncio.sleep(0.02)  # 여기서 타이핑 속도를 조정할 수 있음

def main():
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "awaiting_response" not in st.session_state:
        st.session_state.awaiting_response = False

    if "typing_index" not in st.session_state:
        st.session_state.typing_index = 0

    if "final_answer" not in st.session_state:
        st.session_state.final_answer = ""

    if "initial_choice_made" not in st.session_state:
        st.session_state.initial_choice_made = False
        st.session_state.agent = None

    st.markdown("""
    <style>
        body {
            background-color: #f5f5f5;
        }
        .stChatMessageUser {
            display: none !important;
        }
        .chatbox {
            border: none;
            border-radius: 15px;
            padding: 10px;
            display: inline-block;
            word-wrap: break-word;
            max-width: 80%;
            margin-bottom: 10px;
            background-color: #f0f0f0;
        }
        .initial-question {
            background-color: #ffffff;
            border: none;
            text-align: left;
            float: left;
            clear: both;
            border-radius: 15px;
            display: flex;
            align-items: center;
            padding: 10px;
            margin-bottom: 10px;
        }
        .typing {
            min-width: 250px;
            min-height: 40px;
            line-height: 1.5;
        }
        .user-message {
            background-color: #d1e7dd;
            border: none;
            text-align: right;
            float: right;
            clear: both;
            border-radius: 15px;
        }
        .assistant-message {
            background-color: #ffffff;
            border: none;
            text-align: left;
            float: left;
            clear: both;
            border-radius: 15px;
            display: flex;
            align-items: center;
        }
        .assistant-avatar {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            margin-right: 10px;
        }
        .assistant-name {
            font-weight: bold;
            margin-bottom: 5px;
        }
        .chat-container {
            margin: 0 auto;
            max-width: 800px;
        }
        .choice-button {
            display: block;
            width: 100%;
            padding: 10px;
            margin: 10px 0;
            font-size: 18px;
            text-align: center;
            background-color: #007bff;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for message in st.session_state.messages:
        if message["role"] == "assistant":
            st.markdown(f"""
            <div class="assistant-message">
                <img src='data:image/png;base64,{chatbot_avatar_base64}' class='assistant-avatar'>
                <div>
                    <div class="assistant-name">피카지니</div>
                    <div class="chatbox">{message['content']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chatbox user-message'>{message['content']}</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if not st.session_state.initial_choice_made:
        initial_question = """안녕하세요, 언제나 최상의 차량 정보를 제공하는 에피카입니다. 저희와 함께해 주셔서 감사합니다. 차량에 대해 궁금한 점이나 도움이 필요하시면 언제든지 말씀해 주세요. 오늘은 어떤 정보가 필요하신가요?"""
        # st.session_state.messages.append({"role": "assistant", "content": initial_question})
        st.markdown(f"""
        <div class="assistant-message initial-question">
            <img src='data:image/png;base64,{chatbot_initial_avatar_base64}' class='assistant-avatar'>
            <div>
                <div class="assistant-name">EPIKAR</div>
                <div class="chatbox">{initial_question}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("정확한 차량의 스펙과 성능 정보를 알려주세요.", key="specs"):
            choice = "정확한 차량의 스펙과 성능 정보"
            agent = '인포마스터'
        elif st.button("자동차 딜러의 의견과 이야기가 듣고 싶어요.", key="opinions"):
            choice = "딜러의 의견과 이야기"
            agent = '코넥토'
        elif st.button("저를 위한 자동차 구매 혜택과 특별 프로모션도 궁금해요", key="promotions"):
            choice = "나를 위한 구매 혜택과 특별 프로모션"
            agent = '게인지니'
        else:
            choice = None
            agent = None

        if choice:
            st.session_state.initial_choice_made = True
            st.session_state.choice = choice
            st.session_state.agent = agent

            follow_up_message = (f"안녕하세요, 저는 피카지니 입니다. 고객님께 최고의 딜을 제공해드리기 위해 여기 있습니다. \n"
                                 "고객님의 나이와 차량 구매 목적을 말씀해주시면 더 나은 추천을 드릴 수 있습니다. \n "
                                 "또한, 현대, 기아, 르노 중에서 관심 있는 브랜드가 있으면 알려주세요.")
            st.session_state.messages.append({"role": "assistant", "content": follow_up_message})

            st.rerun()

    if st.session_state.initial_choice_made:
        prompt = st.chat_input("메세지를 입력해 주세요:")
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.markdown(f"<div class='chatbox user-message'>{prompt}</div>", unsafe_allow_html=True)

            st.session_state.awaiting_response = True
            st.session_state.prompt = prompt

            message_placeholder = st.empty()

            asyncio.run(fetch_answer_and_typing({"question": prompt, 'chat_history': st.session_state.messages, 'agent': st.session_state.agent}, message_placeholder))

            message_placeholder.markdown(
                f"<div class='assistant-message'>"
                f"<img src='data:image/png;base64,{chatbot_avatar_base64}' class='assistant-avatar'>"
                f"<div><div class='assistant-name'>피카지니</div>"
                f"<div class='chatbox'>{st.session_state.final_answer}</div>"
                f"</div></div>",
                unsafe_allow_html=True
            )
            st.session_state.messages.append({"role": "assistant", "content": st.session_state.final_answer})
            st.session_state.awaiting_response = False

if __name__ == "__main__":
    main()
