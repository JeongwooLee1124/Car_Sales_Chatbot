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
from langchain_community.vectorstores.chroma import Chroma
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
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

##### API 
api_key = st.secrets['OPENAI_API_KEY']
tavily_key = st.secrets['TAVILY_API_KEY']

##### Knowledge Base
embed_model = OpenAIEmbeddings(api_key=api_key)
vector_index = Chroma(embedding_function=embed_model,
                      persist_directory="renault_chroma",
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

if __name__ == "__main__":
    ## Run
    inputs = {"question": "30대 직장인 남성 출퇴근용 자동차를 추천해주세요.", "chat_history":[]}

    ## Final generation
    ## print(graph_chain.invoke(inputs))