import getpass
import json
import os
from os.path import abspath, exists
from typing import (
    Any,
    List,
    NotRequired,
    Optional,
    TypedDict,
    Union,
    cast,
)

import dotenv
import faiss
from langchain.chat_models import init_chat_model
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import (
    HumanMessage,
    MessageLikeRepresentation,
    SystemMessage,
)
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import ConfigDict

dotenv.load_dotenv()
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")


class LLMResponse:
    content: Union[str, list[Union[str, Any]]]
    additional_kwargs: Any
    response_metadata: Any
    type: str
    name: Optional[str] = None
    id: Optional[str] = None
    model_config = ConfigDict(
        extra="allow",
    )


class Response(TypedDict):
    question: str
    llm_response: LLMResponse
    context_added: NotRequired[Optional[str]]
    expected_response: NotRequired[Optional[str]]


class Payload(TypedDict):
    question: str
    instructions: str
    expected_response: NotRequired[Optional[str]]


class Retriever:
    def __init__(self, file_name_to_embed: str, faiss_db_path: str):
        self._llm = init_chat_model(
            "gpt-4o-mini",
            model_provider="openai",
        )
        self._embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002",
        )
        self._vector_store = FAISS(
            index_to_docstore_id={},
            index=faiss.IndexFlatL2(),
            docstore=InMemoryDocstore(),
            embedding_function=self._embeddings,
        )
        self._reviews_vector_db: FAISS = self._vector_store
        self._generate_embedding(
            faiss_db_path=faiss_db_path,
            file_name_to_embed=file_name_to_embed,
        )

    def invoke(self, payload: Payload):
        question = payload.get("question")
        instructions = payload.get("instructions")
        expected_response = payload.get("expected_response")
        messages: List[MessageLikeRepresentation] = [
            SystemMessage(instructions),
            HumanMessage(question),
        ]
        context_added = self._retrieve(
            messages=messages,
            question=question,
        )
        response = cast(LLMResponse, self._llm.invoke(messages))
        response_dict: Response = {
            "question": question,
            "llm_response": response,
            "context_added": context_added,
            "expected_response": expected_response,
        }
        return response_dict

    def _retrieve(
        self, messages: List[MessageLikeRepresentation], question: str
    ) -> Optional[str]:
        relevant_docs: List[Document] = self._reviews_vector_db.similarity_search(  # type: ignore
            k=3,
            query=question,
            filter={"school_subject": "geography"},
            # score_threshold=0.75,
        )

        return self._prepare_messages_with_context_retrieved(messages, relevant_docs)

    def _prepare_messages_with_context_retrieved(
        self, messages: List[MessageLikeRepresentation], relevant_docs: List[Document]
    ) -> Optional[str]:
        rag_context: Optional[str] = ""
        for doc in relevant_docs:
            rag_context = rag_context + doc.page_content
        messages.append(
            {
                "role": "assistant",
                "content": rag_context,
            }
        )
        return rag_context

    def _generate_embedding(self, faiss_db_path: str, file_name_to_embed: str) -> None:
        if self._should_generate_faiss(faiss_db_path):
            file_path = abspath(file_name_to_embed)
            loader = PyPDFLoader(file_path=file_path)
            file = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=100,
            )
            document_splits = text_splitter.split_documents(file)
            for doc in document_splits:
                metadata: Any = doc.model_dump().get("metadata")
                doc.metadata = {
                    **metadata,
                    "school_subject": "geography",
                }
            self._reviews_vector_db = self._vector_store.from_documents(
                documents=document_splits,
                embedding=self._embeddings,
            )
            self._reviews_vector_db.save_local(faiss_db_path)
        else:
            self._reviews_vector_db = self._vector_store.load_local(
                faiss_db_path,
                embeddings=self._embeddings,
                allow_dangerous_deserialization=True,
            )

    def _should_generate_faiss(self, faiss_db_path: str) -> bool:
        return not exists(faiss_db_path)


file_name_to_embed = "test.pdf"
faiss_db_path = "faiss_index-test"
test = Retriever(
    faiss_db_path=faiss_db_path,
    file_name_to_embed=file_name_to_embed,
)


questions_file_name = "questions.json"
with open(questions_file_name) as f:
    questions_json = json.load(f)["questions"]
for question_json in questions_json:
    question = question_json["pergunta"]
    expected_response = question_json["resposta_esperada"]
    intructions = """
        * Se atenha completamente ao contexto passado, evite acrescentar dados que não foram informados;
        * Seja sucinto, escreva uma resposta direta a pergunta do usuário;
    """

    response = test.invoke(
        {
            "question": question,
            "instructions": intructions,
            "expected_response": expected_response,
        }
    )
    print(
        f"""
            "question": {question},\\
            "llm_response": {response.get("llm_response").content},\\
            "expected_response": {response.get("expected_response")},\\
        """
    )
