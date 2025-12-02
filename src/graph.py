"""
Grafo de Agentes para RAG ENEM usando LangGraph.
Define o estado do sistema e orquestra o fluxo de processamento.
"""

import os
from typing import Dict, List, Any, TypedDict, Literal
from langchain.schema import Document
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Carregar variáveis do .env
load_dotenv()

from agents import (
    retrieve_documents,
    generate_answer,
    check_groundedness,
    apply_safety_layer
)


class ENEMRAGState(TypedDict):
    """
    Estado que flui através do grafo de agentes.
    Contém todas as informações necessárias para o processamento RAG.
    """
    # Entrada do usuário
    question: str
    
    # Documentos recuperados
    documents: List[Dict[str, Any]]
    
    # Resposta gerada pelo LLM
    answer: str
    
    # Verificação se a resposta está bem fundamentada
    is_grounded: bool
    
    # Resposta final com disclaimer
    final_response: str
    
    # Metadados adicionais
    metadata: Dict[str, Any]


class ENEMRAGGraph:
    """
    Classe principal que constrói e gerencia o grafo de agentes RAG.
    """
    
    def __init__(
        self,
        collection_name: str = None,
        embedding_model: str = None,
        llm_model: str = None,
        use_cloud: bool = None,
        chroma_api_key: str = None,
        chroma_tenant: str = None,
        chroma_database: str = None,
        max_documents: int = None,
        google_api_key: str = None
    ):
        # Usar configurações do .env como padrão, permitindo override
        self.collection_name = collection_name or os.getenv('COLLECTION_NAME', 'enem_documents')
        self.embedding_model = embedding_model or os.getenv('EMBEDDING_MODEL', 'BAAI/bge-m3')
        self.llm_model = llm_model or os.getenv('GEMINI_MODEL', 'gemini-pro')
        self.use_cloud = use_cloud if use_cloud is not None else os.getenv('USE_CHROMA_CLOUD', 'true').lower() == 'true'
        self.chroma_api_key = chroma_api_key or os.getenv('CHROMA_API_KEY')
        self.chroma_tenant = chroma_tenant or os.getenv('CHROMA_TENANT')
        self.chroma_database = chroma_database or os.getenv('CHROMA_DATABASE', 'enem_rag')
        self.max_documents = max_documents or int(os.getenv('MAX_DOCUMENTS', '5'))
        self.google_api_key = google_api_key or os.getenv('GOOGLE_API_KEY')
        
        # Validar API key do Google
        if not self.google_api_key:
            raise ValueError("GOOGLE_API_KEY é obrigatória! Configure no arquivo .env")
        
        # Configurar LLM com Google Gemini
        self.llm = ChatGoogleGenerativeAI(
            model=self.llm_model,
            temperature=float(os.getenv('LLM_TEMPERATURE', '0.1')),
            google_api_key=self.google_api_key,
            convert_system_message_to_human=True
        )
        
        print(f"🤖 LLM configurado: {self.llm_model} (Google Gemini)")
        
        # Construir o grafo
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """
        Constrói o grafo de estados com todos os nós e arestas.
        
        Returns:
            Grafo compilado pronto para execução
        """
        # Criar grafo de estados
        workflow = StateGraph(ENEMRAGState)
        
        # Adicionar nós ao grafo
        workflow.add_node("retrieve_documents", retrieve_documents)
        workflow.add_node("generate_answer", generate_answer)
        workflow.add_node("check_groundedness", check_groundedness)
        workflow.add_node("apply_safety_layer", apply_safety_layer)
        
        # Definir ponto de entrada
        workflow.set_entry_point("retrieve_documents")
        
        # Definir arestas sequenciais
        workflow.add_edge("retrieve_documents", "generate_answer")
        workflow.add_edge("generate_answer", "check_groundedness")
        
        # Aresta condicional após verificação de fundamentação
        workflow.add_conditional_edges(
            "check_groundedness",
            self._route_after_groundedness_check,
            {
                "apply_safety": "apply_safety_layer",
                "end_ungrounded": END
            }
        )
        
        # Finalizar após aplicar camada de segurança
        workflow.add_edge("apply_safety_layer", END)
        
        # Compilar o grafo
        return workflow.compile()
    
    def _route_after_groundedness_check(self, state: ENEMRAGState) -> Literal["apply_safety", "end_ungrounded"]:
        """
        Função de roteamento após verificação de fundamentação.
        
        Args:
            state: Estado atual do grafo
            
        Returns:
            Próxima ação baseada na verificação de fundamentação
        """
        if state.get("is_grounded", False):
            return "apply_safety"
        else:
            return "end_ungrounded"
    
    def invoke(self, question: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Executa o pipeline RAG completo para uma pergunta.
        
        Args:
            question: Pergunta do usuário
            metadata: Metadados adicionais (filtros, configurações)
            
        Returns:
            Estado final com resposta processada
        """
        # Estado inicial
        initial_state = {
            "question": question,
            "documents": [],
            "answer": "",
            "is_grounded": False,
            "final_response": "",
            "metadata": metadata or {}
        }
        
        try:
            # Executar o grafo
            final_state = self.graph.invoke(initial_state)
            return final_state
            
        except Exception as e:
            # Em caso de erro, retornar resposta de fallback
            return {
                "question": question,
                "documents": [],
                "answer": "",
                "is_grounded": False,
                "final_response": f"Desculpe, ocorreu um erro ao processar sua pergunta: {str(e)}",
                "metadata": {"error": str(e)}
            }
    
    def stream(self, question: str, metadata: Dict[str, Any] = None):
        """
        Executa o pipeline RAG com streaming dos estados intermediários.
        
        Args:
            question: Pergunta do usuário
            metadata: Metadados adicionais
            
        Yields:
            Estados intermediários do processamento
        """
        # Estado inicial
        initial_state = {
            "question": question,
            "documents": [],
            "answer": "",
            "is_grounded": False,
            "final_response": "",
            "metadata": metadata or {}
        }
        
        try:
            # Executar o grafo com streaming
            for state in self.graph.stream(initial_state):
                yield state
                
        except Exception as e:
            yield {
                "error": {
                    "question": question,
                    "final_response": f"Erro durante processamento: {str(e)}",
                    "metadata": {"error": str(e)}
                }
            }


def create_enem_rag_graph(**kwargs) -> ENEMRAGGraph:
    """
    Factory function para criar o grafo RAG do ENEM.
    Usa configurações do .env por padrão, permitindo override via parâmetros.
    
    Args:
        **kwargs: Argumentos opcionais para override das configurações
        
    Returns:
        Instância configurada do grafo RAG
    """
    return ENEMRAGGraph(**kwargs)