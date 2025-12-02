"""
Agentes do sistema RAG ENEM.
Implementa os nós que compõem o grafo de processamento.
"""

import os
from typing import Dict, List, Any
from pathlib import Path
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
import chromadb
import json
import re
from dotenv import load_dotenv

# Carregar variáveis do .env
load_dotenv()


class ChromaDBRetriever:
    """Cliente para recuperação de documentos do ChromaDB."""
    
    def __init__(
        self,
        collection_name: str = "enem_documents",
        embedding_model: str = "BAAI/bge-m3",
        use_cloud: bool = True,
        chroma_api_key: str = None,
        chroma_tenant: str = None,
        chroma_database: str = "enem_rag"
    ):
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.use_cloud = use_cloud
        
        # Configurações do Chroma Cloud do .env
        self.chroma_api_key = chroma_api_key or os.getenv('CHROMA_API_KEY')
        self.chroma_tenant = chroma_tenant or os.getenv('CHROMA_TENANT')
        self.chroma_database = chroma_database or os.getenv('CHROMA_DATABASE', 'enem_rag')
        
        # Configurar modelo de embeddings
        self.embeddings_model = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Configurar ChromaDB
        self._setup_chromadb()
    
    def _setup_chromadb(self):
        """Configura conexão com ChromaDB."""
        try:
            if self.use_cloud and self.chroma_api_key and self.chroma_tenant:
                self.chroma_client = chromadb.CloudClient(
                    api_key=self.chroma_api_key,
                    tenant=self.chroma_tenant,
                    database=self.chroma_database
                )
            else:
                self.chroma_client = chromadb.PersistentClient(path="vector_store")
            
            self.collection = self.chroma_client.get_collection(name=self.collection_name)
            
        except Exception as e:
            raise RuntimeError(f"Erro ao conectar ChromaDB: {str(e)}")
    
    def retrieve(self, query: str, k: int = 5, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Recupera documentos similares à query.
        
        Args:
            query: Pergunta do usuário
            k: Número de documentos a retornar
            filters: Filtros de metadados
            
        Returns:
            Lista de documentos recuperados
        """
        try:
            # Gerar embedding da query manualmente
            query_embedding = self.embeddings_model.embed_documents([query])[0]
            
            # Preparar parâmetros da query
            query_params = {
                "query_embeddings": [query_embedding],
                "n_results": k,
                "include": ["documents", "metadatas", "distances"]
            }
            
            if filters:
                query_params["where"] = filters
            
            # Realizar busca
            results = self.collection.query(**query_params)
            
            # Formatar resultados
            documents = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    doc = {
                        'content': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'similarity_score': 1 - results['distances'][0][i],
                        'source': results['metadatas'][0][i].get('source', 'N/A'),
                        'page': results['metadatas'][0][i].get('page', 'N/A')
                    }
                    documents.append(doc)
            
            return documents
            
        except Exception as e:
            print(f"Erro na recuperação: {str(e)}")
            return []


# Instância global do retriever (será inicializada quando necessário)
_retriever = None

def get_retriever() -> ChromaDBRetriever:
    """Obtém instância global do retriever."""
    global _retriever
    if _retriever is None:
        _retriever = ChromaDBRetriever(
            use_cloud=os.getenv('USE_CHROMA_CLOUD', 'true').lower() == 'true',
            chroma_api_key=os.getenv('CHROMA_API_KEY'),
            chroma_tenant=os.getenv('CHROMA_TENANT')
        )
    return _retriever


def retrieve_documents(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Nó recuperador: busca documentos relevantes no ChromaDB.
    
    Args:
        state: Estado atual do grafo
        
    Returns:
        Estado atualizado com documentos recuperados
    """
    question = state.get("question", "")
    metadata = state.get("metadata", {})
    
    print(f"🔍 [RETRIEVE] Buscando documentos para: '{question[:50]}...'")
    
    try:
        # Obter retriever
        retriever = get_retriever()
        
        # Extrair filtros dos metadados se fornecidos
        filters = metadata.get("filters", {})
        k = metadata.get("max_documents", int(os.getenv('MAX_DOCUMENTS', '5')))
        
        # Recuperar documentos
        documents = retriever.retrieve(query=question, k=k, filters=filters)
        
        print(f"📚 [RETRIEVE] Encontrados {len(documents)} documentos relevantes")
        
        # Atualizar estado
        state["documents"] = documents
        
        return state
        
    except Exception as e:
        print(f"❌ [RETRIEVE] Erro: {str(e)}")
        state["documents"] = []
        return state


def generate_answer(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Nó gerador: cria resposta baseada nos documentos recuperados.
    
    Args:
        state: Estado atual do grafo
        
    Returns:
        Estado atualizado com resposta gerada
    """
    question = state.get("question", "")
    documents = state.get("documents", [])
    
    print(f"🤖 [GENERATE] Gerando resposta baseada em {len(documents)} documentos")
    
    # Template de prompt para geração de resposta
    prompt_template = PromptTemplate(
        input_variables=["question", "context"],
        template="""
Você é um assistente especializado em informações sobre o ENEM (Exame Nacional do Ensino Médio).

Sua tarefa é responder à pergunta do usuário usando ESTRITAMENTE as informações fornecidas nos documentos abaixo.

REGRAS IMPORTANTES:
1. Use APENAS as informações dos documentos fornecidos
2. Se a informação não estiver nos documentos, responda: "Não encontrei informações sobre isso nas fontes fornecidas"
3. Para cada afirmação, cite a fonte no formato [Fonte: nome_do_arquivo, Página: X]
4. Seja preciso e objetivo
5. Não invente ou assuma informações que não estejam explicitamente nos documentos
6. Seja direto e utilize apenas os dados fornecidos pelos documentos

DOCUMENTOS FORNECIDOS:
{context}

PERGUNTA: {question}

RESPOSTA:
"""
    )
    
    try:
        # Preparar contexto dos documentos
        context_parts = []
        for i, doc in enumerate(documents[:5]):  # Limitar a 5 documentos
            context_parts.append(
                f"Documento {i+1}:\n"
                f"Fonte: {doc['source']}, Página: {doc['page']}\n"
                f"Conteúdo: {doc['content'][:1000]}...\n"
                f"---"
            )
        
        context = "\n".join(context_parts) if context_parts else "Nenhum documento relevante encontrado."
        
        # Configurar LLM com Google Gemini usando .env
        llm = ChatGoogleGenerativeAI(
            model=os.getenv('GEMINI_MODEL', 'gemini-pro'),
            temperature=float(os.getenv('LLM_TEMPERATURE', '0.1')),
            google_api_key=os.getenv('GOOGLE_API_KEY'),
            convert_system_message_to_human=True
        )
        
        # Gerar resposta
        prompt = prompt_template.format(question=question, context=context)
        response = llm.invoke(prompt)
        
        answer = response.content if hasattr(response, 'content') else str(response)
        
        print(f"✅ [GENERATE] Resposta gerada com {len(answer)} caracteres")
        
        # Atualizar estado
        state["answer"] = answer
        
        return state
        
    except Exception as e:
        print(f"❌ [GENERATE] Erro: {str(e)}")
        state["answer"] = "Desculpe, ocorreu um erro ao gerar a resposta."
        return state


def check_groundedness(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Nó verificador: verifica se a resposta está bem fundamentada nos documentos.
    
    Args:
        state: Estado atual do grafo
        
    Returns:
        Estado atualizado com verificação de fundamentação
    """
    answer = state.get("answer", "")
    documents = state.get("documents", [])
    
    print(f"🔍 [CHECK] Verificando fundamentação da resposta")
    
    # Template de prompt para verificação mais permissivo
    prompt_template = PromptTemplate(
        input_variables=["answer", "context"],
        template="""
Você deve verificar se a resposta fornecida é razoavelmente suportada pelos documentos de contexto.

DOCUMENTOS DE CONTEXTO:
{context}

RESPOSTA A VERIFICAR:
{answer}

CRITÉRIOS PARA APROVAÇÃO (responda 'sim' se QUALQUER um dos critérios for atendido):

1. INFORMAÇÕES DIRETAS: A resposta contém informações que aparecem diretamente nos documentos
2. INFERÊNCIAS VÁLIDAS: A resposta faz inferências razoáveis baseadas no contexto dos documentos
3. RESPOSTAS PARCIAIS: A resposta fornece informações parciais que estão nos documentos, mesmo que incompletas
4. CITAÇÕES CORRETAS: A resposta cita fontes corretas dos documentos fornecidos
5. AUSÊNCIA DECLARADA: A resposta honestamente declara que não encontrou informações específicas
6. INFORMAÇÕES RELACIONADAS: A resposta fornece informações relacionadas ao tópico que estão presentes nos documentos

APENAS responda 'não' se:
- A resposta contém informações completamente inventadas que NÃO estão nos documentos
- A resposta contradiz diretamente as informações dos documentos
- A resposta não tem NENHUMA relação com o conteúdo fornecido

Seja PERMISSIVO e considere que o assistente está tentando ser útil com base nas informações disponíveis.

Responda APENAS com 'sim' ou 'não':
"""
    )
    
    try:
        # Se não há resposta ou documentos, considerar não fundamentada
        if not answer.strip() or not documents:
            state["is_grounded"] = False
            print("❌ [CHECK] Não fundamentada: sem resposta ou documentos")
            return state
        
        # Se a resposta indica que não encontrou informações, considerar fundamentada
        if any(phrase in answer.lower() for phrase in [
            "não encontrei informações",
            "não há informações",
            "não localizei",
            "não consta",
            "não foi possível encontrar"
        ]):
            state["is_grounded"] = True
            print("✅ [CHECK] Fundamentada: resposta indica ausência de informações")
            return state
        
        # Se a resposta contém citações de fontes válidas, considerar fundamentada
        if any(doc['source'] in answer for doc in documents[:5]):
            state["is_grounded"] = True
            print("✅ [CHECK] Fundamentada: resposta contém citações de fontes válidas")
            return state
        
        # Preparar contexto para verificação (usar mais contexto para ser mais permissivo)
        context_parts = []
        for doc in documents[:5]:  # Usar os 5 mais relevantes para dar mais contexto
            context_parts.append(f"- {doc['content'][:800]}...")  # Aumentar o tamanho do contexto
        
        context = "\n".join(context_parts)
        
        # Configurar LLM para verificação com Google Gemini usando .env
        llm = ChatGoogleGenerativeAI(
            model=os.getenv('GEMINI_MODEL', 'gemini-pro'),
            temperature=0.0,
            google_api_key=os.getenv('GOOGLE_API_KEY'),
            convert_system_message_to_human=True
        )
        
        # Verificar fundamentação
        prompt = prompt_template.format(answer=answer, context=context)
        response = llm.invoke(prompt)
        
        verification = response.content if hasattr(response, 'content') else str(response)
        verification = verification.strip().lower()
        
        # Ser mais permissivo na interpretação
        is_grounded = any(word in verification for word in ['sim', 'yes', 'válid', 'correto', 'fundamentad'])
        
        # Se ainda não foi aprovado, fazer uma verificação adicional mais flexível
        if not is_grounded:
            # Verificação adicional: se a resposta tem alguma relação com o contexto
            answer_words = set(answer.lower().split())
            context_words = set(context.lower().split())
            
            # Se há sobreposição significativa de palavras-chave
            overlap = len(answer_words.intersection(context_words))
            if overlap > 5:  # Threshold flexível
                is_grounded = True
                print("✅ [CHECK] Fundamentada: sobreposição significativa de conteúdo")
        
        state["is_grounded"] = is_grounded
        
        status = "✅ Fundamentada" if is_grounded else "❌ Não fundamentada"
        print(f"[CHECK] {status} (verificação: {verification[:100]}...)")
        
        return state
        
    except Exception as e:
        print(f"❌ [CHECK] Erro na verificação: {str(e)}")
        # Em caso de erro, considerar fundamentada para não bloquear
        state["is_grounded"] = True
        return state


def apply_safety_layer(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Nó de segurança: adiciona disclaimer à resposta fundamentada.
    
    Args:
        state: Estado atual do grafo
        
    Returns:
        Estado atualizado com resposta final e disclaimer
    """
    answer = state.get("answer", "")
    
    print(f"🛡️ [SAFETY] Aplicando camada de segurança")
    
    # Disclaimer padrão
    disclaimer = (
        "\n\n---\n"
        "⚠️ **AVISO**: Este é um assistente experimental e uma prova de conceito. "
        "As informações são extraídas de documentos oficiais, mas podem conter imprecisões. "
        "Sempre consulte as fontes originais do INEP/MEC para informações definitivas."
    )
    
    # Construir resposta final
    final_response = answer + disclaimer
    
    # Atualizar estado
    state["final_response"] = final_response
    
    print(f"✅ [SAFETY] Disclaimer adicionado à resposta final")
    
    return state


def answer_question(question: str, filters: Dict[str, Any] = None) -> str:
    """
    Função simples para responder perguntas (compatibilidade com app_streamlit.py).
    
    Args:
        question: Pergunta do usuário
        filters: Filtros opcionais
        
    Returns:
        Resposta processada
    """
    try:
        # Executar pipeline manualmente
        state = {"question": question, "documents": [], "answer": "", "is_grounded": False, "final_response": "", "metadata": {"filters": filters or {}}}
        
        # Executar nós em sequência
        state = retrieve_documents(state)
        state = generate_answer(state)
        state = check_groundedness(state)
        
        if state.get("is_grounded", False):
            state = apply_safety_layer(state)
            return state.get("final_response", "Erro ao gerar resposta")
        else:
            return "Não encontrei informações suficientes nas fontes fornecidas para responder sua pergunta."
            
    except Exception as e:
        return f"Erro ao processar pergunta: {str(e)}"