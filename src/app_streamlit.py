import streamlit as st
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Carregar variáveis do .env
load_dotenv()

# Adicionar src ao path para imports
sys.path.append(str(Path(__file__).parent))

# Importações dos módulos do projeto
from agents import answer_question
from graph import create_enem_rag_graph

# Configurar a página do Streamlit
st.set_page_config(
    page_title="ConcursIA - Assistente Inteligente", 
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado para design escuro e profissional
st.markdown("""
<style>
    /* Tema escuro geral */
    .stApp {
        background-color: #0E1117;
    }
    
    /* Header principal */
    .main-header {
        background: linear-gradient(135deg, #1a1d29 0%, #2d3748 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        border-left: 4px solid #4299e1;
    }
    
    .main-title {
        color: #ffffff;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .main-subtitle {
        color: #a0aec0;
        font-size: 1rem;
        margin-top: 0.5rem;
        font-weight: 400;
    }
    
    /* Área de chat */
    .chat-container {
        background-color: #1a202c;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border: 1px solid #2d3748;
    }
    
    /* Botões */
    .stButton > button {
        background-color: #4299e1;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        background-color: #3182ce;
        box-shadow: 0 4px 12px rgba(66, 153, 225, 0.4);
    }
    
    /* Área de texto */
    .stTextArea textarea {
        background-color: #2d3748;
        color: #e2e8f0;
        border: 1px solid #4a5568;
        border-radius: 5px;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #1a202c;
    }
    
    /* Métricas */
    .metric-card {
        background-color: #2d3748;
        padding: 1rem;
        border-radius: 8px;
        border-left: 3px solid #4299e1;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background-color: #2d3748;
        color: #e2e8f0;
        border-radius: 5px;
    }
    
    /* Alertas */
    .stAlert {
        background-color: #2d3748;
        border-left: 4px solid #4299e1;
    }
</style>
""", unsafe_allow_html=True)

# Header principal
st.markdown("""
<div class="main-header">
    <h1 class="main-title">ConcursIA</h1>
    <p class="main-subtitle">Plataforma inteligente de preparação para concursos | Demonstração RAG ENEM</p>
</div>
""", unsafe_allow_html=True)

# Verificar se as variáveis de ambiente estão configuradas
if not os.getenv('GOOGLE_API_KEY'):
    st.error("GOOGLE_API_KEY não configurada. Configure o arquivo .env")
    st.stop()

if not os.getenv('CHROMA_API_KEY') or not os.getenv('CHROMA_TENANT'):
    st.error("Credenciais do ChromaDB não configuradas. Configure CHROMA_API_KEY e CHROMA_TENANT no arquivo .env")
    st.stop()

# Interface principal
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
st.markdown("### Consulta Inteligente")
st.markdown("Faça perguntas sobre editais, provas e conteúdos do ENEM utilizando tecnologia RAG.")

# Campo de entrada para pergunta
user_question = st.text_area(
    "Digite sua pergunta:",
    placeholder="Exemplo: Quais são as datas do ENEM 2025? Como é calculada a nota da redação?",
    height=120,
    label_visibility="collapsed"
)

# Opções avançadas (sidebar)
with st.sidebar:
    st.markdown("## Configurações")
    
    st.markdown("---")
    
    # Filtros de busca
    st.markdown("### Filtros de Busca")
    
    filter_year = st.selectbox(
        "Filtrar por ano:",
        options=["Todos"] + list(range(2014, 2026)),
        index=0
    )
    
    filter_type = st.selectbox(
        "Tipo de documento:",
        options=["Todos", "Prova", "Gabarito", "Edital", "Cartilha"],
        index=0
    )
    
    max_docs = st.slider(
        "Máximo de documentos:",
        min_value=1,
        max_value=10,
        value=int(os.getenv('MAX_DOCUMENTS', '5'))
    )
    
    st.markdown("---")
    
    st.markdown("### Sobre o Sistema")
    st.markdown("""
    **Base de conhecimento:**
    - Editais 2023-2025
    - Cartilha de Redação
    - Matriz de Referência
    
    **Tecnologia:**
    - RAG (Retrieval Augmented Generation)
    - Google Gemini LLM
    - ChromaDB Vector Store
    """)
    
    st.markdown("---")
    
    st.markdown("""
    <div style='text-align: center; color: #718096; font-size: 0.85rem;'>
        <p>ConcursIA</p>
        <p>Plataforma de preparação inteligente</p>
    </div>
    """, unsafe_allow_html=True)

# Botão para fazer pergunta
col1, col2, col3 = st.columns([2, 1, 2])
with col2:
    ask_button = st.button("Consultar", type="primary", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# Processar pergunta quando botão for clicado
if ask_button:
    if user_question.strip():
        # Preparar metadados de filtros
        filters = {}
        if filter_year != "Todos":
            filters["year"] = filter_year
        
        if filter_type != "Todos":
            type_mapping = {
                "Prova": "prova",
                "Gabarito": "gabarito", 
                "Edital": "edital",
                "Cartilha": "cartilha"
            }
            filters["document_type"] = type_mapping[filter_type]
        
        metadata = {
            "filters": filters,
            "max_documents": max_docs
        }
        
        # Mostrar progresso
        with st.spinner("Processando consulta..."):
            try:
                # Criar e usar o grafo RAG completo
                rag_graph = create_enem_rag_graph()
                result = rag_graph.invoke(user_question, metadata)
                
                # Mostrar resultados
                if result.get("final_response"):
                    st.markdown("---")
                    st.success("Resposta gerada com sucesso")
                    
                    # Mostrar a resposta
                    st.markdown("### Resposta")
                    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
                    st.markdown(result["final_response"])
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Mostrar informações adicionais
                    with st.expander("Detalhes da Consulta"):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                            st.metric("Documentos", len(result.get("documents", [])))
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        with col2:
                            grounded = result.get("is_grounded", False)
                            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                            st.metric("Fundamentada", "Sim" if grounded else "Não")
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        with col3:
                            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                            st.metric("Filtros", len(filters))
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Mostrar documentos fonte
                        if result.get("documents"):
                            st.markdown("---")
                            st.markdown("**Fontes Consultadas**")
                            for i, doc in enumerate(result["documents"][:3], 1):
                                st.markdown(f"{i}. {doc.get('source', 'N/A')} | Página {doc.get('page', 'N/A')}")
                                with st.expander(f"Visualizar conteúdo {i}"):
                                    st.code(doc.get('content', 'N/A'), language=None)
                
                else:
                    st.error("Não foi possível gerar uma resposta. Tente reformular sua pergunta.")
                    
            except Exception as e:
                st.error(f"Erro ao processar consulta: {str(e)}")
                with st.expander("Detalhes do erro"):
                    st.markdown("**Possíveis causas:**")
                    st.markdown("- Configuração incorreta das APIs")
                    st.markdown("- Problema de conectividade")
                    st.markdown("- Base de dados não encontrada")
    else:
        st.warning("Por favor, digite uma pergunta.")

# Exemplos de perguntas
st.markdown("---")
st.markdown("### Exemplos de Consultas")
st.markdown("Clique em uma das perguntas abaixo para testá-la:")

example_questions = [
    "Quais são as datas do ENEM 2025?",
    "Como é calculada a nota da redação?",
    "Quais documentos preciso para me inscrever?",
    "Quantas questões tem cada prova do ENEM?",
    "Quais são as competências da redação?",
    "Qual é o prazo de inscrição do ENEM 2025?"
]

cols = st.columns(2)
for i, question in enumerate(example_questions):
    with cols[i % 2]:
        if st.button(question, key=f"example_{i}", use_container_width=True):
            st.rerun()

