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
    page_title="RAG ENEM - Assistente Inteligente", 
    page_icon="📚",
    layout="centered"
)

# Título e descrição
st.title("📚 RAG ENEM - Assistente Inteligente")
st.markdown("""
Este assistente usa Retrieval Augmented Generation (RAG) com Google Gemini para responder perguntas sobre o ENEM 
baseado em documentos oficiais como editais, cartilhas e provas anteriores.
""")

# Verificar se as variáveis de ambiente estão configuradas
if not os.getenv('GOOGLE_API_KEY'):
    st.error("❌ GOOGLE_API_KEY não configurada! Configure o arquivo .env")
    st.stop()

if not os.getenv('CHROMA_API_KEY') or not os.getenv('CHROMA_TENANT'):
    st.error("❌ Credenciais do ChromaDB não configuradas! Configure CHROMA_API_KEY e CHROMA_TENANT no arquivo .env")
    st.stop()

# Interface principal
st.markdown("### 🤔 Faça sua pergunta sobre o ENEM")

# Campo de entrada para pergunta
user_question = st.text_area(
    "Digite sua pergunta:",
    placeholder="Ex: Quais são as datas do ENEM 2025? Como é calculada a nota da redação?",
    height=100
)

# Opções avançadas (sidebar)
with st.sidebar:
    st.header("⚙️ Configurações")
    
    # Filtros de busca
    st.subheader("Filtros de Busca")
    
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
    
    st.subheader("ℹ️ Sobre")
    st.markdown("""
    **Fontes dos dados:**
    - Editais 2023-2025
    - Cartilha de Redação
    - Matriz de Referência
    """)

# Botão para fazer pergunta
col1, col2 = st.columns([1, 4])
with col1:
    ask_button = st.button("🔍 Perguntar", type="primary", use_container_width=True)

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
        with st.spinner("🤖 Processando sua pergunta..."):
            try:
                # Criar e usar o grafo RAG completo
                rag_graph = create_enem_rag_graph()
                result = rag_graph.invoke(user_question, metadata)
                
                # Mostrar resultados
                if result.get("final_response"):
                    st.success("✅ Resposta encontrada!")
                    
                    # Mostrar a resposta
                    st.markdown("### 📝 Resposta:")
                    st.markdown(result["final_response"])
                    
                    # Mostrar informações adicionais
                    with st.expander("📊 Detalhes da busca"):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Documentos encontrados", len(result.get("documents", [])))
                        
                        with col2:
                            grounded = result.get("is_grounded", False)
                            st.metric("Resposta fundamentada", "✅ Sim" if grounded else "❌ Não")
                        
                        with col3:
                            st.metric("Filtros aplicados", len(filters))
                        
                        # Mostrar documentos fonte
                        if result.get("documents"):
                            st.markdown("**📚 Fontes consultadas:**")
                            for i, doc in enumerate(result["documents"][:3], 1):
                                st.markdown(f"**{i}.** {doc.get('source', 'N/A')} (Página {doc.get('page', 'N/A')})")
                                with st.expander(f"Ver conteúdo do documento {i}"):
                                    st.text(doc.get('content', 'N/A'))
                
                else:
                    st.error("❌ Não foi possível gerar uma resposta. Tente reformular sua pergunta.")
                    
            except Exception as e:
                st.error(f"❌ Erro ao processar pergunta: {str(e)}")
                st.markdown("**Possíveis causas:**")
                st.markdown("- Configuração incorreta das APIs")
                st.markdown("- Problema de conectividade")
                st.markdown("- Base de dados não encontrada")
    else:
        st.warning("⚠️ Por favor, digite uma pergunta.")

# Exemplos de perguntas
st.markdown("### 💡 Exemplos de perguntas:")
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
        if st.button(f"💭 {question}", key=f"example_{i}"):
            st.rerun()

