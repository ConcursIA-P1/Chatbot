# Chatbot RAG - ENEM

Módulo de chatbot inteligente baseado em RAG (Retrieval Augmented Generation) para responder perguntas sobre o ENEM (e futuramente consursos). Este é um submódulo do projeto **ConcursIA**.

## Visão Geral

Assistente conversacional que utiliza busca vetorial e LLM para responder perguntas sobre editais, provas e conteúdos do ENEM, fundamentando respostas em documentos oficiais.

**Funcionalidades principais:**
- Consulta inteligente sobre ENEM (datas, regras, conteúdos)
- Respostas fundamentadas em fontes oficiais (editais, cartilhas)
- Citação automática de fontes
- Verificação de qualidade das respostas
- Interface web profissional (Streamlit)

---

## Arquitetura

Pipeline baseado em **LangGraph** com 4 etapas sequenciais:

```
Pergunta → [1. Retrieve] → [2. Generate] → [3. Check] → [4. Safety] → Resposta
```

### Componentes

1. **Retrieve Documents** (`agents.py`)
   - Busca vetorial no ChromaDB usando embeddings (HuggingFace BAAI/bge-m3)
   - Retorna top-k documentos mais similares

2. **Generate Answer** (`agents.py`)
   - LLM Google Gemini gera resposta baseada nos documentos
   - Prompt engineering para garantir fidelidade às fontes

3. **Check Groundedness** (`agents.py`)
   - Valida se resposta está fundamentada nos documentos
   - Usa LLM para verificação automática

4. **Apply Safety Layer** (`agents.py`)
   - Adiciona disclaimers e formatação final


## Stack

- **LangChain + LangGraph** - Framework para RAG e orquestração
- **Google Gemini** - LLM (gemini-pro)
- **HuggingFace** - Embeddings (BAAI/bge-m3)
- **ChromaDB** - Vector database
- **Streamlit** - Interface web
- **PyPDF2** - Processamento de PDFs

## Instalação e Uso

```bash
# Instalar dependências
pip install -r requirements.txt

# Configurar .env
GOOGLE_API_KEY=sua_chave
CHROMA_API_KEY=sua_chave
CHROMA_TENANT=seu_tenant

# Ingerir documentos (primeira vez)
python src/ingest.py

# Executar interface
streamlit run src/app_streamlit.py
```
