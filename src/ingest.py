"""
Processador de documentos do ENEM para ingestão em RAG.
Responsável por carregar, dividir, limpar, gerar embeddings e armazenar documentos PDF.
"""

import os
import re
import hashlib
from typing import List, Dict, Any, Optional
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb
from dotenv import load_dotenv

# Carregar variáveis do .env
load_dotenv()


class ENEMDocumentProcessor:
    """
    Processador de documentos do ENEM para ingestão em RAG.
    Responsável por carregar, dividir, limpar, gerar embeddings e armazenar documentos PDF.
    """
    
    def __init__(
        self, 
        data_dir: str = None, 
        vector_store_dir: str = None,
        chunk_size: int = None, 
        chunk_overlap: int = None,
        collection_name: str = None,
        embedding_model: str = None,
        batch_size: int = None,
        use_cloud: bool = None,
        chroma_api_key: str = None,
        chroma_tenant: str = None,
        chroma_database: str = None
    ):
        # Usar configurações do .env como padrão, permitindo override
        self.data_dir = Path(data_dir or os.getenv('DATA_DIR', 'data'))
        self.vector_store_dir = Path(vector_store_dir or os.getenv('VECTOR_STORE_DIR', 'vector_store'))
        self.chunk_size = chunk_size or int(os.getenv('CHUNK_SIZE', '1000'))
        self.chunk_overlap = chunk_overlap or int(os.getenv('CHUNK_OVERLAP', '150'))
        self.collection_name = collection_name or os.getenv('COLLECTION_NAME', 'enem_documents')
        self.batch_size = batch_size or int(os.getenv('BATCH_SIZE', '50'))
        self.use_cloud = use_cloud if use_cloud is not None else os.getenv('USE_CHROMA_CLOUD', 'true').lower() == 'true'
        
        # Configurações do Chroma Cloud
        self.chroma_api_key = chroma_api_key or os.getenv('CHROMA_API_KEY')
        self.chroma_tenant = chroma_tenant or os.getenv('CHROMA_TENANT')
        self.chroma_database = chroma_database or os.getenv('CHROMA_DATABASE', 'enem_rag')
        
        # Configurar text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Configurar modelo de embeddings
        embedding_model_name = embedding_model or os.getenv('EMBEDDING_MODEL', 'BAAI/bge-m3')
        print(f"Carregando modelo de embedding: {embedding_model_name}")
        self.embeddings_model = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': os.getenv('MODEL_DEVICE', 'cpu')},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Configurar ChromaDB
        self._setup_chromadb()
        
    def _setup_chromadb(self):
        """Configura o cliente ChromaDB (cloud ou local) e a collection."""
        try:
            if self.use_cloud and self.chroma_api_key and self.chroma_tenant:
                print("🌐 Configurando ChromaDB Cloud...")
                
                # Configurar cliente cloud com o formato correto
                self.chroma_client = chromadb.CloudClient(
                    api_key=self.chroma_api_key,
                    tenant=self.chroma_tenant,
                    database=self.chroma_database
                )
                print(f"✅ ChromaDB Cloud conectado - Tenant: {self.chroma_tenant}, Database: {self.chroma_database}")
                
            else:
                print("💾 Configurando ChromaDB Local...")
                
                if self.use_cloud:
                    print("⚠️  Configurações do cloud não encontradas. Usando local como fallback.")
                    print("   Para usar cloud, configure as variáveis:")
                    print("   - CHROMA_API_KEY: Sua chave de API do Chroma")
                    print("   - CHROMA_TENANT: Seu tenant ID do Chroma")
                    print("   - CHROMA_DATABASE: Nome da database (opcional, padrão: enem_rag)")
                    print("   Ou crie uma conta gratuita em: https://www.trychroma.com/")
                
                # Configurar cliente local como fallback
                self.vector_store_dir.mkdir(exist_ok=True)
                
                try:
                    self.chroma_client = chromadb.PersistentClient(
                        path=str(self.vector_store_dir)
                    )
                    print(f"✅ ChromaDB Local configurado em: {self.vector_store_dir}")
                    
                except Exception as e:
                    print(f"⚠️  PersistentClient falhou: {str(e)}")
                    print("Usando cliente em memória...")
                    self.chroma_client = chromadb.Client()
                    print("⚠️  AVISO: Dados serão perdidos após reiniciar o programa")
            
            # Criar ou obter collection
            try:
                self.collection = self.chroma_client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={
                        "description": "Documentos do ENEM processados para RAG",
                        "embedding_model": os.getenv('EMBEDDING_MODEL', 'BAAI/bge-m3'),
                        "chunk_size": self.chunk_size,
                        "chunk_overlap": self.chunk_overlap,
                        "version": "1.0"
                    }
                )
                print(f"✅ Collection '{self.collection_name}' pronta para uso")
                
            except Exception as e:
                # Se falhar com metadados, tenta sem
                print(f"⚠️  Criando collection sem metadados devido ao erro: {str(e)}")
                self.collection = self.chroma_client.get_or_create_collection(
                    name=self.collection_name
                )
                print(f"✅ Collection '{self.collection_name}' criada (sem metadados)")
            
        except Exception as e:
            print(f"❌ Erro ao configurar ChromaDB: {str(e)}")
            print("\n🔧 Soluções possíveis:")
            print("1. Para ChromaDB Cloud:")
            print("   - Verifique se CHROMA_API_KEY, CHROMA_TENANT estão corretos")
            print("   - Confirme se sua conta Chroma está ativa")
            print("   - Teste a conectividade de rede")
            print("2. Para ChromaDB Local:")
            print("   - Verifique se o diretório tem permissões de escrita")
            print("   - Reinstale chromadb: pip install --upgrade chromadb")
            raise RuntimeError(f"Erro crítico ao configurar ChromaDB: {str(e)}")

    def _clean_text(self, text: str) -> str:
        """
        Limpa o texto removendo caracteres indesejados e normalizando espaços.
        
        Args:
            text: Texto a ser limpo
            
        Returns:
            Texto limpo e normalizado
        """
        # Remove caracteres de controle e caracteres especiais problemáticos
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        
        # Remove quebras de linha desnecessárias e múltiplos espaços
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)  # Múltiplas quebras de linha
        text = re.sub(r' {2,}', ' ', text)  # Múltiplos espaços
        text = re.sub(r'\t+', ' ', text)  # Múltiplas tabulações
        
        # Remove cabeçalhos e rodapés comuns do ENEM/INEP
        patterns_to_remove = [
            r'ENEM \d{4}.*?Página \d+',
            r'Instituto Nacional de Estudos e Pesquisas.*?Anísio Teixeira',
            r'INEP.*?\d{4}',
            r'Ministério da Educação.*?MEC',
            r'Exame Nacional do Ensino Médio.*?\d{4}',
            r'www\..*?\.gov\.br',
            r'Página \d+ de \d+',
        ]
        
        for pattern in patterns_to_remove:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
        
        # Normaliza espaços em branco finais
        text = '\n'.join(line.strip() for line in text.split('\n'))
        text = re.sub(r'\n{3,}', '\n\n', text)  # Limita quebras de linha consecutivas
        
        return text.strip()
    
    def _extract_metadata_from_filename(self, filepath: Path) -> Dict[str, Any]:
        """
        Extrai metadados do nome do arquivo e estrutura de pastas.
        
        Args:
            filepath: Caminho do arquivo
            
        Returns:
            Dicionário com metadados extraídos
        """
        metadata = {
            'source': filepath.name,
            'full_path': str(filepath),
            'category': filepath.parent.name,
            'year': None,
            'document_type': None,
        }
        
        filename = filepath.stem.lower()
        
        # Extrai ano do nome do arquivo
        year_match = re.search(r'(\d{4})', filename)
        if year_match:
            metadata['year'] = int(year_match.group(1))
        
        # Classifica tipo de documento baseado na pasta e nome
        if 'exame' in filepath.parent.name:
            metadata['document_type'] = 'prova'
            if 'd1' in filename:
                metadata['day'] = 1
            elif 'd2' in filename:
                metadata['day'] = 2
        elif 'answ' in filepath.parent.name or 'gabarit' in filename:
            metadata['document_type'] = 'gabarito'
            if 'gd1' in filename:
                metadata['day'] = 1
            elif 'gd2' in filename:
                metadata['day'] = 2
        elif 'editais' in filepath.parent.name:
            if 'edital' in filename:
                metadata['document_type'] = 'edital'
            elif 'cartilha' in filename:
                metadata['document_type'] = 'cartilha'
            elif 'bncc' in filename:
                metadata['document_type'] = 'bncc'
            elif 'matriz' in filename:
                metadata['document_type'] = 'matriz_referencia'
        
        return metadata
    
    def load_documents(self) -> List[Document]:
        """
        Carrega todos os documentos PDF das pastas de dados.
        
        Returns:
            Lista de documentos carregados com metadados
        """
        documents = []
        pdf_files = list(self.data_dir.rglob("*.pdf"))
        
        if not pdf_files:
            raise FileNotFoundError(f"Nenhum arquivo PDF encontrado em {self.data_dir}")
        
        print(f"Encontrados {len(pdf_files)} arquivos PDF para processamento...")
        
        for pdf_path in pdf_files:
            try:
                print(f"Carregando: {pdf_path.name}")
                
                # Carrega o PDF usando PyPDFLoader
                loader = PyPDFLoader(str(pdf_path))
                pdf_documents = loader.load()
                
                # Extrai metadados do arquivo
                base_metadata = self._extract_metadata_from_filename(pdf_path)
                
                # Adiciona metadados específicos de página para cada documento
                for i, doc in enumerate(pdf_documents):
                    doc.metadata.update(base_metadata)
                    doc.metadata['page'] = i + 1
                    doc.metadata['total_pages'] = len(pdf_documents)
                    
                    # Limpa o conteúdo do documento
                    doc.page_content = self._clean_text(doc.page_content)
                    
                    # Só adiciona se o conteúdo não estiver vazio após limpeza
                    if doc.page_content.strip():
                        documents.append(doc)
                
                print(f"  → {len(pdf_documents)} páginas carregadas")
                
            except Exception as e:
                print(f"Erro ao carregar {pdf_path.name}: {str(e)}")
                continue
        
        print(f"\nTotal de páginas carregadas: {len(documents)}")
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Divide os documentos em chunks menores para melhor recuperação.
        
        Args:
            documents: Lista de documentos a serem divididos
            
        Returns:
            Lista de chunks com metadados preservados
        """
        print(f"Dividindo {len(documents)} documentos em chunks...")
        print(f"Configuração: chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap}")
        
        all_chunks = []
        
        for doc in documents:
            try:
                # Divide o documento em chunks
                chunks = self.text_splitter.split_documents([doc])
                
                # Adiciona metadados específicos do chunk
                for i, chunk in enumerate(chunks):
                    chunk.metadata['chunk_id'] = i
                    chunk.metadata['total_chunks'] = len(chunks)
                    chunk.metadata['chunk_size'] = len(chunk.page_content)
                    
                    # Preserva referência ao documento original
                    chunk.metadata['original_source'] = doc.metadata['source']
                    chunk.metadata['original_page'] = doc.metadata['page']
                
                all_chunks.extend(chunks)
                
            except Exception as e:
                print(f"Erro ao dividir documento {doc.metadata.get('source', 'unknown')}: {str(e)}")
                continue
        
        print(f"Total de chunks criados: {len(all_chunks)}")
        return all_chunks
    
    def process_documents(self) -> List[Document]:
        """
        Processa todos os documentos: carrega, limpa e divide.
        
        Returns:
            Lista de chunks processados prontos para embedding
        """
        print("=== Iniciando processamento de documentos ENEM ===")
        
        # Fase 1: Carregamento
        documents = self.load_documents()
        
        # Fase 2: Divisão
        chunks = self.split_documents(documents)
        
        # Estatísticas finais
        print("\n=== Estatísticas de Processamento ===")
        print(f"Documentos originais: {len(documents)}")
        print(f"Chunks finais: {len(chunks)}")
        print(f"Tamanho médio dos chunks: {sum(len(chunk.page_content) for chunk in chunks) / len(chunks):.0f} caracteres")
        
        # Estatísticas por categoria
        categories = {}
        for chunk in chunks:
            cat = chunk.metadata.get('category', 'unknown')
            categories[cat] = categories.get(cat, 0) + 1
        
        print("\nChunks por categoria:")
        for cat, count in categories.items():
            print(f"  {cat}: {count} chunks")
        
        return chunks

    def _generate_document_id(self, chunk: Document) -> str:
        """
        Gera um ID único para o documento baseado no conteúdo e metadados.
        
        Args:
            chunk: Documento a ser processado
            
        Returns:
            ID único para o documento
        """
        # Criar string única baseada em metadados essenciais
        id_string = f"{chunk.metadata.get('source', 'unknown')}_" \
                   f"page_{chunk.metadata.get('page', 0)}_" \
                   f"chunk_{chunk.metadata.get('chunk_id', 0)}"
        
        # Adicionar hash do conteúdo para garantir unicidade
        content_hash = hashlib.md5(chunk.page_content.encode()).hexdigest()[:8]
        
        return f"{id_string}_{content_hash}"
    
    def _prepare_metadata_for_chromadb(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepara metadados para armazenamento no ChromaDB.
        ChromaDB requer que metadados sejam tipos básicos (str, int, float, bool).
        
        Args:
            metadata: Metadados originais do documento
            
        Returns:
            Metadados limpos para ChromaDB
        """
        clean_metadata = {}
        
        for key, value in metadata.items():
            if value is None:
                clean_metadata[key] = ""
            elif isinstance(value, (str, int, float, bool)):
                clean_metadata[key] = value
            else:
                clean_metadata[key] = str(value)
        
        return clean_metadata
    
    def store_embeddings(self, chunks: List[Document]) -> None:
        """
        Gera embeddings e armazena os chunks no ChromaDB.
        
        Args:
            chunks: Lista de chunks processados para armazenar
        """
        if not chunks:
            print("Nenhum chunk para armazenar.")
            return
        
        print(f"\n=== Iniciando armazenamento de embeddings ===")
        print(f"Total de chunks para processar: {len(chunks)}")
        print(f"Processando em batches de {self.batch_size}")
        
        # Verificar se já existem documentos na collection
        existing_count = self.collection.count()
        if existing_count > 0:
            print(f"⚠️  Collection já contém {existing_count} documentos")
            response = input("Deseja limpar a collection existente? (y/N): ")
            if response.lower() in ['y', 'yes']:
                self.collection.delete()
                print("Collection limpa.")
            else:
                print("Adicionando novos documentos à collection existente.")
        
        # Processar chunks em batches
        total_batches = (len(chunks) + self.batch_size - 1) // self.batch_size
        
        for batch_idx in range(0, len(chunks), self.batch_size):
            batch_num = (batch_idx // self.batch_size) + 1
            batch = chunks[batch_idx:batch_idx + self.batch_size]
            
            print(f"Processando batch {batch_num}/{total_batches} ({len(batch)} chunks)...")
            
            try:
                # Preparar dados para o batch
                ids = []
                documents = []
                metadatas = []
                
                for chunk in batch:
                    doc_id = self._generate_document_id(chunk)
                    clean_metadata = self._prepare_metadata_for_chromadb(chunk.metadata)
                    
                    ids.append(doc_id)
                    documents.append(chunk.page_content)
                    metadatas.append(clean_metadata)
                
                # Gerar embeddings para o batch
                print(f"  → Gerando embeddings...")
                embeddings = self.embeddings_model.embed_documents(documents)
                
                # Adicionar ao ChromaDB
                print(f"  → Armazenando no ChromaDB...")
                self.collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas
                )
                
                print(f"  ✅ Batch {batch_num} processado com sucesso")
                
            except Exception as e:
                print(f"  ❌ Erro no batch {batch_num}: {str(e)}")
                continue
        
        # Estatísticas finais
        final_count = self.collection.count()
        print(f"\n✅ Armazenamento concluído!")
        print(f"📊 Total de documentos na collection: {final_count}")

    def process_and_store_documents(self) -> Dict[str, Any]:
        """
        Executa o pipeline completo: carrega, processa, gera embeddings e armazena documentos.
        
        Returns:
            Dicionário com estatísticas do processamento
        """
        print("=== Iniciando processamento completo de documentos ENEM ===")
        
        try:
            # Fase 1-3: Carregamento, limpeza e divisão
            chunks = self.process_documents()
            
            # Fase 4: Embedding e armazenamento
            self.store_embeddings(chunks)
            
            # Fase 5: Obter estatísticas finais
            final_count = self.collection.count()
            stats = {
                'total_documents': final_count,
                'collection_name': self.collection_name,
                'embedding_model': os.getenv('EMBEDDING_MODEL', 'BAAI/bge-m3')
            }
            
            print("\n=== Processamento Completo Finalizado ===")
            print(f"✅ {stats.get('total_documents', 0)} documentos processados e armazenados")
            
            return stats
            
        except Exception as e:
            print(f"❌ Erro durante processamento completo: {str(e)}")
            raise


if __name__ == "__main__":
    """Execução do script de ingestão quando chamado diretamente."""
    try:
        print("=== INICIANDO INGESTÃO DE DOCUMENTOS ===")
        
        # Verificar se as variáveis de ambiente estão configuradas
        if not os.getenv('GOOGLE_API_KEY'):
            print("⚠️  GOOGLE_API_KEY não configurada no .env")
        if not os.getenv('CHROMA_API_KEY') and os.getenv('USE_CHROMA_CLOUD', 'true').lower() == 'true':
            print("⚠️  CHROMA_API_KEY não configurada no .env (necessária para cloud)")
        
        # Inicializar processador
        processor = ENEMDocumentProcessor()
        
        # Executar pipeline completo
        stats = processor.process_and_store_documents()
        
        print(f"\n🎉 Sistema RAG do ENEM pronto para uso!")
        print(f"📊 Estatísticas finais: {stats}")
        
    except Exception as e:
        print(f"❌ Erro durante execução: {str(e)}")
        raise