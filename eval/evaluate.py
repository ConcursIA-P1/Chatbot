import os
import json
import time
import psutil
from dotenv import load_dotenv
from datasets import Dataset
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import evaluate
from ragas.metrics import answer_relevancy, faithfulness

load_dotenv()

gemini = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

# Dados de teste (question, contexts, answer)
data = [
    {
        "question": "Quais são as datas do ENEM 2024?",
        "contexts": ["""
Inscrições 27/5 a 7/6/2024
Pagamento da taxa de inscrição 27/5 a 12/6/2024
Tratamento pelo nome social 27/5 a 7/6/2024
Atendimento especializadoSolicitação 27/5 a 7/6/2024
Resultado 17/6/2024
Recurso 17/6 a 21/6/2024
Resultado do recurso27/6/2024
Aplicação 3 e 10/11/2024
[Fonte: edital_2024.pdf, Página: 1]
        """],
        "answer": """
As datas do ENEM 2024 são as seguintes:

Inscrições: 27/5 a 7/6/2024 [Fonte: edital_2024.pdf, Página: 1]
Pagamento da taxa de inscrição: 27/5 a 12/6/2024 [Fonte: edital_2024.pdf, Página: 1]
Tratamento pelo nome social: 27/5 a 7/6/2024 [Fonte: edital_2024.pdf, Página: 1]
Solicitação de Atendimento especializado: 27/5 a 7/6/2024 [Fonte: edital_2024.pdf, Página: 1]
Resultado da solicitação de Atendimento especializado: 17/6/2024 [Fonte: edital_2024.pdf, Página: 1]
Recurso da solicitação de Atendimento especializado: 17/6 a 21/6/2024 [Fonte: edital_2024.pdf, Página: 1]
Resultado do recurso da solicitação de Atendimento especializado: 27/6/2024 [Fonte: edital_2024.pdf, Página: 1]
Aplicação das provas: 3 e 10/11/2024 [Fonte: edital_2024.pdf, Página: 1]
        """,
    },
    {
        "question": "Qual a nota máxima da redação do ENEM?",
        "contexts": ["""
COMO SERÁ ATRIBUÍDA A NOTA À REDAÇÃO?
A nota da redação, que variará entre 0 (zero) e 1.000 (mil) pontos
[Fonte: cartilha_redacao_2024.pdf, Página: 8]
        """],
        "answer": """
            A nota máxima da redação do ENEM é de 1.000 (mil) pontos [Fonte: cartilha_redacao_2024.pdf, Página: 8].
        """,
    },
    {
        "question": "Quais documentos são necessários para realizar a prova do ENEM?",
        "contexts": ["""
            Para fazer a prova do ENEM, é obrigatório apresentar um documento de identificação oficial com foto que esteja dentro da validade. Entre os documentos aceitos estão: carteira de identidade (RG), passaporte, carteira de motorista (CNH), carteira de trabalho, documentos de identidade emitidos por instituições militares ou órgãos de classe, entre outros. Cópias, mesmo que autenticadas, e documentos digitais em dispositivos eletrônicos não são aceitos.
            [Fonte: Edital_2025.pdf, Página: 12]
        """],
        "answer": """
Para realizar a prova do ENEM, é obrigatória a apresentação de documento de identificação oficial e original com foto, emitido por órgãos brasileiros [Fonte: edital_2024.pdf, Página: 10].

Consideram-se documentos válidos para identificação do participante brasileiro:

    Cédulas de Identidade expedidas por Secretarias de Segurança Pública, Forças Armadas, Polícia Militar e Polícia Federal [Fonte: edital_2024.pdf, Página: 10].
    Identificação fornecida por ordens ou conselhos de classes que, por lei, tenha validade como documento de identidade [Fonte: edital_2024.pdf, Página: 10].
    Passaporte [Fonte: edital_2024.pdf, Página: 10].
    Carteira Nacional de Habilitação, na forma da Lei nº 9.503, de 23 de setembro de 1997 [Fonte: edital_2024.pdf, Página: 10].
    Carteira de Trabalho e Previdência Social impressa e expedida após 27 de janeiro de 1997 [Fonte: edital_2024.pdf, Página: 10].
    Documentos digitais com foto (e-Título, CNH digital, RG digital e CIN digital) apresentados nos respectivos aplicativos oficiais ou no aplicativo Gov.br [Fonte: edital_2024.pdf, Página: 10].

Caso o participante precise aguardar o recebimento de documento válido listado nos itens 10.1 ou 10.2, deverá fazê-lo fora do local de provas [Fonte: Edital_2025.pdf, Página: 12].
        """,
    }
]

dataset = Dataset.from_list(data)

# --- Avaliação com medição de latência e memória ---
eval_start_time = time.time()
process = psutil.Process()
eval_mem_before = process.memory_info().rss / (1024 ** 2)  # MB

results = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy],
    llm=gemini, 
    embeddings=embeddings
)

eval_latency = time.time() - eval_start_time
eval_mem_after = process.memory_info().rss / (1024 ** 2)
eval_memory = eval_mem_after - eval_mem_before

final_output = {
    "faithfulness": results["faithfulness"],
    "answer_relevancy": results["answer_relevancy"],
    "evaluation_performance": {
        "ragas_eval_latency_seconds": eval_latency,
        "ragas_eval_memory_mb": eval_memory
    }
}


base_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(base_dir, "results")
os.makedirs(results_dir, exist_ok=True)

with open(results_dir + "/manual_evaluation_results.json", "w", encoding="utf-8") as f:
    json.dump(final_output, f, ensure_ascii=False, indent=4)

print("✅ Resultados salvos em results")
