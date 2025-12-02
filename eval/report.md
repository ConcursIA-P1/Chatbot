# Relatório de Avaliação

Este relatório apresenta os resultados da avaliação do nosso sistema RAG + Agentes, considerando as métricas recomendadas (RAGAS) e os requisitos da atividade.

## Métricas Quantitativas

- **Faithfulness (média):** 0.86  
  > Mede o quanto as respostas se mantêm fiéis ao conteúdo dos documentos de origem, evitando alucinações.

- **Answer Relevancy (média):** 0.90  
  > Mede o quanto as respostas são relevantes e adequadas em relação às perguntas feitas.

- **Latência média (RAGAS eval):** 41 segundos  
  > Tempo médio necessário para processar e avaliar uma query no pipeline.

- **Footprint (memória):** 1671.26 MB  
  > Consumo médio de memória RAM durante a execução da avaliação.

## Análise Crítica

- O valor de **faithfulness (0.86)** indica que, na maioria dos casos, as respostas foram consistentes com as fontes, mas ainda houve uma ocorrência de menor alinhamento (provavelmente relacionada ao terceiro caso testado).  
- O valor de **answer relevancy (0.90)** mostra que o sistema gerou respostas altamente relevantes para as perguntas propostas.  
- A latência (~56s) é relativamente alta, refletindo o custo de avaliação com LLMs e a orquestração via LangGraph.  
- O footprint de **~1.6 GB de RAM** é aceitável para um protótipo local, mas poderia ser otimizado para execução em ambientes com menos recursos.

## Conclusão

O sistema atingiu bons níveis de relevância e consistência, sendo adequado como prova de conceito. Como próximos passos, sugerimos:  
- Otimização do pipeline para reduzir a latência.  
- Investigação de casos de baixa fidelidade (faithfulness) para melhorar o mecanismo de self-check.  
- Testes adicionais em um conjunto maior de perguntas para aumentar a robustez da avaliação.
