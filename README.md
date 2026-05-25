# Análise de Machine Learning: Racismo Ambiental e Saneamento no Brasil

Este repositório contém os dados, scripts e resultados de uma análise de *Machine Learning* voltada para a investigação da hipótese de **Racismo Ambiental** no Brasil. O estudo cruza dados de infraestrutura de saneamento básico, demografia racial e taxas de internação por Doenças Relacionadas ao Saneamento Ambiental Inadequado (DRSAI).

O objetivo principal é validar estatisticamente se a privação de infraestrutura afeta desproporcionalmente populações negras e pardas, refletindo em maior vulnerabilidade na saúde pública, e se esse fenômeno possui raízes estruturais e geográficas.

## 📊 Base de Dados
O estudo analisa uma base tratada contendo **261 municípios brasileiros** e 27 variáveis.
As principais variáveis preditoras incluem:
* `tx_esgoto_pct`: Taxa de cobertura de esgoto adequado.
* `prop_negra_parda_pct`: Proporção da população autodeclarada negra e parda.
* `interacao_esgoto_raca`: Variável de interação calculada para medir o efeito combinado entre saneamento e demografia.
* Variáveis estruturais: `regiao_cod`, `porte_cod`, `capital_cod`.

A variável alvo (target) é a `tx_internacao_100k` (Taxa de Internação por DRSAI a cada 100 mil habitantes).

## 🛠️ Modelos Utilizados
A metodologia foi dividida em três abordagens de *Machine Learning*:
1. **Regressão Linear Múltipla:** Para entender o peso e o coeficiente direcional de cada fator estrutural (*ceteris paribus*).
2. **Árvore de Decisão:** Para mapear a hierarquia de importância das variáveis na separação dos níveis de risco.
3. **Clustering (K-Means):** Aprendizado não supervisionado para agrupar os municípios de acordo com suas similaridades socioambientais.

## 📁 Estrutura do Repositório

* `base_preparada_racismo_ambiental_261mun.xlsx`: Base de dados original utilizada no estudo.
* `ml_racismo_ambiental.py`: Script Python contendo toda a esteira de pré-processamento, treinamento dos modelos e exportação de resultados.
* `racismo_ambiental_ml.ipynb`: Notebook Jupyter (Google Colab) contendo os testes e a execução passo a passo do projeto.
* `/resultados`: Pasta contendo os gráficos gerados (Heatmap, Árvore de Decisão) e a planilha consolidada com as métricas de performance dos modelos.
