# NPS Preditivo — Tech Challenge Fase 1

Projeto desenvolvido como parte do Tech Challenge da Fase 1 da pós-graduação POSTECH (FIAP).

## Objetivo

Analisar dados operacionais de um e-commerce para entender os fatores que influenciam a satisfação dos clientes (NPS) e propor uma estratégia preditiva capaz de antecipar o score antes da aplicação da pesquisa.

## Descrição da base de dados

Arquivo CSV com dados históricos de pedidos, entregas e interações com o atendimento. Principais variáveis:

| Variável | Descrição |
|---|---|
| `customer_id` | Identificador único do cliente |
| `order_id` | Identificador único do pedido |
| `customer_age` | Idade do cliente |
| `customer_region` | Região geográfica |
| `customer_tenure_months` | Tempo de relacionamento (meses) |
| `order_value` | Valor total do pedido |
| `items_quantity` | Quantidade de itens |
| `discount_value` | Valor de desconto aplicado |
| `payment_installments` | Número de parcelas |
| `delivery_time_days` | Tempo total de entrega (dias) |
| `delivery_delay_days` | Dias de atraso na entrega |
| `freight_value` | Valor do frete |
| `delivery_attempts` | Tentativas de entrega |
| `customer_service_contacts` | Contatos com atendimento |
| `resolution_time_days` | Tempo de resolução de problemas (dias) |
| `complaints_count` | Número de reclamações |
| `repeat_purchase_30d` | Recompra em 30 dias (0/1) |
| `csat_internal_score` | Score interno de satisfação |
| `nps_score` | Nota NPS (0–10) |

## 📁 Estrutura do Repositório
 
```
├── data/
│   └── desafio_nps_fase_1.csv    # Base de dados (2.500 registros, 19 variáveis)
├── notebooks/
│   ├── analise_exploratoria_NPS.ipynb   # EDA orientada a negócio
│   └── modelo_regressão_nps.ipynb       # Modelagem preditiva (regressão)
├── images/
│   ├── nps-image-example.png            # Ilustração do NPS
│   ├── CSAT-EXAMPLE.jpg                 # Escala CSAT
│   └── matriz_confusao_random_forest.png
├── models/
│   └── model.pkl                        # Modelo treinado (joblib)
├── docs/
│   └── Tech-Challenge-Fase-1-FIAP.pptx  # Apresentação do projeto
├── requirements.txt
└── README.md
```

## Metodologia

1. **Entendimento do negócio** — definição do problema e impacto do NPS no e-commerce
2. **Definição da target** — análise conceitual da variável `nps_score`
3. **EDA** — análise exploratória orientada a negócio, identificando padrões e pontos de ruptura na experiência do cliente
4. **Modelagem preditiva**  — proposta de modelo de classificação ou regressão para prever NPS antes da pesquisa

## Como reproduzir

```bash
# Clone o repositório
git clone <https://github.com/LucasAlexSant/Tech-Challenge-Fase-1-FIAP.git>
cd <https://github.com/LucasAlexSant/Tech-Challenge-Fase-1-FIAP.git>

# Instale as dependências
pip install -r requirements.txt

# Execute os notebooks na ordem numérica em notebooks/
```
