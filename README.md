# NPS Preditivo — Tech Challenge Fase 1

Projeto desenvolvido como parte do Tech Challenge da Fase 1 da pós-graduação POSTECH (FIAP).

## Objetivo

Analisar dados operacionais de um e-commerce para entender os fatores que influenciam a satisfação dos clientes (NPS) e propor uma estratégia preditiva capaz de antecipar o score antes da aplicação da pesquisa.

---

## Descrição da base de dados

Arquivo CSV com dados históricos de pedidos, entregas e interações com atendimento.

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
| `resolution_time_days` | Tempo de resolução de problemas |
| `complaints_count` | Número de reclamações |
| `repeat_purchase_30d` | Recompra em 30 dias (0/1) |
| `csat_internal_score` | Score interno de satisfação |
| `nps_score` | Nota NPS (0–10) |

---

## 📁 Estrutura do Repositório

```bash
├── data/
│   └── desafio_nps_fase_1.csv
├── notebooks/
│   ├── analise_exploratoria_NPS.ipynb
│   └── modelo_regressao_nps.ipynb
├── images/
├── models/
│   ├── model.pkl
│   └── scaler.pkl
├── docs/
├── app.py
├── requirements.txt
└── README.md
```

---

## Metodologia

1. **Entendimento do negócio** — definição do problema e impacto do NPS no e-commerce  
2. **Definição da target** — análise da variável `nps_score`  
3. **EDA** — análise exploratória orientada a negócio  
4. **Modelagem preditiva** — regressão para previsão de NPS  

---

## Executando os notebooks

```bash
# Clone o repositório
git clone https://github.com/LucasAlexSant/Tech-Challenge-Fase-1-FIAP.git

cd Tech-Challenge-Fase-1-FIAP

# Instale dependências
pip install -r requirements.txt

# Execute o Jupyter
jupyter notebook
```

Abra e execute, nesta ordem:

1. `notebooks/analise_exploratoria_NPS.ipynb`
2. `notebooks/modelo_regressao_nps.ipynb`

---

# API REST

A aplicação disponibiliza uma API Flask para previsão de NPS individual e em lote.

## Como executar a API

```bash
# Execute a aplicação
python app.py
```

A API será iniciada em:

```bash
http://localhost:5000
```

---

## Documentação Swagger

A documentação interativa está disponível em:

```bash
http://localhost:5000/docs
```

Permite:

- visualizar endpoints
- testar requisições
- validar payloads
- consultar exemplos de resposta

---

## Endpoints disponíveis

### Health Check

```http
GET /health
```

Resposta:

```json
{
  "status": "ok"
}
```

---

### Predição individual

```http
POST /predict
```

Payload de exemplo:

```json
{
  "customer_age": 35,
  "customer_region": "Nordeste",
  "customer_tenure_months": 14,
  "order_value": 139.73,
  "items_quantity": 5,
  "discount_value": 20.0,
  "payment_installments": 3,
  "delivery_time_days": 2,
  "delivery_delay_days": 0,
  "freight_value": 15.5,
  "delivery_attempts": 1,
  "customer_service_contacts": 0,
  "resolution_time_days": 0,
  "complaints_count": 0
}
```

Resposta:

```json
{
  "nps_previsto": 7.45,
  "categoria": "Neutro"
}
```

---

### Predição em lote

```http
POST /predict/batch
```

Recebe lista de registros JSON.

Resposta:

```json
[
  {
    "nps_previsto": 9.42,
    "categoria": "Promotor"
  },
  {
    "nps_previsto": 2.31,
    "categoria": "Detrator"
  }
]
```

---

## Tecnologias utilizadas

- Python
- Flask
- Flasgger
- Pandas
- Scikit-learn
- Joblib
- Jupyter Notebook
