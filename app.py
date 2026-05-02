from flask import Flask, request, jsonify
from flasgger import Swagger
import pandas as pd
import joblib

app = Flask(__name__)

swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": "apispec",
            "route": "/apispec.json",
            "rule_filter": lambda rule: True,
            "model_filter": lambda tag: True,
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/docs",
}

swagger_template = {
    "info": {
        "title": "NPS Prediction API",
        "description": "Previsão de NPS Score a partir de dados operacionais do cliente.",
    },
    "tags": [
        {"name": "Predição", "description": "Endpoints de previsão de NPS"},
        {"name": "Saúde",    "description": "Status da API"},
    ],
}

swagger = Swagger(app, config=swagger_config, template=swagger_template)

# ── Artefatos do modelo ────────────────────────────────────────────────────────
modelo = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")

MAP_REGIAO = {
    "Centro-Oeste": 0,
    "Nordeste":     1,
    "Norte":        2,
    "Sudeste":      3,
    "Sul":          4,
}

COLS_MODELO = [
    "customer_age",
    "customer_region",
    "customer_tenure_months",
    "order_value",
    "items_quantity",
    "discount_value",
    "payment_installments",
    "delivery_time_days",
    "delivery_delay_days",
    "freight_value",
    "delivery_attempts",
    "customer_service_contacts",
    "resolution_time_days",
    "complaints_count",
]


def classificar_nps(nota: float) -> str:
    if nota <= 6:
        return "Detrator"
    elif nota <= 8:
        return "Neutro"
    return "Promotor"


def processar_registro(reg: dict) -> dict:
    df = pd.DataFrame([reg])
    if "customer_region" in df.columns:
        df["customer_region"] = df["customer_region"].map(MAP_REGIAO)
    for col in COLS_MODELO:
        if col not in df.columns:
            df[col] = 0
    X_sc = scaler.transform(df[COLS_MODELO])
    pred = float(modelo.predict(X_sc)[0])
    pred = round(max(0.0, min(10.0, pred)), 2)
    return {"nps_previsto": pred, "categoria": classificar_nps(pred)}


# ── Rotas ──────────────────────────────────────────────────────────────────────

@app.route("/predict", methods=["POST"])
def predict():
    """
    Previsão de NPS para um único cliente.
    ---
    tags:
      - Predição
    consumes:
      - application/json
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - customer_region
            - customer_tenure_months
            - order_value
          properties:
            customer_age:
              type: integer
              example: 35
              description: Idade do cliente
            customer_region:
              type: string
              enum: [Norte, Nordeste, Centro-Oeste, Sudeste, Sul]
              example: Nordeste
            customer_tenure_months:
              type: integer
              example: 14
            order_value:
              type: number
              example: 139.73
            items_quantity:
              type: integer
              example: 50
            discount_value:
              type: number
              example: 90.35
            payment_installments:
              type: integer
              example: 4
            delivery_time_days:
              type: integer
              example: 2
            delivery_delay_days:
              type: integer
              example: 0
            freight_value:
              type: number
              example: 55.53
            delivery_attempts:
              type: integer
              example: 3
            customer_service_contacts:
              type: integer
              example: 0
            resolution_time_days:
              type: integer
              example: 4
            complaints_count:
              type: integer
              example: 3
    responses:
      200:
        description: NPS previsto e categoria do cliente
        schema:
          type: object
          properties:
            nps_previsto:
              type: number
              example: 8.45
            categoria:
              type: string
              example: Neutro
      400:
        description: Erro de processamento
        schema:
          type: object
          properties:
            erro:
              type: string
    """
    try:
        data = request.get_json(force=True)
        return jsonify(processar_registro(data))
    except Exception as e:
        return jsonify({"erro": str(e)}), 400


@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    """
    Previsão de NPS para múltiplos clientes em lote.
    ---
    tags:
      - Predição
    summary: Predição batch de NPS
    description: |
      Recebe uma lista de clientes em formato JSON e retorna a previsão
      de NPS para cada registro.

      Use este endpoint para inferência em lote, como processamento de
      múltiplos pedidos ou clientes simultaneamente.

    consumes:
      - application/json

    parameters:
      - in: body
        name: body
        required: true
        description: Lista de clientes para previsão
        schema:
          type: array
          items:
            type: object
            required:
              - customer_age
              - customer_region
              - customer_tenure_months
              - order_value
            properties:
              customer_age:
                type: integer
                example: 35
                description: Idade do cliente

              customer_region:
                type: string
                enum: [Norte, Nordeste, Centro-Oeste, Sudeste, Sul]
                example: Sul
                description: Região do cliente

              customer_tenure_months:
                type: integer
                example: 60
                description: Tempo de relacionamento em meses

              order_value:
                type: number
                example: 250.00

              items_quantity:
                type: integer
                example: 3

              discount_value:
                type: number
                example: 20.50

              payment_installments:
                type: integer
                example: 4

              delivery_time_days:
                type: integer
                example: 3

              delivery_delay_days:
                type: integer
                example: 0

              freight_value:
                type: number
                example: 18.90

              delivery_attempts:
                type: integer
                example: 1

              customer_service_contacts:
                type: integer
                example: 0

              resolution_time_days:
                type: integer
                example: 0

              complaints_count:
                type: integer
                example: 0

          example:
            - customer_age: 35
              customer_region: Sudeste
              customer_tenure_months: 24
              order_value: 300.0
              items_quantity: 2
              discount_value: 10.0
              payment_installments: 3
              delivery_time_days: 2
              delivery_delay_days: 0
              freight_value: 15.0
              delivery_attempts: 1
              customer_service_contacts: 0
              resolution_time_days: 0
              complaints_count: 0

            - customer_age: 50
              customer_region: Norte
              customer_tenure_months: 6
              order_value: 89.9
              items_quantity: 1
              discount_value: 0
              payment_installments: 8
              delivery_time_days: 9
              delivery_delay_days: 3
              freight_value: 40.0
              delivery_attempts: 3
              customer_service_contacts: 4
              resolution_time_days: 7
              complaints_count: 2

    responses:
      200:
        description: Lista de previsões gerada com sucesso
        schema:
          type: array
          items:
            type: object
            properties:
              nps_previsto:
                type: number
                example: 8.42
              categoria:
                type: string
                example: Promotor

      400:
        description: Erro de validação ou processamento
        schema:
          type: object
          properties:
            erro:
              type: string
              example: Esperado um array JSON.
    """
    try:
        data = request.get_json(force=True)

        if not isinstance(data, list):
            return jsonify({"erro": "Esperado um array JSON."}), 400

        return jsonify([processar_registro(r) for r in data])

    except Exception as e:
        return jsonify({"erro": str(e)}), 400


@app.route("/health", methods=["GET"])
def health():
    """
    Verifica se a API está no ar.
    ---
    tags:
      - Saúde
    responses:
      200:
        description: API operacional
        schema:
          type: object
          properties:
            status:
              type: string
              example: ok
    """
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)