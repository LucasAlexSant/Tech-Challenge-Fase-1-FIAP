import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.colors import LinearSegmentedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier

# =========================================================
# PALETAS VISUAIS DO PROJETO
# =========================================================

# Paleta principal do NPS
# Detrator  → vermelho
# Neutro    → amarelo
# Promotor  → verde água

NPS_COLORS = {
    "Detrator": "#e74c6f",
    "Neutro": "#f0a500",
    "Promotor": "#2ec4b6"
}

# Paleta complementar para análises por região
REGION_PAL = [
    "#7c4dff",
    "#00bcd4",
    "#ff6f61",
    "#ffd54f",
    "#69f0ae"
]

# Paleta divergente padrão para gráficos comparativos
DIVERGE_CMAP = "RdYlGn"

# Heatmap customizado usando a paleta do NPS
HEAT_CMAP = LinearSegmentedColormap.from_list(
    "nps",
    [
        "#e74c6f",  # Detrator
        "#f0a500",  # Neutro
        "#2ec4b6"   # Promotor
    ]
)


class AnalisePreditiva:
    """
    Classe responsável pela etapa de modelagem preditiva.
    """

    def __init__(self, X, y):
        self.X = X.copy()
        self.y = y.copy()

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.modelo = None
        self.nome_modelo = None
        self.y_pred = None

        self.resultados_modelos = []

    def dividir_treino_teste(
        self,
        test_size=0.2,
        random_state=42,
        stratify=True,
        mostrar_saida=True
    ):
        """
        Divide os dados em treino e teste.
        """

        stratify_param = self.y if stratify else None

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_param
        )

        if mostrar_saida:
            print("=" * 60)
            print("DIVISÃO TREINO E TESTE")
            print("=" * 60)
            print(f"X_train: {self.X_train.shape}")
            print(f"X_test:  {self.X_test.shape}")
            print(f"y_train: {self.y_train.shape}")
            print(f"y_test:  {self.y_test.shape}")

        return self.X_train, self.X_test, self.y_train, self.y_test

    def calcular_metricas(self, nome_modelo, y_true, y_pred):
        """
        Calcula as principais métricas de classificação.
        """

        metricas = {
            "modelo": nome_modelo,
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_macro": precision_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0
            ),
            "recall_macro": recall_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0
            ),
            "f1_macro": f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0
            ),
            "precision_weighted": precision_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0
            ),
            "recall_weighted": recall_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0
            ),
            "f1_weighted": f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0
            )
        }

        return metricas

    def treinar_random_forest(
        self,
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        mostrar_saida=True
    ):
        """
        Treina um modelo Random Forest para classificação do NPS.
        """

        if self.X_train is None:
            self.dividir_treino_teste(
                random_state=random_state,
                mostrar_saida=mostrar_saida
            )

        self.modelo = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            class_weight=class_weight
        )

        self.nome_modelo = "Random Forest"

        self.modelo.fit(self.X_train, self.y_train)
        self.y_pred = self.modelo.predict(self.X_test)

        metricas = self.calcular_metricas(
            self.nome_modelo,
            self.y_test,
            self.y_pred
        )

        self.resultados_modelos.append(metricas)

        if mostrar_saida:
            print("=" * 60)
            print("MODELO — RANDOM FOREST")
            print("=" * 60)
            print(f"Accuracy: {metricas['accuracy']:.4f}")
            print(f"Precision Macro: {metricas['precision_macro']:.4f}")
            print(f"Recall Macro: {metricas['recall_macro']:.4f}")
            print(f"F1 Macro: {metricas['f1_macro']:.4f}")

            print("\nRelatório de classificação:")
            print(classification_report(self.y_test, self.y_pred))

        return self.modelo

    def obter_matriz_confusao(self):
        """
        Retorna a matriz de confusão do último modelo treinado.
        """

        if self.y_pred is None:
            raise ValueError("Nenhum modelo foi treinado ainda.")

        return confusion_matrix(self.y_test, self.y_pred)

    def obter_resultados(self):
        """
        Retorna os resultados dos modelos treinados.
        """

        return pd.DataFrame(self.resultados_modelos)

    def analisar_previsoes(
        self,
        labels=None,
        mostrar_saida=True,
        qtd_exibir=20
    ):
        """
        Analisa as previsões do último modelo treinado.
        """

        if self.y_pred is None:
            raise ValueError("Nenhum modelo foi treinado ainda.")

        if labels is None:
            labels = ["Detrator", "Neutro", "Promotor"]

        comparacao_previsoes = pd.DataFrame({
            "real": self.y_test.values,
            "previsto": self.y_pred
        })

        comparacao_previsoes["acertou"] = (
            comparacao_previsoes["real"] == comparacao_previsoes["previsto"]
        )

        total_registros = len(comparacao_previsoes)
        total_acertos = comparacao_previsoes["acertou"].sum()
        accuracy_manual = total_acertos / total_registros

        matriz = confusion_matrix(
            self.y_test,
            self.y_pred,
            labels=labels
        )

        matriz_confusao_df = pd.DataFrame(
            matriz,
            index=[f"Real {label}" for label in labels],
            columns=[f"Previsto {label}" for label in labels]
        )

        relatorio = classification_report(
            self.y_test,
            self.y_pred,
            output_dict=True,
            zero_division=0
        )

        relatorio_df = pd.DataFrame(relatorio).T

        if mostrar_saida:
            print("=" * 60)
            print("ANÁLISE DAS PREVISÕES DO MODELO")
            print("=" * 60)

            print(f"Modelo analisado: {self.nome_modelo}")

            print("\nAmostra de previsões:")
            print(comparacao_previsoes.head(qtd_exibir))

            print("\nResumo da acurácia manual:")
            print(f"Total de registros testados: {total_registros}")
            print(f"Total de acertos: {total_acertos}")
            print(f"Acurácia manual: {accuracy_manual:.4f}")

            print("\nMatriz de confusão:")
            print(matriz_confusao_df)

            print("\nRelatório de classificação:")
            print(relatorio_df)

        return {
            "comparacao_previsoes": comparacao_previsoes,
            "accuracy_manual": accuracy_manual,
            "matriz_confusao": matriz_confusao_df,
            "relatorio_classificacao": relatorio_df
        }

    def exportar_matriz_confusao(
        self,
        labels=None,
        salvar_em="../images/matriz_confusao_random_forest.png",
        figsize=(8, 6),
        mostrar_saida=True
    ):
        """
        Gera e exporta o gráfico da matriz de confusão.
        """

        if self.y_pred is None:
            raise ValueError("Nenhum modelo foi treinado ainda.")

        if labels is None:
            labels = ["Detrator", "Neutro", "Promotor"]

        matriz = confusion_matrix(
            self.y_test,
            self.y_pred,
            labels=labels
        )

        matriz_df = pd.DataFrame(
            matriz,
            index=[f"Real {label}" for label in labels],
            columns=[f"Previsto {label}" for label in labels]
        )

        plt.figure(figsize=figsize)

        sns.heatmap(
            matriz_df,
            annot=True,
            fmt="d",
            cmap=HEAT_CMAP,
            linewidths=0.5,
            cbar=True,

            # melhora visual dos números dentro da matriz
            annot_kws={
                "size": 12,
                "weight": "bold"
            },

            # adiciona nome da barra lateral
            cbar_kws={
                "label": "Quantidade de Registros"
            }
        )

        plt.title(
            f"Matriz de Confusão — {self.nome_modelo}",
            fontsize=14,
            fontweight="bold"
        )

        plt.xlabel("Classe Prevista")
        plt.ylabel("Classe Real")

        plt.tight_layout()

        plt.savefig(
            salvar_em,
            dpi=300,
            bbox_inches="tight"
        )

        plt.show()

        if mostrar_saida:
            print("=" * 60)
            print("MATRIZ DE CONFUSÃO EXPORTADA")
            print("=" * 60)
            print(f"Arquivo salvo em: {salvar_em}")

        return matriz_df


    def treinar_regressao_logistica(
        self,
        random_state=42,
        class_weight="balanced",
        max_iter=2000,
        mostrar_saida=True
    ):
        """
        Treina um modelo de Regressão Logística para classificação do NPS.

        Objetivo de negócio:
        - comparar um modelo mais simples e interpretável
        com o Random Forest
        - avaliar se ele melhora principalmente a classe Neutro

        A Regressão Logística:
        - é mais simples
        - mais explicável
        - muito usada em ambientes corporativos
        - excelente benchmark para comparação

        Parâmetros:
        - random_state: reprodutibilidade
        - class_weight: ajuda no balanceamento das classes
        - max_iter: evita erro de convergência
        - mostrar_saida: exibe os resultados no notebook
        """

        # Caso ainda não exista divisão treino/teste,
        # a função executa automaticamente
        if self.X_train is None:
            self.dividir_treino_teste(
                random_state=random_state,
                mostrar_saida=mostrar_saida
            )

        # Instancia o modelo
        self.modelo = LogisticRegression(
            random_state=random_state,
            class_weight=class_weight,
            max_iter=max_iter
        )

        # Nome usado na tabela comparativa final
        self.nome_modelo = "Regressão Logística"

        # Treinamento do modelo
        self.modelo.fit(
            self.X_train,
            self.y_train
        )

        # Geração das previsões
        self.y_pred = self.modelo.predict(
            self.X_test
        )

        # Cálculo das métricas
        metricas = self.calcular_metricas(
            self.nome_modelo,
            self.y_test,
            self.y_pred
        )

        # Salva para comparação futura
        self.resultados_modelos.append(metricas)

        # Exibição no notebook
        if mostrar_saida:
            print("=" * 60)
            print("MODELO — REGRESSÃO LOGÍSTICA")
            print("=" * 60)

            print(f"Accuracy: {metricas['accuracy']:.4f}")
            print(f"Precision Macro: {metricas['precision_macro']:.4f}")
            print(f"Recall Macro: {metricas['recall_macro']:.4f}")
            print(f"F1 Macro: {metricas['f1_macro']:.4f}")

            print("\nRelatório de classificação:")
            print(
                classification_report(
                    self.y_test,
                    self.y_pred
                )
            )

        return self.modelo

    def treinar_svm(
        self,
        kernel="rbf",
        C=1.0,
        gamma="scale",
        class_weight="balanced",
        random_state=42,
        mostrar_saida=True
    ):
        """
        Treina um modelo SVM para classificação do NPS.

        Objetivo de negócio:
        - testar um modelo capaz de separar classes usando fronteiras não lineares
        - comparar o comportamento do SVM com modelos de árvore e Regressão Logística

        Observação importante:
        - O SVM pode ser sensível à escala das variáveis.
        - Por isso, também existe o método treinar_svm_com_scaler.
        """

        if self.X_train is None:
            self.dividir_treino_teste(
                random_state=random_state,
                mostrar_saida=mostrar_saida
            )

        self.modelo = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            class_weight=class_weight,
            random_state=random_state
        )

        self.nome_modelo = "SVM"

        self.modelo.fit(
            self.X_train,
            self.y_train
        )

        self.y_pred = self.modelo.predict(
            self.X_test
        )

        metricas = self.calcular_metricas(
            self.nome_modelo,
            self.y_test,
            self.y_pred
        )

        self.resultados_modelos.append(metricas)

        if mostrar_saida:
            print("=" * 60)
            print("MODELO — SVM")
            print("=" * 60)
            print(f"Accuracy: {metricas['accuracy']:.4f}")
            print(f"Precision Macro: {metricas['precision_macro']:.4f}")
            print(f"Recall Macro: {metricas['recall_macro']:.4f}")
            print(f"F1 Macro: {metricas['f1_macro']:.4f}")

            print("\nRelatório de classificação:")
            print(
                classification_report(
                    self.y_test,
                    self.y_pred,
                    zero_division=0
                )
            )

        return self.modelo

    def treinar_svm_com_scaler(
        self,
        kernel="rbf",
        C=1.0,
        gamma="scale",
        class_weight="balanced",
        random_state=42,
        mostrar_saida=True
    ):
        """
        Treina um modelo SVM com StandardScaler.

        Objetivo de negócio:
        - avaliar o SVM após padronizar as variáveis numéricas
        - reduzir distorções causadas por variáveis em escalas diferentes

        Boa prática:
        - o scaler fica dentro de um Pipeline para evitar vazamento de dados
        entre treino e teste.
        """

        if self.X_train is None:
            self.dividir_treino_teste(
                random_state=random_state,
                mostrar_saida=mostrar_saida
            )

        self.modelo = Pipeline(steps=[
            (
                "scaler",
                StandardScaler()
            ),
            (
                "svm",
                SVC(
                    kernel=kernel,
                    C=C,
                    gamma=gamma,
                    class_weight=class_weight,
                    random_state=random_state
                )
            )
        ])

        self.nome_modelo = "SVM + StandardScaler"

        self.modelo.fit(
            self.X_train,
            self.y_train
        )

        self.y_pred = self.modelo.predict(
            self.X_test
        )

        metricas = self.calcular_metricas(
            self.nome_modelo,
            self.y_test,
            self.y_pred
        )

        self.resultados_modelos.append(metricas)

        if mostrar_saida:
            print("=" * 60)
            print("MODELO — SVM + STANDARDSCALER")
            print("=" * 60)
            print(f"Accuracy: {metricas['accuracy']:.4f}")
            print(f"Precision Macro: {metricas['precision_macro']:.4f}")
            print(f"Recall Macro: {metricas['recall_macro']:.4f}")
            print(f"F1 Macro: {metricas['f1_macro']:.4f}")

            print("\nRelatório de classificação:")
            print(
                classification_report(
                    self.y_test,
                    self.y_pred,
                    zero_division=0
                )
            )

        return self.modelo

    def treinar_extra_trees(
        self,
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        mostrar_saida=True
    ):
        """
        Treina um modelo Extra Trees para classificação do NPS.

        Objetivo de negócio:
        - comparar outro modelo baseado em várias árvores com o Random Forest
        - avaliar se a maior aleatoriedade das árvores melhora a generalização

        Diferença prática:
        - Random Forest cria árvores com amostras e divisões mais controladas.
        - Extra Trees adiciona mais aleatoriedade na escolha dos cortes.
        """

        if self.X_train is None:
            self.dividir_treino_teste(
                random_state=random_state,
                mostrar_saida=mostrar_saida
            )

        self.modelo = ExtraTreesClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            class_weight=class_weight
        )

        self.nome_modelo = "Extra Trees"

        self.modelo.fit(
            self.X_train,
            self.y_train
        )

        self.y_pred = self.modelo.predict(
            self.X_test
        )

        metricas = self.calcular_metricas(
            self.nome_modelo,
            self.y_test,
            self.y_pred
        )

        self.resultados_modelos.append(metricas)

        if mostrar_saida:
            print("=" * 60)
            print("MODELO — EXTRA TREES")
            print("=" * 60)
            print(f"Accuracy: {metricas['accuracy']:.4f}")
            print(f"Precision Macro: {metricas['precision_macro']:.4f}")
            print(f"Recall Macro: {metricas['recall_macro']:.4f}")
            print(f"F1 Macro: {metricas['f1_macro']:.4f}")

            print("\nRelatório de classificação:")
            print(
                classification_report(
                    self.y_test,
                    self.y_pred,
                    zero_division=0
                )
            )

        return self.modelo

    def treinar_arvore_decisao(
        self,
        random_state=42,
        class_weight="balanced",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        mostrar_saida=True
    ):
        """
        Treina um modelo de Árvore de Decisão para classificação do NPS.

        Objetivo de negócio:
        - usar um modelo simples e interpretável como comparação
        - entender regras de decisão que separam Detratores, Neutros e Promotores

        Atenção:
        - árvores isoladas podem sofrer overfitting.
        - por isso, ela deve ser comparada com modelos mais robustos,
        como Random Forest e Extra Trees.
        """

        if self.X_train is None:
            self.dividir_treino_teste(
                random_state=random_state,
                mostrar_saida=mostrar_saida
            )

        self.modelo = DecisionTreeClassifier(
            random_state=random_state,
            class_weight=class_weight,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf
        )

        self.nome_modelo = "Árvore de Decisão"

        self.modelo.fit(
            self.X_train,
            self.y_train
        )

        self.y_pred = self.modelo.predict(
            self.X_test
        )

        metricas = self.calcular_metricas(
            self.nome_modelo,
            self.y_test,
            self.y_pred
        )

        self.resultados_modelos.append(metricas)

        if mostrar_saida:
            print("=" * 60)
            print("MODELO — ÁRVORE DE DECISÃO")
            print("=" * 60)
            print(f"Accuracy: {metricas['accuracy']:.4f}")
            print(f"Precision Macro: {metricas['precision_macro']:.4f}")
            print(f"Recall Macro: {metricas['recall_macro']:.4f}")
            print(f"F1 Macro: {metricas['f1_macro']:.4f}")

            print("\nRelatório de classificação:")
            print(
                classification_report(
                    self.y_test,
                    self.y_pred,
                    zero_division=0
                )
            )

        return self.modelo

