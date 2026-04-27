import pandas as pd


class TratamentoDados:
    """
    Classe responsável pelo tratamento inicial dos dados do projeto NPS.

    Objetivo:
    - centralizar regras de preparação dos dados
    - permitir reaproveitamento em notebooks futuros
    - manter o notebook mais limpo
    """

    def __init__(self, df):
        self.df_original = df.copy()
        self.df = df.copy()

    def diagnostico_inicial(self, mostrar_saida=True):
        """
        Exibe informações básicas da base.
        """
        diagnostico = {
            "linhas": self.df.shape[0],
            "colunas": self.df.shape[1],
            "duplicados": self.df.duplicated().sum(),
            "nulos_por_coluna": self.df.isnull().sum()
        }

        if mostrar_saida:
            print("=" * 60)
            print("DIAGNÓSTICO INICIAL")
            print("=" * 60)
            print(f"Linhas: {diagnostico['linhas']}")
            print(f"Colunas: {diagnostico['colunas']}")
            print(f"Duplicados: {diagnostico['duplicados']}")
            print("\nValores nulos por coluna:")
            print(diagnostico["nulos_por_coluna"])

        return diagnostico

    def remover_duplicados(self, mostrar_saida=True):
        """
        Remove linhas duplicadas.
        """
        linhas_antes = len(self.df)
        self.df = self.df.drop_duplicates().copy()
        linhas_depois = len(self.df)

        if mostrar_saida:
            print("=" * 60)
            print("REMOÇÃO DE DUPLICADOS")
            print("=" * 60)
            print(f"Linhas antes: {linhas_antes}")
            print(f"Linhas depois: {linhas_depois}")
            print(f"Duplicados removidos: {linhas_antes - linhas_depois}")

        return self.df

    def tratar_nulos_simples(self, mostrar_saida=True):
        """
        Tratamento simples de nulos:
        - numéricos: mediana
        - categóricos: moda
        """
        colunas_numericas = self.df.select_dtypes(include="number").columns
        colunas_categoricas = self.df.select_dtypes(exclude="number").columns

        for coluna in colunas_numericas:
            if self.df[coluna].isnull().sum() > 0:
                self.df[coluna] = self.df[coluna].fillna(self.df[coluna].median())

        for coluna in colunas_categoricas:
            if self.df[coluna].isnull().sum() > 0:
                self.df[coluna] = self.df[coluna].fillna(self.df[coluna].mode()[0])

        if mostrar_saida:
            print("=" * 60)
            print("TRATAMENTO DE NULOS")
            print("=" * 60)
            print("Nulos restantes:")
            print(self.df.isnull().sum()[self.df.isnull().sum() > 0])

        return self.df

    def remover_colunas(self, colunas, mostrar_saida=True):
        """
        Remove colunas informadas.
        """
        colunas_existentes = [col for col in colunas if col in self.df.columns]

        self.df = self.df.drop(columns=colunas_existentes).copy()

        if mostrar_saida:
            print("=" * 60)
            print("REMOÇÃO DE COLUNAS")
            print("=" * 60)
            print(f"Colunas removidas: {colunas_existentes}")

        return self.df

    def criar_classe_nps(
        self,
        coluna_nps="nps_score",
        nova_coluna="classe_nps",
        mostrar_saida=True
    ):
        """
        Cria a classe NPS:
        - 0 a 6 = Detrator
        - 7 a 8 = Neutro
        - 9 a 10 = Promotor
        """
        def classificar_nps(valor):
            if valor <= 6:
                return "Detrator"
            elif valor <= 8:
                return "Neutro"
            else:
                return "Promotor"

        self.df[nova_coluna] = self.df[coluna_nps].apply(classificar_nps)

        if mostrar_saida:
            print("=" * 60)
            print("CRIAÇÃO DA CLASSE NPS")
            print("=" * 60)
            print(self.df[nova_coluna].value_counts())

        return self.df

    def criar_classe_csat(
        self,
        coluna_csat="csat_internal_score",
        nova_coluna="classe_csat",
        mostrar_saida=True
    ):
        """
        Cria a classificação de CSAT:

        - 0 a 2  -> Muito insatisfeito
        - 2 a 4  -> Insatisfeito
        - 4 a 6  -> Indiferente
        - 6 a 8  -> Satisfeito
        - 8 a 10 -> Muito satisfeito
        """
        def classificar_csat(valor):
            if valor <= 2:
                return "Muito insatisfeito"
            elif valor <= 4:
                return "Insatisfeito"
            elif valor <= 6:
                return "Indiferente"
            elif valor <= 8:
                return "Satisfeito"
            else:
                return "Muito satisfeito"

        self.df[nova_coluna] = self.df[coluna_csat].apply(classificar_csat)

        if mostrar_saida:
            print("=" * 60)
            print("CRIAÇÃO DA CLASSE CSAT")
            print("=" * 60)
            print(f"Coluna utilizada: {coluna_csat}")
            print("\nDistribuição da classificação:")
            print(self.df[nova_coluna].value_counts())

        return self.df

    def obter_dataframe(self):
        """
        Retorna uma cópia da base tratada.
        """
        return self.df.copy()
    
    def preparar_base_modelagem(
        self,
        coluna_alvo="classe_nps",
        colunas_remover=[],
        drop_first=True,
        mostrar_saida=True
    ):
        """
        Prepara a base para modelagem preditiva.

        Parâmetros:
        - coluna_alvo: variável target do modelo
        - colunas_remover: lista de colunas que não devem participar
        do treinamento (informar manualmente no notebook)
        - drop_first: remove a primeira dummy para evitar multicolinearidade
        - mostrar_saida: exibe informações da preparação

        Etapas:
        - remove colunas informadas
        - separa X e y
        - aplica get_dummies nas variáveis categóricas
        """

        df_modelo = self.df.copy()

        # Remove apenas colunas que realmente existem
        colunas_existentes = [
            col for col in colunas_remover
            if col in df_modelo.columns
        ]

        df_modelo = df_modelo.drop(
            columns=colunas_existentes,
            errors="ignore"
        )

        # Separa target
        if coluna_alvo not in df_modelo.columns:
            raise ValueError(
                f"A coluna alvo '{coluna_alvo}' não existe no dataframe."
            )

        y = df_modelo[coluna_alvo].copy()

        # Remove target de X
        X = df_modelo.drop(columns=[coluna_alvo]).copy()

        # Converte variáveis categóricas
        X = pd.get_dummies(
            X,
            drop_first=drop_first
        )

        if mostrar_saida:
            print("=" * 60)
            print("PREPARAÇÃO DA BASE PARA MODELAGEM")
            print("=" * 60)
            print(f"Variável alvo: {coluna_alvo}")
            print(f"Colunas removidas: {colunas_existentes}")
            print(f"Shape de X: {X.shape}")
            print(f"Shape de y: {y.shape}")
            print("\nPrimeiras colunas de X:")
            print(X.columns.tolist()[:20])

        return X, y