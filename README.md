# 🔍 Detecção de Fraudes em Transações Financeiras

## 📌 Contexto e Objetivos

Fraudes financeiras causam prejuízos bilionários todos os anos. O desafio de detectá‑las automaticamente, porém, esbarra em um problema central: **menos de 1% das transações são fraudulentas**. Modelos tradicionais que miram apenas a acurácia podem simplesmente “chutar” que tudo é normal e ainda assim acertar 99% dos casos – sem jamais encontrar uma fraude.

**Meu objetivo** com este projeto é:

- Construir um pipeline completo de machine learning para identificar transações fraudulentas.
- Aprender a lidar com **classes extremamente desbalanceadas** usando técnicas de reamostragem e pesos.
- Avaliar o modelo com métricas adequadas (Recall, Precision, F1‑Score, ROC AUC).
- Gerar um guia reutilizável para futuros projetos de detecção de anomalias.

O dataset utilizado é o [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) do Kaggle.

---

## 📚 Curadoria de Fontes

| Fonte | Link |
|-------|------|
| Dataset original (Kaggle) | [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| Documentação do XGBoost | [XGBoost Parameters](https://xgboost.readthedocs.io/en/stable/parameter.html) |
| Paper sobre SMOTE | [SMOTE: Synthetic Minority Over-sampling Technique](https://arxiv.org/abs/1106.1813) |
| Documentação imbalanced-learn | [imbalanced-learn.org](https://imbalanced-learn.org/stable/) |

---

## 🛠️ Engenharia de Prompts e “Cicatrizes”

Nesta seção documento **o raciocínio por trás de cada etapa** do notebook, as perguntas que me fiz, os trechos de código e as dificuldades encontradas (as tais “cicatrizes”).

### 1️⃣ Primeiro contato com os dados
**Pergunta norteadora:** *Qual a cara dos dados? As fraudes são mesmo tão raras?*

```python
import pandas as pd

url = "https://raw.githubusercontent.com/.../creditcard.csv"  # substituir pelo caminho real
df = pd.read_csv(url)
df.head()
```

**Cicatriz:** O dataset tem 284.807 linhas; no Jupyter online, exibir tudo pode travar. Limite a visualização com `.head()`.

---

### 2️⃣ Verificando o desbalanceamento
**Pergunta:** *Se eu treinar um modelo agora, ele vai aprender a encontrar fraudes?*

```python
# Proporção das classes (0 = normal, 1 = fraude)
print(df["Class"].value_counts(normalize=True))
```

**Cicatriz:** O resultado mostra que apenas ≈0.17% das transações são fraude. Um modelo que sempre chuta “normal” teria **99.83% de acurácia** – e seria inútil. Isso me fez abandonar a acurácia como métrica principal e focar em **Recall**.

---

### 3️⃣ Feature Engineering

**Transformação logarítmica do valor**  
*O Amount tem uma distribuição muito assimétrica; o log ajuda a comprimir outliers.*

```python
import numpy as np

df["Amount_log"] = np.log1p(df["Amount"])
```

**Escalonamento**  
*Algoritmos como Regressão Logística e XGBoost se beneficiam de features na mesma escala.*

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df["Amount_scaled"] = scaler.fit_transform(df[["Amount"]])
```

**Cicatriz importante:**  
Eu escalei **antes** de separar treino e teste – isso vaza informação do teste para o treino. Em versões posteriores corrigi usando um **Pipeline** (veja etapa 9). Guarde essa cicatriz como aprendizado.

**Divisão estratificada**  
*Como as fraudes são raras, `stratify=y` garante que a proporção de classes seja mantida nos dois conjuntos.*

```python
from sklearn.model_selection import train_test_split

X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.3, random_state=42
)
```

---

### 4️⃣ Modelo baseline: Regressão Logística

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

**Cicatriz:** Apareceu um aviso de que o número de iterações (`max_iter`) poderia ser insuficiente. Ignorei a princípio, mas depois descobri que o **pipeline** com scaler ajudou na convergência.

**Relatório de classificação**

```python
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
```

A saída revela:  
- **Acurácia** ≈ 99.9%  
- **Recall da classe 1 (fraude)** → muito baixo (ex.: 0.60).  

**Ou seja, o modelo errou quase metade das fraudes.** É exatamente por isso que precisamos de métricas melhores.

---

### 5️⃣ Curva ROC e AUC

*Para visualizar o trade‑off entre acertar fraudes (TPR) e alarmar falsos positivos (FPR).*

```python
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

y_probs = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_probs)

plt.plot(fpr, tpr)
plt.title("Curva ROC – Regressão Logística")
plt.xlabel("Taxa de Falsos Positivos")
plt.ylabel("Taxa de Verdadeiros Positivos")
plt.show()

print("AUC:", roc_auc_score(y_test, y_probs))
```

---

### 6️⃣ Balanceamento manual (undersampling)

*Remover transações normais para igualar o número de fraudes.*

```python
fraudes = df[df["Class"] == 1]
normais = df[df["Class"] == 0].sample(len(fraudes), random_state=42)
df_under = pd.concat([fraudes, normais])
```

**Cicatriz:** Undersampling joga fora 99% dos dados – o modelo perde muita informação. Usei apenas para testar o conceito.

---

### 7️⃣ Oversampling com SMOTE

*Cria exemplos sintéticos da classe minoritária.*

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE()
X_res, y_res = smote.fit_resample(X, y)
```

**Cicatriz:** SMOTE pode gerar ruído se aplicado antes da divisão treino/teste. O correto é aplicar **apenas no conjunto de treino**. No pipeline final (etapa 9) resolvi isso com `class_weight` e `scale_pos_weight`.

---

### 8️⃣ Modelos mais robustos

**Random Forest** (com peso balanceado)

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=50,
    max_depth=10,               # corrigido de "1-"
    class_weight="balanced",
    n_jobs=-1,
    random_state=42             # corrigido de "random_stat"
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print(classification_report(y_test, y_pred_rf))
```

**XGBoost** (com ajuste de desbalanceamento)

```python
from xgboost import XGBClassifier

xgb = XGBClassifier(
    scale_pos_weight=10,        # ajuda no desbalanceamento
    use_label_encoder=False,    # corrigido de "use_label_encooder"
    eval_metric="logloss"
)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
print(classification_report(y_test, y_pred_xgb))
```

---

### 9️⃣ Pipeline profissional (corrigindo o data leakage)

```python
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
```

**Cicatriz encerrada:** Agora o scaler aprende apenas no treino e aplica no teste, sem vazamento.

---

### 🔟 Ajuste de threshold

*Nem sempre a probabilidade 0.5 é o melhor ponto de corte para classes desbalanceadas.*

```python
threshold = 0.3
y_pred_custom = (y_probs > threshold).astype(int)
print(classification_report(y_test, y_pred_custom))
```

---

### 1️⃣1️⃣ Busca de hiperparâmetros com GridSearch

*Testamos várias combinações para melhorar o modelo, com foco em Recall.*

```python
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

param_grid = {
    "max_depth": [3, 5],
    "n_estimators": [50, 100]
}
grid = GridSearchCV(
    XGBClassifier(eval_metric="logloss"),   # corrigido de "CGBClassifier"
    param_grid,
    scoring="recall",
    cv=3
)
grid.fit(X_train, y_train)
print("Melhor modelo:", grid.best_params_)
```

---

### 1️⃣2️⃣ Explicabilidade com SHAP

*Entender quais variáveis mais influenciam as decisões do modelo.*

```python
import shap

explainer = shap.Explainer(xgb)
shap_values = explainer(X_test[:100])
shap.plots.bar(shap_values)
```

O gráfico de barras exibe a importância média das features – para fraudes, geralmente `V14`, `V4`, `V12` (as variáveis anonimizadas) lideram.

---

## 📘 Miniguia de Estudo

### ✅ Resumos Estruturados

1. **O problema da detecção de fraudes**  
   - Classe minoritária representa <1% dos casos.  
   - Acurácia não é métrica suficiente; é preciso otimizar Recall e F1‑Score.

2. **Pré‑processamento e Feature Engineering**  
   - Transformações logarítmicas reduzem assimetria (ex.: `Amount_log`).  
   - `StandardScaler` coloca features na mesma escala.  
   - **Sempre aplicar scaling após split** (use Pipelines).

3. **Métricas para dados desbalanceados**  
   - **Recall (Sensibilidade):** quantas fraudes reais foram detectadas.  
   - **Precision:** quantas das transações classificadas como fraude eram realmente fraude.  
   - **F1‑Score:** média harmônica entre Precision e Recall.  
   - **ROC AUC:** capacidade geral de discriminação do modelo.

4. **Técnicas de balanceamento**  
   - **Undersampling:** remove exemplos da classe majoritária (perde dados).  
   - **SMOTE (Oversampling):** gera exemplos sintéticos da classe minoritária (cuidado com overfitting).  
   - **Pesos de classe (`class_weight` / `scale_pos_weight`):** método embutido que penaliza mais os erros na classe minoritária.

5. **Modelos utilizados**  
   - Regressão Logística (simples, interpretável).  
   - Random Forest (robusto, captura não‑linearidades).  
   - XGBoost (gradiente boosting, estado da arte para tabelas).  

6. **Pipelines e boas práticas**  
   - `Pipeline` do sklearn evita vazamento de dados e organiza o código.  
   - Sempre separar treino/teste com `stratify`.  
   - Ajustar o threshold de decisão para melhorar recall.  

7. **Explicabilidade**  
   - SHAP mostra o impacto de cada variável nas previsões, essencial em modelos críticos.

### 📖 Glossário

- **Classe minoritária:** a classe rara que queremos prever (fraude = 1).  
- **Recall:** TP / (TP + FN) – proporção de fraudes encontradas.  
- **Precision:** TP / (TP + FP) – confiabilidade das detecções.  
- **F1‑Score:** 2 * (Precision * Recall) / (Precision + Recall).  
- **ROC AUC:** Área sob a curva ROC; quanto mais próximo de 1, melhor.  
- **SMOTE:** Synthetic Minority Over‑sampling Technique – cria exemplos sintéticos.  
- **Undersampling:** reduz aleatoriamente a classe majoritária.  
- **Class weight / Scale pos weight:** penalização diferenciada durante o treino.  
- **Pipeline:** sequência de transformações e um modelo aplicados em ordem.  
- **Threshold:** ponto de corte da probabilidade para decidir a classe.  
- **SHAP:** SHapley Additive exPlanations – método para interpretar modelos caixa‑preta.

### 🚀 Prompts Reutilizáveis (Códigos para futuros projetos)

1. **Carregar e verificar desbalanceamento**
   ```python
   df = pd.read_csv("seus_dados.csv")
   print(df["target"].value_counts(normalize=True))
   ```

2. **Aplicar log + scaling (dentro do pipeline)**
   ```python
   from sklearn.pipeline import Pipeline
   from sklearn.preprocessing import StandardScaler, FunctionTransformer

   pipeline = Pipeline([
       ("log", FunctionTransformer(np.log1p)),
       ("scaler", StandardScaler())
   ])
   ```

3. **Dividir mantendo proporção**
   ```python
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, stratify=y, test_size=0.3, random_state=42
   )
   ```

4. **Treinar com XGBoost e avaliar**
   ```python
   model = XGBClassifier(scale_pos_weight=10, eval_metric="logloss")
   model.fit(X_train, y_train)
   print(classification_report(y_test, model.predict(X_test)))
   ```

5. **Curva ROC**
   ```python
   probs = model.predict_proba(X_test)[:,1]
   fpr, tpr, _ = roc_curve(y_test, probs)
   plt.plot(fpr, tpr); plt.title("ROC"); plt.show()
   print("AUC:", roc_auc_score(y_test, probs))
   ```

6. **GridSearch com foco em Recall**
   ```python
   grid = GridSearchCV(estimator, param_grid, scoring="recall", cv=3)
   grid.fit(X_train, y_train)
   ```

7. **Explicabilidade com SHAP**
   ```python
   explainer = shap.Explainer(model)
   shap_values = explainer(X_test[:100])
   shap.plots.bar(shap_values)
   ```
