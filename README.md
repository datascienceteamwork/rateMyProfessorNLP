# RateMyProfessors — Sentiment Analysis Pipeline

Progetto di Data Science per il corso di **Ingegneria Informatica e dell'Automazione**  
Università Politecnica delle Marche — A.A. 2025/2026

> **Docente:** Prof. Domenico Ursino  
> **Gruppo:** Lorenzo Meloccaro · Yassir Flavio Suarez Sanchez · Domenico La Porta

---

## Descrizione

Pipeline completa di **sentiment analysis** su recensioni di professori universitari tratte dalla piattaforma [RateMyProfessors](https://www.ratemyprofessors.com/), disponibile su HuggingFace Datasets Hub.

Il progetto confronta due approcci:

- **Classificazione Binaria** — Positivo vs Negativo (etichette originali del dataset)
- **Classificazione Multi-classe** — Positivo / Neutro / Negativo (classe neutra generata automaticamente tramite TextBlob polarity)

Per ciascun approccio vengono addestrati, ottimizzati e confrontati sei modelli base più tre ensemble (Voting Hard, Voting Soft, Stacking).

---

## Struttura del Progetto

```
.
├── prova_avanzata.py          # Script principale della pipeline
├── requirements.txt           # Dipendenze Python
├── README.md                  # Questo file
│
├── outputs/
│   ├── dataset_raw.csv        # Dataset scaricato e cachato (generato al primo run)
│   ├── run_log_TIMESTAMP.txt  # Log completo di ogni esecuzione
│   ├── plots_2class/          # Grafici classificazione binaria
│   │   ├── plot_01_class_distribution.png
│   │   ├── plot_02_confusion_matrix.png
│   │   ├── plot_03_roc_curves.png
│   │   ├── plot_04_precision_recall_curves.png
│   │   ├── plot_05_top_tfidf_features.png
│   │   ├── plot_06_cv_scores.png
│   │   └── plot_07_learning_curves.png
│   └── plots_3class/          # Grafici classificazione multi-classe
│       └── (stessa struttura di plots_2class)
│
└── models/                    # Artefatti salvati (generati dopo il training)
    ├── vectorizer_2class.joblib
    ├── model_2class.joblib
    ├── vectorizer_3class.joblib
    ├── model_3class.joblib
    └── thresholds_3class.joblib
```

---

## Installazione

### 1. Clona la repository

```bash
git clone https://github.com/<tuo-username>/ratemyprofessors-sentiment.git
cd ratemyprofessors-sentiment
```

### 2. Crea un ambiente virtuale (consigliato)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3. Installa le dipendenze

```bash
pip install -r requirements.txt
```

---

## Utilizzo

### Esecuzione completa della pipeline

```bash
python prova_avanzata.py
```

Al primo avvio lo script:
1. Scarica automaticamente il dataset da HuggingFace e lo salva in `outputs/dataset_raw.csv`
2. Esegue preprocessing, feature extraction, training e valutazione per entrambe le classificazioni
3. Genera tutti i grafici nelle rispettive cartelle `outputs/plots_*`
4. Salva i modelli addestrati in `models/`
5. Salva il log completo in `outputs/run_log_YYYYMMDD_HHMMSS.txt`

Dalle esecuzioni successive il dataset viene caricato dal CSV locale, saltando il download.

---

## Pipeline

```
Raw Text
   │
   ▼
Preprocessing
  • Lowercasing
  • Rimozione URL, punteggiatura, numeri
  • Tokenizzazione (NLTK)
  • Rimozione stop words
  • Negation handling  (NOT_<token>)
  • Lemmatizzazione (WordNet)
  • Filtering token corti (< 3 char)
   │
   ▼
Feature Extraction
  • TF-IDF Vectorizer  (5000 features, ngram (1,2))
  • Feature custom     (lunghezza, esclamativi, uppercase ratio, ...)
   │
   ▼
Bilanciamento Classi
  • Confronto automatico: None vs Random Oversampling
  • Selezione della strategia migliore via probe LR
   │
   ▼
Training & Tuning  (RandomizedSearchCV, 3-Fold CV)
  • Logistic Regression
  • Linear SVM (CalibratedClassifierCV)
  • Naive Bayes (MultinomialNB)
  • Random Forest
  • XGBoost
  • LightGBM
   │
   ▼
Ensemble
  • Voting Hard
  • Voting Soft
  • Stacking  (meta-learner: Logistic Regression)
   │
   ▼
Valutazione  (5-Fold Stratified CV + Test Set)
  • Accuracy, Precision, Recall, F1-score
  • AUC-ROC, Average Precision
  • Confusion Matrix
  • Learning Curves
```

---

## Modelli e Risultati

### Classificazione Binaria (2 classi)

| Modello | CV F1 (weighted) |
|---|---|
| Linear SVM | 0.9321 ± 0.0027 |
| Logistic Regression | 0.9307 ± 0.0026 |
| LightGBM | 0.9254 ± 0.0022 |
| Naive Bayes | 0.9202 ± 0.0034 |
| XGBoost | 0.9175 ± 0.0013 |
| Random Forest | 0.9038 ± 0.0015 |

**Best model (test set):** Stacking — F1 = 0.935, AUC = 0.978

### Classificazione Multi-classe (3 classi)

| Modello | CV F1 (weighted) |
|---|---|
| Logistic Regression | 0.8015 ± 0.0017 |
| LightGBM | 0.7667 ± 0.0045 |
| Linear SVM | 0.7643 ± 0.0038 |
| XGBoost | 0.7634 ± 0.0025 |
| Random Forest | 0.7088 ± 0.0042 |
| Naive Bayes | 0.6812 ± 0.0080 |

**Best model (test set):** Stacking — Negative 85.6%, Neutral 78.6%, Positive 81.0%

---

## Configurazione

Tutti i parametri principali sono centralizzati nella classe `PipelineConfig` all'interno dello script:

| Parametro | Valore default | Descrizione |
|---|---|---|
| `TEST_SIZE` | 0.20 | Proporzione test set |
| `CV_FOLDS` | 5 | Fold per cross-validation finale |
| `N_ITER_SEARCH` | 10 | Iterazioni RandomizedSearchCV |
| `CV_TUNING_FOLDS` | 3 | Fold per tuning iperparametri |
| `TFIDF max_features` | 5000 | Feature massime TF-IDF |
| `TFIDF ngram_range` | (1, 2) | Unigrammi e bigrammi |
| `NEUTRAL_QUANTILE_LOW` | 0.30 | Soglia inferiore classe neutra |
| `NEUTRAL_QUANTILE_HIGH` | 0.70 | Soglia superiore classe neutra |
| `RANDOM_STATE` | 42 | Seed per riproducibilità |

---

## Dataset

- **Nome:** `ZephyrUtopia/ratemyprofessors-reviews-2-labels`
- **Fonte:** [HuggingFace Datasets Hub](https://huggingface.co/datasets/ZephyrUtopia/ratemyprofessors-reviews-2-labels)
- **Campioni totali:** 39.574
- **Lingua:** Inglese
- **Distribuzione originale:** 73.8% Positive — 26.2% Negative
- **Task:** Sentiment Analysis su recensioni di docenti universitari

---

## Requisiti di Sistema

- Python ≥ 3.10
- RAM consigliata: ≥ 8 GB (per XGBoost/LightGBM su matrice TF-IDF)
- Connessione internet richiesta solo al primo avvio (download dataset)

---

## Licenza

Progetto accademico — Università Politecnica delle Marche, A.A. 2025/2026.
