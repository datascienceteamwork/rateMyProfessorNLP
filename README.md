# RateMyProfessors — Sentiment Analysis Pipeline

> **Gruppo:** Lorenzo Meloccaro · Yassir Flavio Suarez Sanchez · Domenico La Porta



## Descrizione

Pipeline completa di **sentiment analysis** su recensioni di professori universitari tratte dalla piattaforma [RateMyProfessors](https://www.ratemyprofessors.com/), disponibile su HuggingFace Datasets Hub.

Il progetto confronta due approcci:

- **Classificazione Binaria** — Positivo vs Negativo (etichette originali del dataset)
- **Classificazione Multi-classe** — Positivo / Neutro / Negativo (classe neutra generata automaticamente tramite TextBlob polarity)

Per ciascun approccio vengono addestrati, ottimizzati e confrontati sei modelli base più tre ensemble (Voting Hard, Voting Soft, Stacking).

Il progetto include inoltre un task di **Information Extraction**, che affianca alla classificazione due tecniche complementari:

- **Relation Extraction** — estrazione di triple (aspetto, descrittore, sentiment) tramite dependency parsing (spaCy)
- **Keyword/Keyphrase Extraction** — estrazione di espressioni ricorrenti per classe di sentiment tramite RAKE, con filtro di frequenza documentale


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
python -m spacy download en_core_web_sm
```

> Il modello `en_core_web_sm` di spaCy è richiesto solo per lo script di Information Extraction (`information_extraction_v6.py`).

---

## Utilizzo

### Esecuzione della pipeline di sentiment analysis

```bash
python gradeit.py
```

Al primo avvio lo script:
1. Scarica automaticamente il dataset da HuggingFace e lo salva in locale
2. Esegue preprocessing, feature extraction, training e valutazione per entrambe le classificazioni
3. Genera tutti i grafici nelle rispettive cartelle `outputs/plots_*`
4. Salva i modelli addestrati in `models/`
5. Salva il log completo in `outputs/run_log_YYYYMMDD_HHMMSS.txt`

Dalle esecuzioni successive il dataset viene caricato dal CSV locale, saltando il download.

### Esecuzione dell'Information Extraction

```bash
python information_extraction_v6.py
```

Riusa la cache del dataset già generata da `gradeit.py` (`outputs/rateMyProffesor_HuggingFace_dataset.csv`). Parametri opzionali:

```bash
python information_extraction_v6.py --n_reviews 4000 --top_k 15 --top_k_relations 15
```

Output generati in `outputs/information_extraction/`:

| File | Contenuto |
|---|---|
| `aspect_sentiment_matrix.csv` / `aspect_sentiment_heatmap.png` | matrice aspetto × sentiment |
| `aspect_relations.csv` / `aspect_relations_barplot.png` | top relazioni (aspetto, descrittore) |
| `keywords_per_class.csv` / `keywords_barplot.png` | top keyphrase per classe (RAKE) |
| `ie_report.txt` | report testuale riassuntivo |

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

### Pipeline di Information Extraction (`information_extraction_v6.py`)

```
Raw Text (cache condivisa con gradeit.py)
   │
   ▼
Relation Extraction (spaCy dependency parsing)
  • Match aspetto (dizionario di trigger: teaching, exams, workload, ...)
  • Estrazione descrittore (amod / nsubj+acomp / conj)
  • Sentiment di frase (TextBlob polarity)
   │
   ▼
Keyword Extraction (RAKE)
  • Filtro lingua (langdetect, solo recensioni in inglese)
  • Estrazione candidate su pool ampio
  • Ordinamento per frequenza documentale, poi per score RAKE
  • Filtro doc_freq ≥ 3 (con fallback automatico)
   │
   ▼
Output: matrice aspetto×sentiment, top relazioni, top keyphrase, report
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

### Information Extraction

- **Relation Extraction:** 7 aspetti monitorati (teaching, exams, workload, clarity, grading, attitude, attendance). Aspetto più menzionato: *teaching* (657 relazioni positive su un campione di 4.000 recensioni).
- **Keyword Extraction:** keyphrase ricorrenti per classe dopo filtro di frequenza documentale, es. *"gives pop quizzes"* (Negative, 4 recensioni), *"would definitely recommend taking"* (Positive, 5 recensioni).

---

## Configurazione

Tutti i parametri principali della pipeline di sentiment analysis sono centralizzati nella classe `PipelineConfig` all'interno di `gradeit.py`:

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

I parametri della pipeline di Information Extraction sono centralizzati nella classe `Config` all'interno di `information_extraction_v6.py`:

| Parametro | Valore default | Descrizione |
|---|---|---|
| `N_REVIEWS` | 4000 | Subset di recensioni analizzate |
| `TOP_K_KEYWORDS` | 15 | Keyphrase estratte per classe |
| `TOP_K_RELATIONS` | 15 | Relazioni (aspetto, descrittore) estratte |
| `MIN_DOC_FREQUENCY` | 3 | Frequenza documentale minima per una keyphrase |
| `KEYWORD_CANDIDATE_POOL` | 800 | Pool di candidate RAKE su cui calcolare la doc_freq |
| `SPACY_MODEL` | en_core_web_sm | Modello spaCy per il dependency parsing |

---

## Dataset

- **Nome:** `ZephyrUtopia/ratemyprofessors-reviews-2-labels`
- **Fonte:** [HuggingFace Datasets Hub](https://huggingface.co/datasets/ZephyrUtopia/ratemyprofessors-reviews-2-labels)
- **Campioni totali:** 39.574
- **Lingua:** prevalentemente inglese (minoranza di recensioni in francese/italiano, escluse nel task di Information Extraction)
- **Distribuzione originale:** 73.8% Positive — 26.2% Negative
- **Task:** Sentiment Analysis e Information Extraction su recensioni di docenti universitari

---

## Requisiti di Sistema

- Python ≥ 3.10
- RAM consigliata: ≥ 8 GB (per XGBoost/LightGBM su matrice TF-IDF)
- Connessione internet richiesta solo al primo avvio (download dataset e modello spaCy)

---

## Licenza

Progetto accademico — Università Politecnica delle Marche, A.A. 2025/2026.
