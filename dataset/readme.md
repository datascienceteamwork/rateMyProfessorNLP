# Dataset — RateMyProfessors Reviews

## Panoramica

Il dataset contiene recensioni testuali di professori universitari raccolte dalla
piattaforma [RateMyProfessors.com](https://www.ratemyprofessors.com/) e reso
disponibile su HuggingFace Datasets Hub.

| Proprietà | Valore |
|---|---|
| **Nome HuggingFace** | `ZephyrUtopia/ratemyprofessors-reviews-2-labels` |
| **File locale** | `rateMyProffesor_HuggingFace_dataset.csv` |
| **Campioni totali** | 39.574 |
| **Colonne** | 2 (`text`, `label`) |
| **Lingua** | Inglese |
| **Task** | Sentiment Analysis (classificazione binaria) |

---

## Struttura

Il dataset è composto da due sole colonne:

| Colonna | Tipo | Descrizione |
|---|---|---|
| `text` | `string` | Testo della recensione scritta dallo studente |
| `label` | `int64` | Etichetta di sentiment: `0` = Negativo, `1` = Positivo |

### Esempio di record

```
text:  "Professor Agresta taught our ITAL101 class this semester and she was
        awesome! The class was super easy. No quizzes/tests, just a weekly
        homework and lecture assignment. She gives good feedback and lets you
        turn in things late with no penalty. Great professor!"
label: 1
```

```
text:  "Shes nice but her assignments aren't clear and are a lot of work.
        She also doesn't grade assignments until like 1-2 months later."
label: 0
```

---

## Distribuzione delle Classi

| Classe | Label | Conteggio | Percentuale |
|---|---|---|---|
| Positivo | 1 | 29.180 | 73.7% |
| Negativo | 0 | 10.394 | 26.3% |
| **Totale** | — | **39.574** | **100%** |

**Imbalance ratio:** 2.81 : 1 (positivi vs negativi)

Il dataset è significativamente sbilanciato verso la classe positiva, riflettendo
il comportamento reale degli utenti della piattaforma, che tendono a scrivere
recensioni positive più frequentemente di quelle negative.

---

## Statistiche Testuali

### Lunghezza in caratteri

| Statistica | Totale | Negativi | Positivi |
|---|---|---|---|
| Media | 258.1 | 267.1 | 254.9 |
| Mediana | 300 | 321 | 293 |
| Std | 99.6 | 100.6 | 99.0 |
| Min | 1 | 1 | 1 |
| Max | 350 | 350 | 350 |
| 25° percentile | 191 | 206 | 187 |
| 75° percentile | 343 | 345 | 342 |

### Numero di parole

| Statistica | Totale | Negativi | Positivi |
|---|---|---|---|
| Media | 46.6 | 48.3 | 46.0 |
| Mediana | 53 | 56 | 52 |
| Std | 18.4 | 18.6 | 18.2 |
| Min | 0 | 0 | 0 |
| Max | 81 | 81 | 77 |

Le recensioni negative risultano mediamente leggermente più lunghe (+12 caratteri,
+2.3 parole) rispetto alle positive. Questo può riflettere la tendenza degli
studenti a motivare più estesamente un giudizio negativo.

---

## Qualità dei Dati

| Problema | Quantità | Note |
|---|---|---|
| Valori nulli (`NaN`) | 3 | Da rimuovere in preprocessing |
| Testi vuoti (stringa vuota) | 19 | Da rimuovere in preprocessing |
| Testi troncati a 350 caratteri | 2.458 (6.2%) | Limite della piattaforma di origine |

> **Nota sul troncamento:** 2.458 recensioni (6.2% del dataset) raggiungono
> esattamente il limite massimo di 350 caratteri, suggerendo che il testo
> originale fosse più lungo e sia stato tagliato dalla piattaforma o dal processo
> di raccolta dati. Questo introduce una potenziale distorsione nelle recensioni
> più elaborate.

---

## Utilizzo nel Progetto

Il dataset viene impiegato in due esperimenti distinti:

### Esperimento 1 — Classificazione Binaria
Le etichette originali (`0` / `1`) vengono utilizzate direttamente per distinguere
recensioni negative da positive.

### Esperimento 2 — Classificazione Multi-classe
Le etichette originali vengono ignorate. Viene calcolata la **polarity score** di
TextBlob per ciascuna recensione e le tre classi vengono generate automaticamente
tramite quantili:

| Classe | Label | Condizione |
|---|---|---|
| Negativo | 0 | polarity < quantile 30° |
| Neutro | 1 | quantile 30° ≤ polarity ≤ quantile 70° |
| Positivo | 2 | polarity > quantile 70° |

Questo produce una distribuzione approssimativamente bilanciata:
29.6% Negativi — 40.5% Neutrali — 30.0% Positivi.

---

## Come Caricare il Dataset

### Da HuggingFace (richiede connessione internet)

```python
from datasets import load_dataset

ds = load_dataset("ZephyrUtopia/ratemyprofessors-reviews-2-labels", split="train")
df = ds.to_pandas()
```

### Da file CSV locale

```python
import pandas as pd

df = pd.read_csv("rateMyProffesor_HuggingFace_dataset.csv")
print(df.shape)      # (39574, 2)
print(df.dtypes)
# text     object
# label     int64
```

### Statistiche rapide

```python
print(df['label'].value_counts())
# 1    29180
# 0    10394

print(df['text'].str.len().describe())
print(df.isnull().sum())
```

---

## Riferimento

```bibtex
@dataset{ratemyprofessors2labels,
  title   = {RateMyProfessors Reviews — 2 Labels},
  author  = {ZephyrUtopia},
  year    = {2024},
  url     = {https://huggingface.co/datasets/ZephyrUtopia/ratemyprofessors-reviews-2-labels},
  note    = {HuggingFace Datasets Hub}
}
```
