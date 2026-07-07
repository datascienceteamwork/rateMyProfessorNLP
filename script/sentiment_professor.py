"""
information_extraction_v6.py
=============================
Modulo di Information Extraction per il progetto di sentiment analysis
sulle recensioni RateMyProfessor.

Rispetto alla v5, corregge il vero problema del filtro anti-rumore:

  FIX 7 — Il filtro doc_freq guardava nel posto sbagliato:
    in v4/v5 si prendevano solo le keyphrase con RAKE score piu' alto,
    e SOLO DOPO si controllava quante volte ricorrevano. Ma tra le
    frasi con score alto (spesso rare e "dense" di parole insolite)
    quasi nessuna ricorre identica in piu' recensioni: il fallback
    finiva quindi sempre per abbassare la soglia fino a 1, vanificando
    il filtro. Ora si allarga MOLTO il pool di candidate RAKE, si
    calcola la doc_freq per tutte, e si ordina PRIMA per frequenza e
    POI per score — cosi' le frasi davvero ricorrenti nel corpus
    emergono per prime, invece di essere sommerse da migliaia di frasi
    rare con score numericamente piu' alto.

Combina QUATTRO componenti di IE:

  1) RELATION EXTRACTION (spaCy dependency parsing)
     Estrae triple (aspetto, descrittore, sentiment) analizzando le
     relazioni sintattiche della frase (non solo co-occorrenza di parole).

  2) AGGREGAZIONE aspetto -> sentiment
     Per confronto e per generare la heatmap.

  3) TOP RELAZIONI aspetto-descrittore
     Le coppie (aspetto, descrittore) piu' frequenti, con relativo
     sentiment prevalente.

  4) KEYWORD / KEYPHRASE EXTRACTION (RAKE, solo su recensioni in inglese,
     ordinata per frequenza nel corpus, non solo per score RAKE)
     Estrae le keyphrase piu' rilevanti per ciascuna classe di sentiment,
     in modo data-driven, privilegiando i pattern ricorrenti.

Output (in outputs/information_extraction/):
  - aspect_sentiment_matrix.csv        matrice aspetto x sentiment
  - aspect_sentiment_heatmap.png       heatmap della matrice
  - aspect_relations.csv               top relazioni (aspetto, descrittore)
  - aspect_relations_barplot.png       barplot delle top relazioni
  - keywords_per_class.csv             top keyphrase per classe
  - keywords_barplot.png               barplot delle keyphrase
  - ie_report.txt                     report testuale pronto per la relazione

Uso:
    python information_extraction_v6.py
    python information_extraction_v6.py --n_reviews 3000 --top_k 15
"""

import os
import argparse
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from textblob import TextBlob

import spacy
from rake_nltk import Rake
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 42  # risultati deterministici

from datasets import load_dataset


# ============================================================================
# CONFIG
# ============================================================================

class Config:
    DATASET_NAME = "ZephyrUtopia/ratemyprofessors-reviews-2-labels"
    CACHE_CSV    = os.path.join("outputs", "rateMyProffesor_HuggingFace_dataset.csv")
    OUT_DIR      = os.path.join("outputs", "information_extraction")
    SPACY_MODEL  = "en_core_web_sm"
    N_REVIEWS    = 4000     # subset per velocita' 
    TOP_K_KEYWORDS  = 15
    TOP_K_RELATIONS = 15
    MIN_DOC_FREQUENCY = 3   # FIX 5 — una keyphrase deve comparire in almeno N recensioni distinte
    KEYWORD_CANDIDATE_POOL = 800  # FIX 7 — quante candidate RAKE considerare per il calcolo doc_freq
    RANDOM_SEED  = 42

    # Dizionario aspetti: aspetto -> lemmi/parole trigger che identificano
    ASPECTS = {
        "teaching":   ["teach", "lecture", "explain", "explanation"],
        "exams":      ["exam", "test", "quiz", "midterm", "final"],
        "workload":   ["homework", "assignment", "workload", "reading", "project"],
        "clarity":    ["clarity", "organization", "structure"],
        "grading":    ["grade", "grading", "score", "curve"],
        "attitude":   ["nice", "rude", "helpful", "caring", "mean", "funny"],
        "attendance": ["attendance", "absence", "mandatory"],
    }


def banner(title, w=78):
    print("\n" + "=" * w)
    print(title.center(w))
    print("=" * w)


def ensure_setup():
    os.makedirs(Config.OUT_DIR, exist_ok=True)
    for pkg in ["punkt", "punkt_tab", "stopwords"]:
        nltk.download(pkg, quiet=True)
    try:
        return spacy.load(Config.SPACY_MODEL)
    except OSError:
        print(f"[SETUP] Modello spaCy '{Config.SPACY_MODEL}' non trovato, download...")
        from spacy.cli import download
        download(Config.SPACY_MODEL)
        return spacy.load(Config.SPACY_MODEL)


# ============================================================================
# DATA LOADING 
# ============================================================================

def load_data(n_reviews: int) -> pd.DataFrame:
    if os.path.exists(Config.CACHE_CSV):
        print(f"[DATA] Dataset trovato in cache: {Config.CACHE_CSV}")
        df = pd.read_csv(Config.CACHE_CSV)
    else:
        print(f"[DATA] Download dataset da HuggingFace: {Config.DATASET_NAME}")
        dataset = load_dataset(Config.DATASET_NAME, split="train")
        df = pd.DataFrame(dataset)
        os.makedirs("outputs", exist_ok=True)
        df.to_csv(Config.CACHE_CSV, index=False)

    df = df.dropna(subset=["text"]).reset_index(drop=True)
    df = df[df["text"].astype(str).str.strip().str.len() > 0].reset_index(drop=True)

    df = df.sample(n=min(n_reviews, len(df)), random_state=Config.RANDOM_SEED).reset_index(drop=True)
    print(f"[DATA] Campioni usati: {len(df):,} (subset per velocita')")
    return df


# ============================================================================
# 1) RELATION EXTRACTION (spaCy dependency parsing)
# ============================================================================

def _match_aspect(token) -> str:
    """Ritorna il nome dell'aspetto se il token (lemma, testo o una loro
    forma affine) e' un trigger. Il controllo substring gestisce i casi
    in cui il lemmatizer non riduce la forma nominale al trigger esatto
    (es. 'teaching' come sostantivo resta 'teaching', non 'teach')."""
    lemma = token.lemma_.lower()
    text = token.text.lower()
    for aspect, triggers in Config.ASPECTS.items():
        if lemma in triggers or text in triggers:
            return aspect
        if any(t in text for t in triggers if len(t) > 3):
            return aspect
    return None


def extract_aspect_relations(df: pd.DataFrame, nlp) -> pd.DataFrame:
    """
    Per ogni recensione, analizza l'albero delle dipendenze e cerca,
    per ciascun token "aspetto" (es. 'exams'), i descrittori collegati
    sintatticamente:

      - amod diretto:      "hard exams"            -> hard --amod--> exams
      - predicato+copula:  "exams were hard"       -> exams --nsubj--> were <--acomp-- hard
      - congiunzioni:      "hard and confusing"    -> espande anche i conj dell'amod/acomp

    Ritorna un dataframe lungo con colonne: aspect, descriptor, sentiment.
    """
    rows = []
    docs = nlp.pipe(df["text"].astype(str).tolist(), batch_size=64, disable=["ner"])

    for doc in docs:
        for token in doc:
            aspect_name = _match_aspect(token)
            if aspect_name is None:
                continue

            descriptor_tokens = []

            for child in token.children:
                if child.dep_ == "amod":
                    descriptor_tokens.append(child)
            head = token.head
            if token.dep_ in ("nsubj", "nsubjpass") and head.pos_ in ("VERB", "AUX"):
                for sib in head.children:
                    if sib.dep_ in ("acomp", "attr", "oprd"):
                        descriptor_tokens.append(sib)

            if not descriptor_tokens:
                continue

            # Espande le congiunzioni: "hard and confusing" -> aggiunge 'confusing'
            expanded = list(descriptor_tokens)
            for d in descriptor_tokens:
                expanded.extend([c for c in d.children if c.dep_ == "conj"])

            sent_text = token.sent.text.strip()
            polarity = TextBlob(sent_text).sentiment.polarity
            if polarity > 0.1:
                sentiment = "Positive"
            elif polarity < -0.1:
                sentiment = "Negative"
            else:
                sentiment = "Neutral"

            for d in expanded:
                rows.append({
                    "aspect": aspect_name,
                    "descriptor": d.lemma_.lower(),
                    "sentiment": sentiment,
                })

    result = pd.DataFrame(rows)
    print(f"[RELATIONS] Relazioni (aspetto, descrittore) estratte: {len(result):,}")
    return result


def build_aspect_matrix(relation_df: pd.DataFrame) -> pd.DataFrame:
    if relation_df.empty:
        print("[RELATIONS][WARN] Nessuna relazione individuata: controlla il dizionario "
              "ASPECTS o il testo in input.")
        return pd.DataFrame(0, index=list(Config.ASPECTS.keys()),
                             columns=["Negative", "Neutral", "Positive"])

    matrix = pd.crosstab(relation_df["aspect"], relation_df["sentiment"])
    for col in ["Negative", "Neutral", "Positive"]:
        if col not in matrix.columns:
            matrix[col] = 0
    matrix = matrix[["Negative", "Neutral", "Positive"]]
    return matrix


def build_relations_table(relation_df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    """Conta le coppie (aspetto, descrittore) piu' frequenti con il loro
    sentiment prevalente, pronte per relazione/report."""
    if relation_df.empty:
        return pd.DataFrame(columns=["aspect", "descriptor", "count", "prevalent_sentiment"])

    grouped = (
        relation_df.groupby(["aspect", "descriptor"])
        .agg(count=("sentiment", "size"),
             prevalent_sentiment=("sentiment", lambda s: s.mode().iat[0]))
        .reset_index()
        .sort_values("count", ascending=False)
        .head(top_k)
    )
    return grouped


def plot_aspect_heatmap(matrix: pd.DataFrame, out_path: str):
    plt.figure(figsize=(8, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="RdYlGn",
                cbar_kws={"label": "N. relazioni"})
    plt.title("Relation Extraction — Aspetto vs Sentiment\n(dependency parsing, spaCy)")
    plt.ylabel("Aspetto")
    plt.xlabel("Sentiment")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[RELATIONS] Heatmap salvata in: {out_path}")


def plot_relations(relations_df: pd.DataFrame, out_path: str):
    if relations_df.empty:
        print("[RELATIONS][WARN] Nessuna relazione da plottare.")
        return

    palette = {"Negative": "#e74c3c", "Neutral": "#f39c12", "Positive": "#2ecc71"}
    labels = [f"{r.aspect} — {r.descriptor}" for r in relations_df.itertuples()]
    colors = [palette.get(r.prevalent_sentiment, "#3498db") for r in relations_df.itertuples()]

    plt.figure(figsize=(9, max(4, 0.4 * len(labels))))
    plt.barh(labels[::-1], relations_df["count"].tolist()[::-1], color=colors[::-1])
    plt.title("Top relazioni (aspetto, descrittore) estratte via dependency parsing")
    plt.xlabel("N. occorrenze")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[RELATIONS] Barplot relazioni salvato in: {out_path}")


def _is_english(text: str) -> bool:
    try:
        return detect(text) == "en"
    except Exception:
        return False


def filter_english_reviews(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """FIX 3 — il dataset contiene anche recensioni in francese/italiano;
    RAKE con stopword inglesi le tratta come rumore. Le esclude prima
    dell'estrazione delle keyphrase."""
    mask = df[text_col].astype(str).apply(_is_english)
    filtered = df[mask].reset_index(drop=True)
    removed = len(df) - len(filtered)
    print(f"[LANG] Recensioni non in inglese escluse: {removed:,} "
          f"(rimaste: {len(filtered):,})")
    return filtered


# ============================================================================
# 2) KEYWORD / KEYPHRASE EXTRACTION 
# ============================================================================

def extract_keywords_per_class(df: pd.DataFrame, label_col: str,
                                label_names: dict, top_k: int,
                                min_doc_freq: int = Config.MIN_DOC_FREQUENCY,
                                candidate_pool: int = Config.KEYWORD_CANDIDATE_POOL) -> pd.DataFrame:
    """
    Per ciascuna classe (Negative/Positive), unisce tutte le recensioni
    e applica RAKE per estrarre le keyphrase.

    FIX 7 — a differenza delle versioni precedenti, NON si prendono solo
    le top-k per score RAKE per poi controllarne la frequenza (le frasi
    con score piu' alto sono quasi sempre rare e non ricorrono mai
    davvero). Invece:
      1) si allarga il pool di candidate a `candidate_pool` frasi
      2) si calcola la doc_freq (n. recensioni distinte in cui compare)
         per TUTTE le candidate del pool
      3) si ordina prima per doc_freq (decrescente) e poi per score,
         cosi' emergono i pattern davvero ricorrenti nel corpus
      4) si tengono solo le frasi con doc_freq >= min_doc_freq; se
         nessuna la supera (corpus piccolo/eterogeneo), si ripiega sulle
         frasi con la doc_freq piu' alta disponibile, avvisando nel log.
    """
    columns = ["class", "keyphrase", "score", "doc_freq"]
    rows = []
    r = Rake()

    for label_val, label_name in label_names.items():
        subset = df[df[label_col] == label_val]
        texts_lower = subset["text"].astype(str).str.lower().tolist()
        text_blob = " . ".join(texts_lower)

        if not text_blob.strip():
            print(f"[KEYWORDS][WARN] Nessun testo per la classe '{label_name}', salto.")
            continue

        r.extract_keywords_from_text(text_blob)
        ranked = r.get_ranked_phrases_with_scores()

        
        best_score = {}
        for score, phrase in ranked:
            if len(phrase.split()) > 6:
                continue
            if phrase not in best_score or score > best_score[phrase]:
                best_score[phrase] = score
        candidates = sorted(best_score.items(), key=lambda kv: kv[1], reverse=True)[:candidate_pool]

        # calcola la doc_freq per tutte le candidate del pool
        scored = []
        for phrase, score in candidates:
            doc_freq = sum(1 for t in texts_lower if phrase in t)
            scored.append({"class": label_name, "keyphrase": phrase,
                           "score": score, "doc_freq": doc_freq})

        # ordina per doc_freq decrescente, poi per score decrescente
        scored.sort(key=lambda d: (d["doc_freq"], d["score"]), reverse=True)

        class_rows = [d for d in scored if d["doc_freq"] >= min_doc_freq][:top_k]
        if not class_rows:
            max_freq = max((d["doc_freq"] for d in scored), default=0)
            print(f"[KEYWORDS][WARN] Classe '{label_name}': nessuna keyphrase con "
                  f"doc_freq>={min_doc_freq}; ripiego sulle piu' frequenti disponibili "
                  f"(max doc_freq={max_freq}).")
            class_rows = scored[:top_k]

        rows.extend(class_rows)

    result = pd.DataFrame(rows, columns=columns)
    print(f"[KEYWORDS] Keyphrase estratte (soglia doc_freq>={min_doc_freq}, "
          f"pool candidate={candidate_pool}): {len(result):,}")
    return result


def plot_keywords(keywords_df: pd.DataFrame, out_path: str, top_k: int):
    if keywords_df.empty or "class" not in keywords_df.columns:
        print("[KEYWORDS][WARN] Nessuna keyphrase da plottare, salto il barplot.")
        return

    classes = keywords_df["class"].unique()
    fig, axes = plt.subplots(1, len(classes), figsize=(7 * len(classes), 6))
    if len(classes) == 1:
        axes = [axes]

    palette = {"Negative": "#e74c3c", "Positive": "#2ecc71"}

    for ax, cls in zip(axes, classes):
        sub = keywords_df[keywords_df["class"] == cls].nlargest(top_k, "score")
        sub = sub.sort_values("score")
        ax.barh(sub["keyphrase"], sub["score"], color=palette.get(cls, "#3498db"))
        ax.set_title(f"Top keyphrase — {cls}")
        ax.set_xlabel("RAKE score")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[KEYWORDS] Barplot salvato in: {out_path}")


# ============================================================================
# REPORT
# ============================================================================

def write_report(matrix: pd.DataFrame, relations_df: pd.DataFrame,
                  keywords_df: pd.DataFrame, out_path: str):
    lines = []
    sep = "=" * 65
    lines += [sep, "  INFORMATION EXTRACTION — REPORT PER LA RELAZIONE", sep, ""]

    lines += ["  1) RELATION EXTRACTION (spaCy dependency parsing)", "-" * 65]
    lines += ["  Le relazioni (aspetto, descrittore) sono estratte analizzando",
              "  l'albero sintattico delle frasi (amod, nsubj+acomp, conj),",
              "  non tramite semplice matching di parole chiave.", ""]
    lines += ["  Matrice aspetto x sentiment (numero di relazioni):", ""]
    lines += ["  " + matrix.to_string().replace("\n", "\n  "), ""]

    lines += ["", "  Top relazioni (aspetto, descrittore):", ""]
    for row in relations_df.itertuples():
        lines.append(f"    - {row.aspect:<12} + {row.descriptor:<15} "
                      f"(n={row.count}, sentiment prevalente={row.prevalent_sentiment})")

    lines += ["", "  2) KEYWORD EXTRACTION (RAKE, filtro doc_freq>=" 
              f"{Config.MIN_DOC_FREQUENCY})", "-" * 65]
    if keywords_df.empty or "class" not in keywords_df.columns:
        lines.append("  Nessuna keyphrase ha superato il filtro di frequenza minima.")
    else:
        for cls in keywords_df["class"].unique():
            lines.append(f"  Classe: {cls}")
            sub = keywords_df[keywords_df["class"] == cls].nlargest(10, "score")
            for _, row in sub.iterrows():
                lines.append(f"    - {row['keyphrase']:<40} "
                              f"(score={row['score']:.1f}, presente in {row['doc_freq']} recensioni)")
            lines.append("")

    lines += [sep]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[REPORT] Report salvato in: {out_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_reviews", type=int, default=Config.N_REVIEWS)
    parser.add_argument("--top_k", type=int, default=Config.TOP_K_KEYWORDS)
    parser.add_argument("--top_k_relations", type=int, default=Config.TOP_K_RELATIONS)
    args = parser.parse_args()

    banner("INFORMATION EXTRACTION — RELATION EXTRACTION + KEYWORD EXTRACTION")

    nlp = ensure_setup()
    df = load_data(args.n_reviews)

    # --- 1) Relation extraction
    banner("1/2 — RELATION EXTRACTION (dependency parsing)")
    relation_df = extract_aspect_relations(df, nlp)
    matrix = build_aspect_matrix(relation_df)
    matrix.to_csv(os.path.join(Config.OUT_DIR, "aspect_sentiment_matrix.csv"))
    plot_aspect_heatmap(matrix, os.path.join(Config.OUT_DIR, "aspect_sentiment_heatmap.png"))

    relations_table = build_relations_table(relation_df, args.top_k_relations)
    relations_table.to_csv(os.path.join(Config.OUT_DIR, "aspect_relations.csv"), index=False)
    plot_relations(relations_table, os.path.join(Config.OUT_DIR, "aspect_relations_barplot.png"))

    banner("2/2 — KEYWORD EXTRACTION (RAKE)")
    df_en = filter_english_reviews(df)
    label_col = "label" if "label" in df_en.columns else df_en.columns[-1]
    label_names = {0: "Negative", 1: "Positive"}
    keywords_df = extract_keywords_per_class(df_en, label_col, label_names, args.top_k)
    keywords_df.to_csv(os.path.join(Config.OUT_DIR, "keywords_per_class.csv"), index=False)
    plot_keywords(keywords_df, os.path.join(Config.OUT_DIR, "keywords_barplot.png"), args.top_k)

    # --- Report ---
    write_report(matrix, relations_table, keywords_df,
                 os.path.join(Config.OUT_DIR, "ie_report.txt"))

    banner("COMPLETATO")
    print(f"  Tutti i file sono in: {os.path.abspath(Config.OUT_DIR)}/")


if __name__ == "__main__":
    main()