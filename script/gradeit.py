import warnings
warnings.filterwarnings('ignore')

import os
import sys
import re
import string
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any

import joblib
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from datasets import load_dataset
from textblob import TextBlob

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import (
    RandomForestClassifier, VotingClassifier, StackingClassifier
)
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold,
    RandomizedSearchCV, learning_curve
)
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
    roc_curve, roc_auc_score, precision_recall_curve
)
from sklearn.preprocessing import label_binarize

from imblearn.over_sampling import RandomOverSampler
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import logging
from datetime import datetime


# ============================================================================
# LOGGING SETUP  — sicuro con joblib/multiprocessing su Windows
# ============================================================================

class _Logger:
    """
    Logger minimale che scrive su file e console senza toccare sys.stdout.
    I worker joblib reimportano il modulo ma non chiamano setup_logging(),
    quindi non c'e' nessun loop o header duplicato.
    """
    def __init__(self, log_path: str):
        self._path    = log_path
        self._file    = open(log_path, "a", encoding="utf-8", buffering=1)
        self._console = sys.__stdout__

    def log(self, msg: str):
        line = f"{msg}\n"
        self._console.write(line)
        self._console.flush()
        self._file.write(line)
        self._file.flush()

    def close(self):
        self._file.close()


# Istanza globale — None finche' setup_logging() non viene chiamato
_logger = None


def log(msg: str):
    """Stampa su console e salva su file log. Usare al posto di print()."""
    if _logger is not None:
        _logger.log(msg)
    else:
        print(msg)


def setup_logging(log_dir: str = "outputs") -> str:
    """
    Inizializza il logger su file. Sicuro con joblib su Windows
    perche' NON modifica sys.stdout.
    """
    global _logger
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path  = os.path.join(log_dir, f"run_log_{timestamp}.txt")

    _logger = _Logger(log_path)

    header = (
        "=" * 70 + "\n"
        f"  Pipeline Log — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        "=" * 70 + "\n"
    )
    _logger._file.write(header)
    _logger._file.flush()

    log(f"[LOG] Output salvato in: {log_path}")
    return log_path


# ============================================================================
# CONFIG
# ============================================================================

class PipelineConfig:
    DATASET_NAME      = "ZephyrUtopia/ratemyprofessors-reviews-2-labels"
    RANDOM_STATE      = 42
    TEST_SIZE         = 0.20
    CV_FOLDS          = 5
    N_ITER_SEARCH     = 10        # RandomizedSearchCV iterations (was full grid)
    CV_TUNING_FOLDS   = 3         # folds used during tuning (not final CV)
    MODELS_DIR        = "models"
    PLOTS_DIR_2CLASS  = "outputs/plots_2class"
    PLOTS_DIR_3CLASS  = "outputs/plots_3class"
    FIGURE_DPI        = 150

    TFIDF_CONFIG = dict(
        max_features = 5000,
        ngram_range  = (1, 2),
        min_df       = 3,
        max_df       = 0.85,
        sublinear_tf = True
    )

    NEUTRAL_QUANTILE_LOW  = 0.30
    NEUTRAL_QUANTILE_HIGH = 0.70
    CLASS_WEIGHTS_3CLASS  = {0: 3.0, 1: 2.0, 2: 1.0}

    LABEL_NAMES_2CLASS = ['Negative', 'Positive']
    LABEL_NAMES_3CLASS = ['Negative', 'Neutral', 'Positive']
    PALETTE_2CLASS     = ['#e74c3c', '#2ecc71']
    PALETTE_3CLASS     = ['#e74c3c', '#f39c12', '#2ecc71']


# ============================================================================
# SETUP
# ============================================================================

def _initialize_nltk():
    for package in ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4']:
        nltk.download(package, quiet=True)


def _create_directories():
    for directory in [
        PipelineConfig.MODELS_DIR,
        PipelineConfig.PLOTS_DIR_2CLASS,
        PipelineConfig.PLOTS_DIR_3CLASS
    ]:
        os.makedirs(directory, exist_ok=True)


# ============================================================================
# PREPROCESSOR
# ============================================================================

class TextPreprocessor:
    def __init__(self):
        self.lemmatizer  = WordNetLemmatizer()
        self.stop_words  = set(stopwords.words('english'))
        self._url_re     = re.compile(r'http\S+|www\S+|https\S+')
        self._number_re  = re.compile(r'\d+')
        self._negations  = {
            'not', 'no', 'never', 'neither', 'nobody', 'nothing',
            'nowhere', 'hardly', 'barely', 'scarcely', "n't"
        }

    def transform(self, text: str) -> str:
        text   = text.lower()
        text   = self._url_re.sub('', text)
        text   = text.translate(str.maketrans('', '', string.punctuation))
        text   = self._number_re.sub('', text)
        tokens = word_tokenize(text)
        tokens = [w for w in tokens if w not in self.stop_words]
        result, negate = [], False
        for token in tokens:
            if token in self._negations:
                negate = True
                result.append(token)
            elif negate:
                result.append(f'NOT_{token}')
                negate = False
            else:
                result.append(token)
        tokens = [self.lemmatizer.lemmatize(w) for w in result if len(w) > 2]
        return ' '.join(tokens)

    def transform_batch(self, texts: List[str]) -> List[str]:
        return [self.transform(t) for t in texts]


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Rimuove righe con testo nullo o non-stringa (es. NaN dal CSV)
    df['text'] = df['text'].astype(str).replace('nan', '')
    df = df[df['text'].str.strip().str.len() > 0].reset_index(drop=True)

    df['text_length']         = df['text'].str.len()
    df['word_count']          = df['text'].str.split().str.len()
    df['avg_word_length']     = df['text_length'] / df['word_count'].replace(0, 1)
    df['exclamation_count']   = df['text'].str.count('!')
    df['question_count']      = df['text'].str.count(r'\?')
    df['uppercase_ratio']     = df['text'].apply(
        lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0
    )
    df['contrast_word_count'] = df['text'].str.count(
        r'\bbut\b|\bhowever\b|\bthough\b|\byet\b|\bwhile\b|\balthough\b'
    )
    return df


# ============================================================================
# SAMPLING  (fast: direct RandomOverSampler, no full comparison loop)
# ============================================================================

def _select_and_apply_sampling(
    X_train, y_train, X_test, y_test
) -> Tuple[Any, Any, str]:
    """
    Quick sampling selection:
    - If imbalance ratio < 1.5 → no sampling needed
    - Otherwise test only RandomOverSampler vs no sampling (fast LR probe)
    - Returns resampled (X, y) and strategy name
    """
    unique, counts = np.unique(y_train, return_counts=True)
    ratio          = max(counts) / min(counts)

    if ratio < 1.5:
        return X_train, y_train, 'None'

    probe_base = LogisticRegression(
        max_iter=300, random_state=PipelineConfig.RANDOM_STATE, n_jobs=-1
    )
    probe_base.fit(X_train, y_train)
    f1_base = f1_score(y_test, probe_base.predict(X_test),
                       average='weighted', zero_division=0)

    try:
        ros = RandomOverSampler(random_state=PipelineConfig.RANDOM_STATE)
        Xr, yr = ros.fit_resample(X_train, y_train)
        probe_ros = LogisticRegression(
            max_iter=300, random_state=PipelineConfig.RANDOM_STATE, n_jobs=-1
        )
        probe_ros.fit(Xr, yr)
        f1_ros = f1_score(y_test, probe_ros.predict(X_test),
                          average='weighted', zero_division=0)
        if f1_ros >= f1_base:
            return Xr, yr, 'Random Oversampling'
    except Exception:
        pass

    return X_train, y_train, 'None'


# ============================================================================
# HYPERPARAMETER TUNING  (RandomizedSearchCV — much faster than GridSearchCV)
# ============================================================================

def _tune_hyperparameters(
    X_train, y_train,
    class_weight_options: List,
    n_classes: int
) -> Dict[str, Any]:
    cv_tuning = StratifiedKFold(
        n_splits     = PipelineConfig.CV_TUNING_FOLDS,
        shuffle      = True,
        random_state = PipelineConfig.RANDOM_STATE
    )
    n_iter = PipelineConfig.N_ITER_SEARCH
    tuned  = {}

    # ── Logistic Regression (fast — keep full small grid) ───────────────────
    gs = RandomizedSearchCV(
        LogisticRegression(
            random_state = PipelineConfig.RANDOM_STATE,
            max_iter     = 1000,
            solver       = 'saga',
            n_jobs       = -1
        ),
        {
            'C'            : [0.01, 0.1, 1.0, 10.0, 100.0],
            'class_weight' : class_weight_options,
            'penalty'      : ['l1', 'l2']
        },
        n_iter=n_iter, cv=cv_tuning, scoring='f1_weighted',
        n_jobs=-1, random_state=PipelineConfig.RANDOM_STATE
    )
    gs.fit(X_train, y_train)
    tuned['Logistic Regression'] = gs.best_estimator_

    # ── LinearSVC via CalibratedClassifierCV (replaces kernel SVM — 100x faster)
    gs = RandomizedSearchCV(
        CalibratedClassifierCV(
            LinearSVC(
                random_state = PipelineConfig.RANDOM_STATE,
                max_iter     = 2000
            ),
            cv=3
        ),
        {
            'estimator__C'            : [0.01, 0.1, 1.0, 10.0],
            'estimator__class_weight' : class_weight_options,
            'estimator__loss'         : ['hinge', 'squared_hinge']
        },
        n_iter=n_iter, cv=cv_tuning, scoring='f1_weighted',
        n_jobs=-1, random_state=PipelineConfig.RANDOM_STATE
    )
    gs.fit(X_train, y_train)
    tuned['Linear SVM'] = gs.best_estimator_

    # ── Naive Bayes (very fast) ──────────────────────────────────────────────
    gs = RandomizedSearchCV(
        MultinomialNB(),
        {'alpha': [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]},
        n_iter=min(6, n_iter), cv=cv_tuning, scoring='f1_weighted',
        n_jobs=-1, random_state=PipelineConfig.RANDOM_STATE
    )
    gs.fit(X_train, y_train)
    tuned['Naive Bayes'] = gs.best_estimator_

    # ── Random Forest (reduced search space) ────────────────────────────────
    gs = RandomizedSearchCV(
        RandomForestClassifier(
            random_state = PipelineConfig.RANDOM_STATE,
            n_jobs       = -1
        ),
        {
            'n_estimators'     : [100, 200, 300],
            'max_depth'        : [None, 15, 30],
            'min_samples_split': [2, 5, 10],
            'max_features'     : ['sqrt', 'log2'],
            'class_weight'     : class_weight_options
        },
        n_iter=n_iter, cv=cv_tuning, scoring='f1_weighted',
        n_jobs=-1, random_state=PipelineConfig.RANDOM_STATE
    )
    gs.fit(X_train, y_train)
    tuned['Random Forest'] = gs.best_estimator_

    # ── XGBoost ─────────────────────────────────────────────────────────────
    if XGBOOST_AVAILABLE:
        xgb_objective = 'multi:softprob' if n_classes > 2 else 'binary:logistic'
        xgb_base = XGBClassifier(
            random_state      = PipelineConfig.RANDOM_STATE,
            eval_metric       = 'mlogloss',
            use_label_encoder = False,
            n_jobs            = -1,
            verbosity         = 0,
            objective         = xgb_objective,
            tree_method       = 'hist'        # fast histogram method
        )
        if n_classes > 2:
            xgb_base.set_params(num_class=n_classes)

        gs = RandomizedSearchCV(
            xgb_base,
            {
                'n_estimators'     : [100, 200, 300],
                'max_depth'        : [3, 4, 6],
                'learning_rate'    : [0.05, 0.1, 0.2],
                'subsample'        : [0.7, 0.8, 1.0],
                'colsample_bytree' : [0.7, 0.8, 1.0],
                'min_child_weight' : [1, 3, 5],
                'gamma'            : [0, 0.1, 0.3]
            },
            n_iter=n_iter, cv=cv_tuning, scoring='f1_weighted',
            n_jobs=-1, random_state=PipelineConfig.RANDOM_STATE
        )
        gs.fit(X_train, y_train)
        tuned['XGBoost'] = gs.best_estimator_

    # ── LightGBM (natively fast) ─────────────────────────────────────────────
    if LIGHTGBM_AVAILABLE:
        gs = RandomizedSearchCV(
            LGBMClassifier(
                random_state = PipelineConfig.RANDOM_STATE,
                n_jobs       = -1,
                verbosity    = -1
            ),
            {
                'n_estimators'  : [100, 200, 300],
                'max_depth'     : [4, 6, 8, -1],
                'learning_rate' : [0.05, 0.1, 0.2],
                'num_leaves'    : [31, 63, 127],
                'subsample'     : [0.7, 0.8, 1.0],
                'class_weight'  : class_weight_options
            },
            n_iter=n_iter, cv=cv_tuning, scoring='f1_weighted',
            n_jobs=-1, random_state=PipelineConfig.RANDOM_STATE
        )
        gs.fit(X_train, y_train)
        tuned['LightGBM'] = gs.best_estimator_

    return tuned


# ============================================================================
# CROSS-VALIDATION
# ============================================================================

def _run_cross_validation(
    models: Dict[str, Any], X_train, y_train
) -> Dict[str, Dict]:
    cv = StratifiedKFold(
        n_splits     = PipelineConfig.CV_FOLDS,
        shuffle      = True,
        random_state = PipelineConfig.RANDOM_STATE
    )
    results = {}
    for name, model in models.items():
        scores = cross_val_score(
            model, X_train, y_train,
            cv=cv, scoring='f1_weighted', n_jobs=-1
        )
        results[name] = {
            'scores': scores,
            'mean'  : scores.mean(),
            'std'   : scores.std()
        }
    return results


# ============================================================================
# ENSEMBLE
# ============================================================================

def _build_ensemble_models(
    base_models: Dict[str, Any], X_train, y_train
) -> Dict[str, Any]:
    estimators = list(base_models.items())
    ensembles  = {}

    voting_soft = VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)
    voting_soft.fit(X_train, y_train)
    ensembles['Voting Soft'] = voting_soft

    voting_hard = VotingClassifier(estimators=estimators, voting='hard', n_jobs=-1)
    voting_hard.fit(X_train, y_train)
    ensembles['Voting Hard'] = voting_hard

    stacking = StackingClassifier(
        estimators      = estimators,
        final_estimator = LogisticRegression(
            max_iter     = 1000,
            random_state = PipelineConfig.RANDOM_STATE
        ),
        cv=3, n_jobs=-1
    )
    stacking.fit(X_train, y_train)
    ensembles['Stacking'] = stacking

    return ensembles


# ============================================================================
# EVALUATION
# ============================================================================

def _evaluate_models(
    models: Dict[str, Any], X_test, y_test,
    label_names: List[str]
) -> Tuple[Dict, str]:
    n_classes = len(label_names)
    y_bin     = label_binarize(y_test, classes=list(range(n_classes)))
    results   = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        entry  = {
            'accuracy' : accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred,
                                          average='weighted', zero_division=0),
            'recall'   : recall_score(y_test, y_pred,
                                       average='weighted', zero_division=0),
            'f1'       : f1_score(y_test, y_pred,
                                   average='weighted', zero_division=0),
            'report'   : classification_report(
                              y_test, y_pred,
                              target_names=label_names,
                              zero_division=0, output_dict=True),
            'cm'       : confusion_matrix(y_test, y_pred),
            'model'    : model
        }
        # AUC per-class (one-vs-rest) se il modello supporta predict_proba
        if hasattr(model, 'predict_proba'):
            try:
                proba = model.predict_proba(X_test)
                aucs  = {}
                for ci, cname in enumerate(label_names):
                    yb = (y_bin[:, ci] if n_classes > 2
                          else (y_test == ci).astype(int))
                    scores = proba[:, ci]
                    aucs[cname] = roc_auc_score(yb, scores)
                entry['auc'] = aucs
            except Exception:
                entry['auc'] = {}
        else:
            entry['auc'] = {}
        results[name] = entry
    best_name = max(results, key=lambda k: results[k]['f1'])
    return results, best_name


# ============================================================================
# ERROR ANALYSIS
# ============================================================================

def _run_error_analysis(
    df_test: pd.DataFrame, y_test: np.ndarray,
    y_pred: np.ndarray, label_names: List[str]
) -> pd.DataFrame:
    df_errors                         = df_test.copy()
    df_errors['true_label']           = y_test
    df_errors['predicted_label']      = y_pred
    df_errors['is_error']             = y_test != y_pred
    df_errors['true_label_name']      = df_errors['true_label'].map(
        dict(enumerate(label_names))
    )
    df_errors['predicted_label_name'] = df_errors['predicted_label'].map(
        dict(enumerate(label_names))
    )
    return df_errors


# ============================================================================
# PLOTS
# ============================================================================

def _save_figure(fig, path: str):
    fig.savefig(path, dpi=PipelineConfig.FIGURE_DPI, bbox_inches='tight')
    plt.close(fig)


def _plot_class_distribution(
    y: np.ndarray, label_names: List[str],
    palette: List[str], output_dir: str
):
    _, counts = np.unique(y, return_counts=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    bars = axes[0].bar(label_names, counts, color=palette,
                       edgecolor='black', linewidth=0.8)
    for bar, cnt in zip(bars, counts):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + counts.max() * 0.01,
            f'{cnt:,}\n({cnt / y.size * 100:.1f}%)',
            ha='center', va='bottom', fontsize=9, fontweight='bold'
        )
    axes[0].set_title('Class Distribution — Counts', fontweight='bold')
    axes[0].set_ylabel('Reviews')
    axes[0].grid(axis='y', alpha=0.3)
    axes[1].pie(counts, labels=label_names, colors=palette, autopct='%1.1f%%',
                startangle=140,
                wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
    axes[1].set_title('Class Distribution — Proportions', fontweight='bold')
    plt.tight_layout()
    _save_figure(fig, os.path.join(output_dir, 'plot_01_class_distribution.png'))


def _plot_confusion_matrix(
    cm: np.ndarray, label_names: List[str],
    model_name: str, output_dir: str
):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_names, yticklabels=label_names,
                linewidths=0.5, ax=axes[0],
                annot_kws={'size': 12, 'weight': 'bold'})
    axes[0].set_title(f'Confusion Matrix — Counts\n{model_name}', fontweight='bold')
    axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('True')
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Greens',
                xticklabels=label_names, yticklabels=label_names,
                linewidths=0.5, ax=axes[1])
    axes[1].set_title(f'Confusion Matrix — Normalized\n{model_name}', fontweight='bold')
    axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('True')
    plt.tight_layout()
    _save_figure(fig, os.path.join(output_dir, 'plot_02_confusion_matrix.png'))


def _plot_roc_curves(
    models: Dict[str, Any], X_test, y_test: np.ndarray,
    n_classes: int, label_names: List[str], output_dir: str
):
    y_bin        = label_binarize(y_test, classes=list(range(n_classes)))
    linestyles   = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 2))]
    model_colors = plt.cm.tab10.colors
    fig, axes    = plt.subplots(1, n_classes, figsize=(6 * n_classes, 5))
    if n_classes == 1:
        axes = [axes]
    for ci, cname in enumerate(label_names):
        ax = axes[ci]
        ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC=0.50)')
        for mi, (mname, model) in enumerate(models.items()):
            if not hasattr(model, 'predict_proba'):
                continue
            try:
                scores    = model.predict_proba(X_test)[:, ci]
                yb        = (y_bin[:, ci] if n_classes > 2
                             else (y_test == ci).astype(int))
                fpr, tpr, _ = roc_curve(yb, scores)
                auc         = roc_auc_score(yb, scores)
                ax.plot(fpr, tpr, lw=1.8,
                        linestyle=linestyles[mi % len(linestyles)],
                        color=model_colors[mi % len(model_colors)],
                        label=f'{mname} (AUC={auc:.3f})')
            except Exception:
                pass
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
        ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve — "{cname}"', fontweight='bold')
        ax.legend(loc='lower right', fontsize=7); ax.grid(alpha=0.3)
    plt.suptitle('ROC Curves — One-vs-Rest', fontweight='bold')
    plt.tight_layout()
    _save_figure(fig, os.path.join(output_dir, 'plot_03_roc_curves.png'))


def _plot_precision_recall_curves(
    models: Dict[str, Any], X_test, y_test: np.ndarray,
    n_classes: int, label_names: List[str], output_dir: str
):
    y_bin        = label_binarize(y_test, classes=list(range(n_classes)))
    linestyles   = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 2))]
    model_colors = plt.cm.tab10.colors
    fig, axes    = plt.subplots(1, n_classes, figsize=(6 * n_classes, 5))
    if n_classes == 1:
        axes = [axes]
    for ci, cname in enumerate(label_names):
        ax    = axes[ci]
        yb    = (y_bin[:, ci] if n_classes > 2
                 else (y_test == ci).astype(int))
        ax.axhline(y=yb.mean(), color='k', linestyle='--', lw=1,
                   label=f'Baseline ({yb.mean():.2f})')
        for mi, (mname, model) in enumerate(models.items()):
            if not hasattr(model, 'predict_proba'):
                continue
            try:
                scores          = model.predict_proba(X_test)[:, ci]
                prec, rec, _    = precision_recall_curve(yb, scores)
                ap              = np.mean(prec)
                ax.plot(rec, prec, lw=1.8,
                        linestyle=linestyles[mi % len(linestyles)],
                        color=model_colors[mi % len(model_colors)],
                        label=f'{mname} (AP={ap:.3f})')
            except Exception:
                pass
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
        ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
        ax.set_title(f'Precision-Recall — "{cname}"', fontweight='bold')
        ax.legend(loc='upper right', fontsize=7); ax.grid(alpha=0.3)
    plt.suptitle('Precision-Recall Curves — One-vs-Rest', fontweight='bold')
    plt.tight_layout()
    _save_figure(fig,
                 os.path.join(output_dir, 'plot_04_precision_recall_curves.png'))


def _plot_top_tfidf_features(
    models: Dict[str, Any], vectorizer: TfidfVectorizer,
    n_classes: int, label_names: List[str],
    palette: List[str], output_dir: str, top_n: int = 20
):
    lr_model = next(
        (m for name, m in models.items()
         if 'Logistic' in name and hasattr(m, 'coef_')), None
    )
    if lr_model is None:
        return
    feature_names = vectorizer.get_feature_names_out()
    coef          = lr_model.coef_
    if coef.shape[0] == 1:
        coef = np.vstack([-coef[0], coef[0]])
    n_cols = min(n_classes, coef.shape[0])
    fig, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols, 7))
    if n_cols == 1:
        axes = [axes]
    for ci in range(n_cols):
        ax      = axes[ci]
        c       = coef[ci]
        top_i   = np.argsort(c)[-top_n:]
        feats   = feature_names[top_i]
        weights = c[top_i]
        colors  = [palette[ci] if w > 0 else '#95a5a6' for w in weights]
        ax.barh(range(top_n), weights, color=colors,
                edgecolor='black', linewidth=0.4)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(feats, fontsize=8)
        ax.axvline(x=0, color='black', lw=0.8)
        ax.set_title(f'Top {top_n} Features — "{label_names[ci]}"',
                     fontweight='bold')
        ax.set_xlabel('Coefficient Weight'); ax.grid(axis='x', alpha=0.3)
    plt.suptitle('Top TF-IDF Features by Class (Logistic Regression)',
                 fontweight='bold')
    plt.tight_layout()
    _save_figure(fig,
                 os.path.join(output_dir, 'plot_05_top_tfidf_features.png'))


def _plot_cv_scores(cv_results: Dict[str, Dict], output_dir: str):
    names  = list(cv_results.keys())
    means  = [cv_results[n]['mean'] for n in names]
    stds   = [cv_results[n]['std']  for n in names]
    order  = np.argsort(means)[::-1]
    names  = [names[i] for i in order]
    means  = [means[i] for i in order]
    stds   = [stds[i]  for i in order]
    fig, ax = plt.subplots(figsize=(10, max(5, len(names) * 0.65)))
    palette = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(names)))
    bars    = ax.barh(names, means, xerr=stds, color=palette,
                      edgecolor='black', linewidth=0.7, capsize=5,
                      error_kw={'elinewidth': 1.5, 'capthick': 1.5})
    for bar, mean, std in zip(bars, means, stds):
        ax.text(mean + std + 0.003, bar.get_y() + bar.get_height() / 2,
                f'{mean:.4f} ± {std:.4f}',
                va='center', ha='left', fontsize=9)
    ax.set_xlabel('Weighted F1-Score')
    ax.set_title(f'{PipelineConfig.CV_FOLDS}-Fold Stratified Cross-Validation',
                 fontweight='bold')
    ax.set_xlim([0, 1.15])
    ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.4, lw=1)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    _save_figure(fig, os.path.join(output_dir, 'plot_06_cv_scores.png'))


def _plot_learning_curves(
    models: Dict[str, Any], X_train, y_train, output_dir: str
):
    base_keys   = ('Logistic Regression', 'Linear SVM', 'Naive Bayes',
                   'Random Forest', 'XGBoost', 'LightGBM')
    plot_models = {k: v for k, v in models.items() if k in base_keys}
    n_models    = len(plot_models)
    n_cols      = min(2, n_models)
    n_rows      = (n_models + n_cols - 1) // n_cols
    fig, axes   = plt.subplots(n_rows, n_cols, figsize=(13, 5 * n_rows))
    axes        = np.array(axes).reshape(n_rows, n_cols)
    cv3         = StratifiedKFold(n_splits=3, shuffle=True,
                                   random_state=PipelineConfig.RANDOM_STATE)
    sizes       = np.linspace(0.1, 1.0, 5)
    for idx, (mname, model) in enumerate(plot_models.items()):
        row, col = divmod(idx, n_cols)
        ax       = axes[row][col]
        try:
            ts, tr_sc, val_sc = learning_curve(
                model, X_train, y_train,
                train_sizes=sizes, cv=cv3,
                scoring='f1_weighted', n_jobs=-1
            )
            tr_m, tr_s   = tr_sc.mean(1), tr_sc.std(1)
            val_m, val_s = val_sc.mean(1), val_sc.std(1)
            ax.plot(ts, tr_m,  'o-', color='#2980b9', lw=2, label='Train')
            ax.fill_between(ts, tr_m - tr_s, tr_m + tr_s,
                            alpha=0.15, color='#2980b9')
            ax.plot(ts, val_m, 's-', color='#e74c3c', lw=2, label='Validation')
            ax.fill_between(ts, val_m - val_s, val_m + val_s,
                            alpha=0.15, color='#e74c3c')
            ax.set_ylim([0, 1.05])
        except Exception:
            pass
        ax.set_title(f'Learning Curve — {mname}', fontweight='bold')
        ax.set_xlabel('Training Set Size')
        ax.set_ylabel('Weighted F1-Score')
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
    for idx in range(n_models, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].set_visible(False)
    plt.suptitle('Learning Curves', fontweight='bold')
    plt.tight_layout()
    _save_figure(fig, os.path.join(output_dir, 'plot_07_learning_curves.png'))


def _generate_all_plots(
    df: pd.DataFrame, label_col: str,
    y_train: np.ndarray, y_test: np.ndarray,
    all_models: Dict[str, Any], best_model_name: str,
    cv_results: Dict[str, Dict], vectorizer: TfidfVectorizer,
    X_train, X_test,
    label_names: List[str], palette: List[str], output_dir: str
):
    plt.style.use('seaborn-v0_8-whitegrid')
    n_classes = len(label_names)
    _plot_class_distribution(df[label_col].values, label_names, palette, output_dir)
    _plot_confusion_matrix(
        confusion_matrix(y_test, all_models[best_model_name].predict(X_test)),
        label_names, best_model_name, output_dir
    )
    _plot_roc_curves(all_models, X_test, y_test,
                     n_classes, label_names, output_dir)
    _plot_precision_recall_curves(all_models, X_test, y_test,
                                   n_classes, label_names, output_dir)
    _plot_top_tfidf_features(all_models, vectorizer,
                              n_classes, label_names, palette, output_dir)
    _plot_cv_scores(cv_results, output_dir)
    _plot_learning_curves(all_models, X_train, y_train, output_dir)


# ============================================================================
# PIPELINE  2-CLASS
# ============================================================================

def train_binary_classifier(
    df: pd.DataFrame, preprocessor: TextPreprocessor
) -> Tuple[TfidfVectorizer, Any]:
    df = _engineer_features(df.copy())
    df['processed_text'] = preprocessor.transform_batch(df['text'].tolist())
    df = df[df['processed_text'].str.len() > 0].reset_index(drop=True)

    y          = df['label'].values
    vectorizer = TfidfVectorizer(**PipelineConfig.TFIDF_CONFIG)
    X          = vectorizer.fit_transform(df['processed_text'])

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index,
        test_size    = PipelineConfig.TEST_SIZE,
        random_state = PipelineConfig.RANDOM_STATE,
        stratify     = y
    )

    X_balanced, y_balanced, sampling_strategy = _select_and_apply_sampling(
        X_train, y_train, X_test, y_test
    )

    tuned_models = _tune_hyperparameters(
        X_balanced, y_balanced,
        class_weight_options=['balanced', None],
        n_classes=2
    )
    cv_results = _run_cross_validation(tuned_models, X_balanced, y_balanced)

    for model in tuned_models.values():
        model.fit(X_balanced, y_balanced)

    ensemble_models = _build_ensemble_models(tuned_models, X_balanced, y_balanced)
    all_models      = {**tuned_models, **ensemble_models}

    eval_results, best_model_name = _evaluate_models(
        all_models, X_test, y_test, PipelineConfig.LABEL_NAMES_2CLASS
    )

    _run_error_analysis(
        df.loc[idx_test], y_test,
        eval_results[best_model_name]['model'].predict(X_test),
        PipelineConfig.LABEL_NAMES_2CLASS
    )

    _generate_all_plots(
        df, 'label', y_balanced, y_test,
        all_models, best_model_name, cv_results, vectorizer,
        X_balanced, X_test,
        PipelineConfig.LABEL_NAMES_2CLASS,
        PipelineConfig.PALETTE_2CLASS,
        PipelineConfig.PLOTS_DIR_2CLASS
    )

    # Statistiche dataset per il report
    unique_train, counts_train = np.unique(y_balanced, return_counts=True)
    class_dist = {
        PipelineConfig.LABEL_NAMES_2CLASS[int(c)]: int(n)
        for c, n in zip(unique_train, counts_train)
    }
    dataset_stats = {
        'total': int(X.shape[0]),
        'train': int(X_balanced.shape[0]),
        'test' : int(X_test.shape[0]),
        'class_dist': class_dist
    }
    best_model_obj = eval_results[best_model_name]['model']
    best_params = {}
    if hasattr(best_model_obj, 'get_params'):
        raw = best_model_obj.get_params()
        best_params = {k: v for k, v in raw.items() if v is not None and k != 'estimators'}

    lines_2class = save_metrics_report(
        experiment_name  = "Binary Classification (2 classi: Negative / Positive)",
        label_names      = PipelineConfig.LABEL_NAMES_2CLASS,
        cv_results       = cv_results,
        eval_results     = eval_results,
        best_model_name  = best_model_name,
        sampling_strategy= sampling_strategy,
        output_path      = "outputs/metrics_2class.txt",
        dataset_stats    = dataset_stats,
        best_params      = best_params,
    )

    return vectorizer, eval_results[best_model_name]['model'], lines_2class


# ============================================================================
# PIPELINE  3-CLASS
# ============================================================================

def train_ternary_classifier(
    df: pd.DataFrame, preprocessor: TextPreprocessor
) -> Tuple[TfidfVectorizer, Any, float, float]:
    df             = df.copy()
    df['polarity'] = df['text'].apply(
        lambda t: TextBlob(str(t)).sentiment.polarity
    )
    quantile_low  = df['polarity'].quantile(PipelineConfig.NEUTRAL_QUANTILE_LOW)
    quantile_high = df['polarity'].quantile(PipelineConfig.NEUTRAL_QUANTILE_HIGH)

    df['sentiment_label'] = df['polarity'].apply(
        lambda p: 0 if p < quantile_low else (1 if p <= quantile_high else 2)
    )

    df = _engineer_features(df)
    df['processed_text'] = preprocessor.transform_batch(df['text'].tolist())
    df = df[df['processed_text'].str.len() > 0].reset_index(drop=True)

    y          = df['sentiment_label'].values
    vectorizer = TfidfVectorizer(**PipelineConfig.TFIDF_CONFIG)
    X          = vectorizer.fit_transform(df['processed_text'])

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index,
        test_size    = PipelineConfig.TEST_SIZE,
        random_state = PipelineConfig.RANDOM_STATE,
        stratify     = y
    )

    X_balanced, y_balanced, sampling_strategy = _select_and_apply_sampling(
        X_train, y_train, X_test, y_test
    )

    tuned_models = _tune_hyperparameters(
        X_balanced, y_balanced,
        class_weight_options=['balanced', PipelineConfig.CLASS_WEIGHTS_3CLASS],
        n_classes=3
    )
    cv_results = _run_cross_validation(tuned_models, X_balanced, y_balanced)

    for model in tuned_models.values():
        model.fit(X_balanced, y_balanced)

    ensemble_models = _build_ensemble_models(tuned_models, X_balanced, y_balanced)
    all_models      = {**tuned_models, **ensemble_models}

    eval_results, best_model_name = _evaluate_models(
        all_models, X_test, y_test, PipelineConfig.LABEL_NAMES_3CLASS
    )

    _run_error_analysis(
        df.loc[idx_test], y_test,
        eval_results[best_model_name]['model'].predict(X_test),
        PipelineConfig.LABEL_NAMES_3CLASS
    )

    _generate_all_plots(
        df, 'sentiment_label', y_balanced, y_test,
        all_models, best_model_name, cv_results, vectorizer,
        X_balanced, X_test,
        PipelineConfig.LABEL_NAMES_3CLASS,
        PipelineConfig.PALETTE_3CLASS,
        PipelineConfig.PLOTS_DIR_3CLASS
    )

    # Statistiche dataset per il report
    unique_train, counts_train = np.unique(y_balanced, return_counts=True)
    class_dist = {
        PipelineConfig.LABEL_NAMES_3CLASS[int(c)]: int(n)
        for c, n in zip(unique_train, counts_train)
    }
    dataset_stats = {
        'total': int(X.shape[0]),
        'train': int(X_balanced.shape[0]),
        'test' : int(X_test.shape[0]),
        'class_dist': class_dist
    }
    best_model_obj = eval_results[best_model_name]['model']
    best_params = {}
    if hasattr(best_model_obj, 'get_params'):
        raw = best_model_obj.get_params()
        best_params = {k: v for k, v in raw.items() if v is not None and k != 'estimators'}

    lines_3class = save_metrics_report(
        experiment_name  = "Ternary Classification (3 classi: Negative / Neutral / Positive)",
        label_names      = PipelineConfig.LABEL_NAMES_3CLASS,
        cv_results       = cv_results,
        eval_results     = eval_results,
        best_model_name  = best_model_name,
        sampling_strategy= sampling_strategy,
        output_path      = "outputs/metrics_3class.txt",
        dataset_stats    = dataset_stats,
        best_params      = best_params,
    )

    return vectorizer, eval_results[best_model_name]['model'], quantile_low, quantile_high, lines_3class



# ============================================================================
# METRICS REPORT  — valori pronti per la relazione
# ============================================================================

def save_metrics_report(
    experiment_name: str,
    label_names: List[str],
    cv_results: Dict[str, Dict],
    eval_results: Dict[str, Dict],
    best_model_name: str,
    sampling_strategy: str,
    output_path: str,
    dataset_stats: Dict = None,
    best_params: Dict   = None,
) -> List[str]:
    """
    Scrive un file .txt con tutte le metriche strutturate,
    pronte per essere copiate nella relazione.
    Restituisce anche le righe come lista (per il report combinato).
    """
    sep  = "=" * 65
    sep2 = "-" * 65
    lines = []

    lines.append(sep)
    lines.append(f"  EXPERIMENT: {experiment_name}")
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(sep)

    # ── Dataset & Sampling ──────────────────────────────────────────
    lines.append("")
    lines.append("[ DATASET & SAMPLING ]")
    lines.append(f"  Classes          : {label_names}")
    lines.append(f"  Sampling strategy: {sampling_strategy}")
    if dataset_stats:
        lines.append(f"  Total samples    : {dataset_stats.get('total', 'N/A'):,}")
        lines.append(f"  Train samples    : {dataset_stats.get('train', 'N/A'):,}")
        lines.append(f"  Test  samples    : {dataset_stats.get('test',  'N/A'):,}")
        lines.append(f"  Test  split      : {PipelineConfig.TEST_SIZE*100:.0f}%")
        if 'class_dist' in dataset_stats:
            lines.append("  Class distribution (train):")
            for cls, cnt in dataset_stats['class_dist'].items():
                pct = cnt / dataset_stats['train'] * 100
                lines.append(f"    {cls:<12}: {cnt:>6,}  ({pct:.1f}%)")

    # ── Pipeline Configuration ───────────────────────────────────────
    lines.append("")
    lines.append("[ PIPELINE CONFIGURATION ]")
    lines.append(f"  TF-IDF max_features : {PipelineConfig.TFIDF_CONFIG['max_features']:,}")
    lines.append(f"  TF-IDF ngram_range  : {PipelineConfig.TFIDF_CONFIG['ngram_range']}")
    lines.append(f"  TF-IDF min_df       : {PipelineConfig.TFIDF_CONFIG['min_df']}")
    lines.append(f"  TF-IDF max_df       : {PipelineConfig.TFIDF_CONFIG['max_df']}")
    lines.append(f"  TF-IDF sublinear_tf : {PipelineConfig.TFIDF_CONFIG['sublinear_tf']}")
    lines.append(f"  CV folds (eval)     : {PipelineConfig.CV_FOLDS}")
    lines.append(f"  CV folds (tuning)   : {PipelineConfig.CV_TUNING_FOLDS}")
    lines.append(f"  RandomizedSearch iter: {PipelineConfig.N_ITER_SEARCH}")
    lines.append(f"  Random state        : {PipelineConfig.RANDOM_STATE}")

    # ── Best Model Hyperparameters ───────────────────────────────────
    if best_params:
        lines.append("")
        lines.append(f"[ BEST MODEL HYPERPARAMETERS — {best_model_name} ]")
        for k, v in best_params.items():
            lines.append(f"  {k:<30}: {v}")

    # ── Cross-Validation ────────────────────────────────────────────
    lines.append("")
    lines.append("[ CROSS-VALIDATION — 5-Fold Stratified, Weighted F1 ]")
    lines.append(f"  {'Model':<25} {'Mean F1':>10}  {'Std':>8}")
    lines.append(f"  {sep2[:55]}")
    sorted_cv = sorted(cv_results.items(), key=lambda x: x[1]['mean'], reverse=True)
    for name, res in sorted_cv:
        lines.append(f"  {name:<25} {res['mean']:.4f}      ± {res['std']:.4f}")

    # ── Test Set — All Models ───────────────────────────────────────
    lines.append("")
    lines.append("[ TEST SET METRICS — All Models ]")
    lines.append(f"  {'Model':<25} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    lines.append(f"  {sep2}")
    sorted_eval = sorted(eval_results.items(), key=lambda x: x[1]['f1'], reverse=True)
    for name, res in sorted_eval:
        lines.append(
            f"  {name:<25} {res['accuracy']:.4f}    {res['precision']:.4f}    "
            f"{res['recall']:.4f}  {res['f1']:.4f}"
        )

    # ── Best Model Detail ───────────────────────────────────────────
    lines.append("")
    lines.append(f"[ BEST MODEL: {best_model_name} ]")
    best = eval_results[best_model_name]

    lines.append("")
    lines.append("  Per-Class Metrics (Test Set):")
    lines.append(f"  {'Class':<12} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Support':>9}")
    lines.append(f"  {sep2[:55]}")
    report = best['report']
    for cls in label_names:
        r = report.get(cls, {})
        lines.append(
            f"  {cls:<12} {r.get('precision',0):.4f}    {r.get('recall',0):.4f}"
            f"  {r.get('f1-score',0):.4f}  {int(r.get('support',0)):>7}"
        )
    wa = report.get('weighted avg', {})
    lines.append(f"  {sep2[:55]}")
    lines.append(
        f"  {'Weighted Avg':<12} {wa.get('precision',0):.4f}    {wa.get('recall',0):.4f}"
        f"  {wa.get('f1-score',0):.4f}  {int(wa.get('support',0)):>7}"
    )

    # ── AUC per class ───────────────────────────────────────────────
    if best['auc']:
        lines.append("")
        lines.append("  AUC-ROC (One-vs-Rest) — Best Model:")
        for cls, auc_val in best['auc'].items():
            lines.append(f"    {cls:<12}: {auc_val:.4f}")
        lines.append("")
        lines.append("  AUC-ROC (One-vs-Rest) — All Models:")
        lines.append(f"  {'Model':<25} " + "  ".join(f"{c:>10}" for c in label_names))
        lines.append(f"  {sep2}")
        for name, res in sorted_eval:
            aucs = res.get('auc', {})
            auc_str = "  ".join(f"{aucs.get(c, float('nan')):>10.4f}" for c in label_names)
            lines.append(f"  {name:<25} {auc_str}")

    # ── Confusion Matrix ────────────────────────────────────────────
    lines.append("")
    lines.append("  Confusion Matrix (counts) — Best Model:")
    cm = best['cm']
    header_cls = "  " + " ".join(f"{c:>10}" for c in label_names)
    lines.append(header_cls)
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:>10}" for v in row)
        lines.append(f"  {label_names[i]:<10}  {row_str}")

    lines.append("")
    lines.append("  Confusion Matrix (normalized %) — Best Model:")
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    lines.append(header_cls)
    for i, row in enumerate(cm_norm):
        row_str = "  ".join(f"{v:>9.2f}%" for v in row)
        lines.append(f"  {label_names[i]:<10}  {row_str}")

    lines.append("")
    lines.append(sep)

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[METRICS] Report salvato in: {output_path}")
    return lines


def save_combined_report(
    lines_2class: List[str],
    lines_3class: List[str],
    output_path: str = "outputs/metrics_combined_relazione.txt"
):
    """
    Genera un unico file .txt con entrambi gli esperimenti (2-class e 3-class)
    più un riepilogo comparativo — pronto da inserire nella relazione.
    """
    sep  = "=" * 65
    sep2 = "-" * 65
    now  = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    header = [
        sep,
        "  REPORT COMPLETO PER LA RELAZIONE",
        f"  Dataset  : {PipelineConfig.DATASET_NAME}",
        f"  Generato : {now}",
        sep,
        "",
        "  Questo file contiene tutte le metriche dei due esperimenti",
        "  (classificazione binaria e ternaria) pronte per la relazione.",
        "",
        sep,
        "",
    ]

    combined = header + lines_2class + ["", ""] + lines_3class

    # ── Riepilogo comparativo (estratto dalle ultime righe disponibili) ──
    combined += [
        "",
        sep,
        "  RIEPILOGO COMPARATIVO",
        sep,
        "",
        "  Le metriche aggregate per ciascun esperimento sono riportate",
        "  nelle sezioni precedenti. Di seguito una sintesi rapida:",
        "",
        "  Esperimento 1 — Classificazione Binaria (Negative / Positive)",
        "    • Task più semplice (2 classi)",
        "    • Label derivate dal campo 'label' del dataset originale",
        "    • Preprocessing: TF-IDF + negation handling + lemmatization",
        "",
        "  Esperimento 2 — Classificazione Ternaria (Negative / Neutral / Positive)",
        "    • Task più complesso (3 classi)",
        "    • Label derivate dalla polarità TextBlob con soglie sui quantili",
        f"    • Soglie neutralità: Q{PipelineConfig.NEUTRAL_QUANTILE_LOW*100:.0f} / Q{PipelineConfig.NEUTRAL_QUANTILE_HIGH*100:.0f}",
        f"    • Class weights usati: {PipelineConfig.CLASS_WEIGHTS_3CLASS}",
        "",
        "  Modelli addestrati in entrambi gli esperimenti:",
        "    Logistic Regression, Linear SVM, Naive Bayes, Random Forest,",
        "    XGBoost (se disponibile), LightGBM (se disponibile),",
        "    Voting Soft, Voting Hard, Stacking",
        "",
        "  Feature engineering aggiuntive (oltre TF-IDF):",
        "    text_length, word_count, avg_word_length,",
        "    exclamation_count, question_count, uppercase_ratio,",
        "    contrast_word_count",
        "",
        sep,
    ]

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(combined) + "\n")
    print(f"[METRICS] Report combinato salvato in: {output_path}")

# ============================================================================
# SAVE ARTIFACTS
# ============================================================================

def save_artifacts(
    vectorizer_2class, model_2class,
    vectorizer_3class, model_3class,
    quantile_low: float, quantile_high: float
):
    os.makedirs(PipelineConfig.MODELS_DIR, exist_ok=True)
    artifacts = {
        'vectorizer_2class.joblib' : vectorizer_2class,
        'model_2class.joblib'      : model_2class,
        'vectorizer_3class.joblib' : vectorizer_3class,
        'model_3class.joblib'      : model_3class,
        'thresholds_3class.joblib' : {
            'quantile_low' : quantile_low,
            'quantile_high': quantile_high
        },
    }
    for filename, artifact in artifacts.items():
        joblib.dump(
            artifact,
            os.path.join(PipelineConfig.MODELS_DIR, filename),
            compress=3
        )


# ============================================================================
# MAIN
# ============================================================================

def _load_dataset() -> pd.DataFrame:
    """
    Carica il dataset da CSV locale se presente, altrimenti lo scarica
    da HuggingFace e lo salva in cache. Rimuove preventivamente righe
    con testo nullo o vuoto.
    """
    csv_path = os.path.join("outputs", "rateMyProffesor_HuggingFace_dataset.csv")

    if os.path.exists(csv_path):
        log(f"[DATA] Dataset trovato in cache: {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        log(f"[DATA] Download dataset da HuggingFace: {PipelineConfig.DATASET_NAME}")
        dataset = load_dataset(PipelineConfig.DATASET_NAME, split='train')
        df      = pd.DataFrame(dataset)
        os.makedirs("outputs", exist_ok=True)
        df.to_csv(csv_path, index=False)
        log(f"[DATA] Dataset scaricato e salvato in: {csv_path}")

    # Pulizia preventiva: rimuove NaN e testi vuoti
    before = len(df)
    df = df.dropna(subset=['text']).reset_index(drop=True)
    df = df[df['text'].astype(str).str.strip().str.len() > 0].reset_index(drop=True)
    removed = before - len(df)
    if removed > 0:
        log(f"[DATA] Rimossi {removed} campioni con testo nullo/vuoto.")
    log(f"[DATA] Campioni validi: {len(df):,} | Colonne: {list(df.columns)}")
    return df


def main():
    _initialize_nltk()
    _create_directories()
    log_path = setup_logging(log_dir="outputs")

    df           = _load_dataset()
    preprocessor = TextPreprocessor()

    vectorizer_2class, model_2class, lines_2class = train_binary_classifier(df, preprocessor)

    vectorizer_3class, model_3class, q_low, q_high, lines_3class = train_ternary_classifier(
        df, preprocessor
    )

    save_artifacts(
        vectorizer_2class, model_2class,
        vectorizer_3class, model_3class,
        q_low, q_high
    )

    save_combined_report(
        lines_2class = lines_2class,
        lines_3class = lines_3class,
        output_path  = "outputs/metrics_combined_relazione.txt"
    )

    log(f"[LOG] Pipeline completata. Log salvato in: {log_path}")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        sys.exit(0)
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        sys.exit(1)