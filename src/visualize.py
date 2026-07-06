import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import learning_curve

NUMERIC_COLS = [
    'Total Waste (Tons)',
    'Avg Waste per Capita (Kg)',
    'Population (Million)',
    'Household Waste (%)',
    'Economic Loss (Million $)',
]


def plot_boxplot(df: pd.DataFrame) -> None:
    """Raspodela ukupnog otpada po kategorijama hrane."""
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='Food Category', y='Total Waste (Tons)')
    plt.xticks(rotation=45)
    plt.title('Raspodela ukupnog otpada po kategorijama hrane')
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Korelaciona matrica numerickih atributa."""
    corr = df[NUMERIC_COLS].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Korelacije numerickih atributa')
    plt.tight_layout()
    plt.show()


def plot_population_vs_waste(df: pd.DataFrame) -> None:
    """Scatter: populacija vs ukupni otpad, bojeno po kategoriji hrane."""
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=df,
        x='Population (Million)',
        y='Total Waste (Tons)',
        hue='Food Category',
        alpha=0.7,
    )
    plt.title('Population (Million) vs. Total Waste (Tons) by Food Category')
    plt.xlabel('Population (Million)')
    plt.ylabel('Total Waste (Tons)')
    plt.legend(title='Food Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


def plot_actual_vs_predicted(y_test, results: dict) -> None:
    """Stvarne vs. predvidjene vrednosti za sve modele."""
    plt.figure(figsize=(8, 6))

    for name, res in results.items():
        plt.scatter(y_test, res['preds'], alpha=0.5, label=name)

    lim = [y_test.min(), y_test.max()]
    plt.plot(lim, lim, 'k--', label='Idealna predikcija')
    plt.xlabel('Stvarna vrednost (tone)')
    plt.ylabel('Predvidjena vrednost (tone)')
    plt.legend()
    plt.title('Stvarne vs. predvidjene vrednosti — poredjenje modela')
    plt.tight_layout()
    plt.show()


def plot_learning_curve(pipeline, X_train, y_train) -> None:
    """Learning curve za dati pipeline (R2 metrika)."""
    train_sizes, train_scores, val_scores = learning_curve(
        pipeline, X_train, y_train,
        cv=5, scoring='r2',
        train_sizes=np.linspace(0.1, 1.0, 8),
    )
    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, train_scores.mean(axis=1), label='Trening R2')
    plt.plot(train_sizes, val_scores.mean(axis=1),   label='Validacioni R2')
    plt.xlabel('Velicina trening skupa')
    plt.ylabel('R2')
    plt.legend()
    plt.title('Learning curve — Random Forest')
    plt.tight_layout()
    plt.show()


def plot_feature_importances(pipeline, top_n: int = 10) -> None:
    """Top N najznacajnijih atributa iz Random Forest modela."""
    importances  = pipeline.named_steps['model'].feature_importances_
    feature_names = pipeline.named_steps['prep'].get_feature_names_out()

    feat_imp = (
        pd.Series(importances, index=feature_names)
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(8, 6))
    feat_imp.head(top_n).plot(kind='barh')
    plt.gca().invert_yaxis()
    plt.title(f'Top {top_n} najznacajnijih atributa (Random Forest)')
    plt.xlabel('Relativni znacaj')
    plt.tight_layout()
    plt.show()
