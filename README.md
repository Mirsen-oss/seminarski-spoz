# Global Food Waste Analysis

Analiza globalnog rasipanja hrane korišćenjem Linear Regression i Random Forest modela.

## Struktura projekta

```
food-waste-analysis/
├── data/
│   └── global_food_wastage_dataset.csv
├── src/
│   ├── preprocessing.py   # učitavanje, feature engineering, train/test split
│   ├── train.py           # definisanje i evaluacija modela
│   └── visualize.py       # sve funkcije za vizualizaciju
├── main.py                # entry point
└── requirements.txt
```

## Instalacija

```bash
pip install -r requirements.txt
```

## Pokretanje

```bash
# Samo EDA grafici
python main.py --eda

# Samo treniranje i evaluacija modela
python main.py --train

# Sve odjednom
python main.py --all
```

## Modeli

| Model | Metrike |
|---|---|
| Linear Regression | RMSE, MAE, R2 |
| Random Forest (200 stabala) | RMSE, MAE, R2 |

## Dataset

[Global Food Wastage Dataset](https://www.kaggle.com/datasets/atharvasoundankar/global-food-wastage-dataset-2018-2024) — postaviti CSV u `data/` folder.
