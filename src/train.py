import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def get_models() -> dict:
    return {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=200, random_state=42),
    }


def build_pipeline(preprocessor, model) -> Pipeline:
    return Pipeline([
        ('prep', preprocessor),
        ('model', model),
    ])


def evaluate(y_true, y_pred) -> dict:
    return {
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE':  mean_absolute_error(y_true, y_pred),
        'R2':   r2_score(y_true, y_pred),
    }


def train_all(models: dict, preprocessor, X_train, X_test, y_train, y_test) -> dict:
    results = {}

    for name, model in models.items():
        pipe = build_pipeline(preprocessor, model)
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        metrics = evaluate(y_test, preds)

        results[name] = {
            'metrics':  metrics,
            'pipeline': pipe,
            'preds':    preds,
        }

        print(f"{name}: RMSE={metrics['RMSE']:.2f}, MAE={metrics['MAE']:.2f}, R2={metrics['R2']:.4f}")

    return results
