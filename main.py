import argparse

from src.preprocessing import load_data, get_features_target, build_preprocessor, split_data
from src.train import get_models, train_all
from src import visualize as viz

DATA_PATH = 'data/global_food_wastage_dataset.csv'


def parse_args():
    parser = argparse.ArgumentParser(description='Food Waste Analysis')
    parser.add_argument('--eda',   action='store_true', help='Pokazi EDA grafike')
    parser.add_argument('--train', action='store_true', help='Treniraj modele i prikazi rezultate')
    parser.add_argument('--all',   action='store_true', help='Pokreni sve')
    return parser.parse_args()


def run_eda(df):
    print('\n--- EDA ---')
    print(df.shape)
    print(df.dtypes)
    print(df.isnull().sum())
    print(df.describe())

    viz.plot_boxplot(df)
    viz.plot_correlation_heatmap(df)
    viz.plot_population_vs_waste(df)


def run_training(df):
    print('\n--- Treniranje modela ---')
    X, y = get_features_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f'Trening: {X_train.shape}, Test: {X_test.shape}')

    preprocessor = build_preprocessor()
    models = get_models()
    results = train_all(models, preprocessor, X_train, X_test, y_train, y_test)

    viz.plot_actual_vs_predicted(y_test, results)
    viz.plot_learning_curve(results['Random Forest']['pipeline'], X_train, y_train)
    viz.plot_feature_importances(results['Random Forest']['pipeline'])


def main():
    args = parse_args()
    df = load_data(DATA_PATH)

    if args.all or args.eda:
        run_eda(df)

    if args.all or args.train:
        run_training(df)

    if not any([args.eda, args.train, args.all]):
        print('Pokreni sa --eda, --train ili --all')
        print('Primer: python main.py --all')


if __name__ == '__main__':
    main()
