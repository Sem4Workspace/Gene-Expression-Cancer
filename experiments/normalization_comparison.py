from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def compare_normalizations(X_raw, X_log, X_scaled, compute_svd, train_model, y):
    datasets = {
        "Raw": X_raw,
        "Log": X_log,
        "Log+Z": X_scaled
    }

    results = {}

    for name, X in datasets.items():
        U_k, _, _ = compute_svd(X, n_components=10)

        X_train, X_test, y_train, y_test = train_test_split(
            U_k, y, test_size=0.2, random_state=42, stratify=y
        )

        model = train_model(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))

        results[name] = acc

    return results