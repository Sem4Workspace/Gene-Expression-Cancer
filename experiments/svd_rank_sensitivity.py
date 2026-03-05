import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def evaluate_svd_ranks(X, y, compute_svd, train_model, ranks=[5,10,20,50]):
    results = {}

    for k in ranks:
        U_k, _, _ = compute_svd(X, n_components=k)

        X_train, X_test, y_train, y_test = train_test_split(
            U_k, y, test_size=0.2, random_state=42, stratify=y
        )

        model = train_model(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        results[k] = acc

    return results