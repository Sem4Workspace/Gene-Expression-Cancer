from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


def compare_classifiers(X, y, compute_svd, n_components=10):
    """
    Compare multiple classifiers on SVD-reduced features.

    Parameters:
        X : ndarray or DataFrame – input data
        y : array-like – class labels
        compute_svd : callable – SVD function
        n_components : int – number of SVD components

    Returns:
        results : dict – {model_name: accuracy}
    """
    U_k, _, _ = compute_svd(X, n_components=n_components)

    X_train, X_test, y_train, y_test = train_test_split(
        U_k, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM": SVC(),
        "Random Forest": RandomForestClassifier(random_state=42)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        results[name] = acc

    return results