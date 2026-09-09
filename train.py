# train.py  (project root)
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

DATA_PATH = "dataset/features.csv"
MODEL_DIR = "models"
PATTERNS = ["Doji", "Hammer", "ShootingStar", "SpinningTop",
            "BullishEngulfing", "BearishEngulfing"]


def load_data():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["label"])
    y = df["label"]
    return X, y


def split_and_scale(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    # Fit the scaler on TRAINING data only, then apply it to both sets --
    # fitting on test data would leak test-set statistics into training.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def get_models():
    return {
        "SVM": SVC(kernel="rbf", C=1.0, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "DecisionTree": DecisionTreeClassifier(max_depth=8, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
    }


def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    print(f"\n{'=' * 50}\n{name}\n{'=' * 50}")
    print(f"Accuracy:  {acc:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1 Score:  {f1:.3f}")
    print("\nDetailed report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    return {"name": name, "accuracy": acc, "precision": precision,
            "recall": recall, "f1": f1, "y_pred": y_pred}


def plot_confusion_matrix(name, y_test, y_pred, labels):
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, labels=labels, xticks_rotation=45, cmap="Blues", ax=ax
    )
    ax.set_title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_{name}.png")
    plt.close(fig)


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    X, y = load_data()
    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)

    results = []
    for name, model in get_models().items():
        model.fit(X_train, y_train)
        result = evaluate_model(name, model, X_test, y_test)
        results.append(result)
        plot_confusion_matrix(name, y_test, result["y_pred"], labels=PATTERNS)
        joblib.dump(model, os.path.join(MODEL_DIR, f"{name}.pkl"))

    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

    print(f"\n{'=' * 50}\nModel comparison (sorted by F1 score)\n{'=' * 50}")
    results_sorted = sorted(results, key=lambda r: r["f1"], reverse=True)
    for r in results_sorted:
        print(f"{r['name']:15s} Accuracy={r['accuracy']:.3f}  F1={r['f1']:.3f}")

    best = results_sorted[0]
    print(f"\nBest model: {best['name']} (F1={best['f1']:.3f})")


if __name__ == "__main__":
    main()