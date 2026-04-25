from pathlib import Path
from urllib.parse import quote_plus
import base64
import io
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sklearn.datasets import load_iris
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from models import MLPModel, PerceptronModel

BASE_DIR = Path(__file__).resolve().parent

UPLOAD_FOLDER = BASE_DIR / "uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {"csv", "xlsx", "json", "txt", "db"}

IRIS_BINARY_SPECIES = ["setosa", "versicolor"]


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def load_dataset(selected_dataset: str) -> pd.DataFrame:
    if selected_dataset == "Iris dataset":
        iris = load_iris(as_frame=True)
        result = iris.frame.copy()
        result["species"] = result["target"].map(lambda value: iris.target_names[value])
        return result

    data_path = UPLOAD_FOLDER / selected_dataset
    if not data_path.exists():
        raise FileNotFoundError("Uploaded dataset could not be found.")

    extension = data_path.suffix.lower()
    if extension == ".csv":
        return pd.read_csv(data_path)
    if extension in {".xls", ".xlsx"}:
        return pd.read_excel(data_path)
    if extension == ".json":
        return pd.read_json(data_path)
    if extension == ".txt":
        try:
            return pd.read_csv(data_path)
        except Exception:
            return pd.read_csv(data_path, sep="\t")

    raise ValueError("Unsupported file type for model training.")


def preprocess_features(df: pd.DataFrame, target_column: str):
    X = df.drop(columns=[target_column]).copy()
    if X.shape[1] == 0:
        raise ValueError("Dataset has no features available for training.")

    numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = X.select_dtypes(include=[object, "category"]).columns.tolist()

    for column in numeric_columns:
        X[column] = X[column].astype(float).fillna(X[column].median())

    for column in categorical_columns:
        X[column] = X[column].fillna("missing")

    if categorical_columns:
        X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)

    if X.shape[1] == 0:
        raise ValueError("No usable features were found after preprocessing.")

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    return X_scaled


def prepare_target(y: pd.Series, task: str):
    if task in {"binary", "multi"}:
        y_values = y.astype(str).fillna("missing")
        label_encoder = LabelEncoder().fit(y_values)
        encoded_labels = label_encoder.transform(y_values)
        classes = list(label_encoder.classes_)

        if task == "multi":
            one_hot = np.eye(len(classes))[encoded_labels]
            return one_hot, encoded_labels, classes

        return encoded_labels, encoded_labels, classes

    if task == "regression":
        if not pd.api.types.is_numeric_dtype(y):
            raise ValueError("Regression requires a numeric target column.")
        return y.astype(float).fillna(y.median()).to_numpy(), None, None

    raise ValueError("Unknown task type.")


def prepare_dataset(df: pd.DataFrame, task: str):
    target_column = df.columns[-1]
    target = df[target_column]

    if task == "regression" and not pd.api.types.is_numeric_dtype(target):
        raise ValueError("Regression requires the final column to be numeric.")
    if task == "binary" and target.nunique() != 2:
        raise ValueError("Binary classification requires a target column with exactly 2 distinct values.")
    if task == "multi" and target.nunique() < 3:
        raise ValueError("Multi-class classification requires at least 3 distinct target classes.")

    X_processed = preprocess_features(df, target_column).to_numpy()
    y_processed, y_labels, classes = prepare_target(target, task)
    return X_processed, y_processed, y_labels, classes, target_column


def train_and_evaluate(X, y_processed, y_labels, task: str, model, split_ratio: float = 0.2):
    if len(X) < 8:
        raise ValueError("The dataset is too small for reliable training.")

    if task == "regression":
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_processed, test_size=split_ratio, random_state=42
        )
        y_train_labels = y_test_labels = None
    else:
        X_train, X_test, y_train, y_test, y_train_labels, y_test_labels = train_test_split(
            X,
            y_processed,
            y_labels,
            test_size=split_ratio,
            random_state=42,
            stratify=y_labels,
        )

    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)

    if task in {"binary", "multi"}:
        fit_target = y_train if isinstance(model, MLPModel) else y_train_labels
    else:
        fit_target = y_train

    if isinstance(model, BaseEstimator):
        model.fit(X_train, fit_target)
    else:
        model.fit(X_train, fit_target, task)

    train_prediction = model.predict(X_train)
    test_prediction = model.predict(X_test)
    history = getattr(model, "history", None)
    if history is None:
        history = {
            "loss": getattr(model, "loss_curve_", []).tolist() if hasattr(model, "loss_curve_") else [],
            "accuracy": [],
        }

    if task in {"binary", "multi"}:
        train_accuracy = accuracy_score(y_train_labels, train_prediction)
        test_accuracy = accuracy_score(y_test_labels, test_prediction)
        loss_name = "binary cross-entropy" if task == "binary" else "categorical cross-entropy"
        algorithm = (
            "historical perceptron updates"
            if isinstance(model, PerceptronModel)
            else "backpropagation with gradient descent"
        )
        regularization = " L2 regularization applied." if getattr(model, "l2", 0) else ""
        return {
            "metric_text": f"Accuracy: {test_accuracy:.2%}",
            "details": (
                f"Trained a {model.__class__.__name__} with {X.shape[1]} features using {loss_name} loss and {algorithm}."
                + regularization
            ),
            "y_test": y_test_labels,
            "prediction": test_prediction,
            "train_actual": y_train_labels,
            "train_prediction": train_prediction,
            "train_score": train_accuracy,
            "test_score": test_accuracy,
            "history": history,
            "X_test": X_test,
            "model": model,
        }

    y_train = np.ravel(y_train)
    y_test = np.ravel(y_test)
    train_prediction = np.ravel(train_prediction)
    test_prediction = np.ravel(test_prediction)
    train_rmse = float(np.sqrt(mean_squared_error(y_train, train_prediction)))
    test_rmse = float(np.sqrt(mean_squared_error(y_test, test_prediction)))
    train_r2 = float(r2_score(y_train, train_prediction))
    test_r2 = float(r2_score(y_test, test_prediction))
    regularization = " L2 regularization applied." if getattr(model, "l2", 0) else ""
    history = getattr(model, "history", None)
    if history is None:
        history = {
            "loss": getattr(model, "loss_curve_", []).tolist() if hasattr(model, "loss_curve_") else [],
            "accuracy": [],
        }
    return {
        "metric_text": f"RMSE: {test_rmse:.3f}, R²: {test_r2:.3f}",
        "details": (
            f"Trained a {model.__class__.__name__} with {X.shape[1]} features using mean squared error loss." + regularization
        ),
        "y_test": y_test,
        "prediction": test_prediction,
        "train_actual": y_train,
        "train_prediction": train_prediction,
        "train_score": train_rmse,
        "test_score": test_rmse,
        "history": history,
        "X_test": X_test,
        "model": model,
    }


def create_image_from_figure(fig):
    buffer = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buffer, format="png", dpi=120)
    plt.close(fig)
    buffer.seek(0)
    encoded = base64.b64encode(buffer.read()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def plot_confusion_matrix(y_test, prediction, labels):
    matrix = confusion_matrix(y_test, prediction)
    fig, ax = plt.subplots(figsize=(6, 4))
    cax = ax.matshow(matrix, cmap="Blues")
    fig.colorbar(cax, ax=ax)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, matrix[i, j], ha="center", va="center", color="black")
    return create_image_from_figure(fig)


def plot_distribution(y, labels):
    counts = pd.Series(y).value_counts().sort_index()
    names = [labels[i] for i in counts.index]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(names, counts.values, color="#60a5fa")
    ax.set_title("Target Class Distribution")
    ax.set_ylabel("Count")
    ax.set_xlabel("Class")
    return create_image_from_figure(fig)


def plot_regression_results(y_test, prediction):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_test, prediction, color="#38bdf8", edgecolor="#0f172a", alpha=0.8)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="#1d4ed8", linestyle="--")
    ax.set_title("True vs Predicted")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    return create_image_from_figure(fig)


def plot_training_curves(history, task):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["loss"], marker="o", color="#0ea5e9")
    axes[0].set_title("Loss vs Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.25)

    if task != "regression" and history.get("accuracy"):
        axes[1].plot(history["accuracy"], marker="o", color="#22c55e")
        axes[1].set_title("Accuracy vs Epochs")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].grid(True, alpha=0.25)
    else:
        axes[1].text(0.5, 0.5, "Accuracy is not available for regression.", ha="center", va="center", fontsize=12)
        axes[1].set_axis_off()

    return create_image_from_figure(fig)


def plot_train_test_comparison(task, train_actual, train_prediction, test_actual, test_prediction, classes=None):
    if task == "regression":
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(train_actual, train_prediction, color="#60a5fa", edgecolor="#0f172a", alpha=0.7, label="Train")
        ax.scatter(test_actual, test_prediction, color="#f97316", edgecolor="#0f172a", alpha=0.7, label="Test")
        ax.plot([min(train_actual.min(), test_actual.min()), max(train_actual.max(), test_actual.max())],
                [min(train_actual.min(), test_actual.min()), max(train_actual.max(), test_actual.max())],
                color="#1d4ed8", linestyle="--")
        ax.set_title("Train vs Test Predictions")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.legend()
        ax.grid(True, alpha=0.2)
        return create_image_from_figure(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    train_acc = accuracy_score(train_actual, train_prediction)
    test_acc = accuracy_score(test_actual, test_prediction)
    bars = ax.bar(["Train", "Test"], [train_acc, test_acc], color=["#38bdf8", "#fbbf24"])
    ax.set_ylim(0, 1)
    ax.set_title("Train vs Test Accuracy")
    ax.set_ylabel("Accuracy")
    for bar, score in zip(bars, [train_acc, test_acc]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{score:.2%}", ha="center")
    return create_image_from_figure(fig)


def plot_experiment_comparison(results, title, task):
    names = [item["name"] for item in results]
    values = [item["test_score"] for item in results]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(names, values, color=["#0ea5e9" if i == 0 else "#f97316" for i in range(len(values))])
    label = "Accuracy" if task in {"binary", "multi"} else "Test RMSE"
    if task in {"binary", "multi"}:
        ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.set_ylabel(label)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + (0.02 if task != "regression" else 0.05), f"{value:.2%}" if task in {"binary", "multi"} else f"{value:.3f}", ha="center")
    ax.grid(True, alpha=0.2, axis="y")
    return create_image_from_figure(fig)


def plot_decision_boundary(model, X, y, task, classes=None):
    if X.shape[1] < 2:
        raise ValueError("Decision boundary visualization requires at least 2 features.")

    X_plot = np.asarray(X)
    feature_names = ["Feature 1", "Feature 2"]
    x_min, x_max = X_plot[:, 0].min() - 1, X_plot[:, 0].max() + 1
    y_min, y_max = X_plot[:, 1].min() - 1, X_plot[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 120),
        np.linspace(y_min, y_max, 120),
    )
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    if X_plot.shape[1] > 2:
        mean_rest = np.mean(X_plot[:, 2:], axis=0, keepdims=True)
        repeated_rest = np.repeat(mean_rest, grid_points.shape[0], axis=0)
        grid_points = np.hstack([grid_points, repeated_rest])

    predictions = model.predict(grid_points)
    Z = np.array(predictions).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(6, 4))
    contour = ax.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
    scatter = ax.scatter(X_plot[:, 0], X_plot[:, 1], c=y, cmap="coolwarm", edgecolor="#0f172a", s=50)
    ax.set_title("Decision Boundary")
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    if classes is not None:
        legend_labels = [classes[i] for i in np.unique(y)]
        handles, _ = scatter.legend_elements()
        ax.legend(handles, legend_labels, title="Classes")
    return create_image_from_figure(fig)


def plot_feature_space_3d(X, y, task, classes=None):
    if X.shape[1] < 3:
        raise ValueError("3D feature visualization requires at least 3 features.")

    X_plot = np.asarray(X)
    x_vals = X_plot[:, 0]
    y_vals = X_plot[:, 1]
    z_vals = X_plot[:, 2]

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")

    if task in {"binary", "multi"}:
        labels = np.asarray(y)
        scatter = ax.scatter(x_vals, y_vals, z_vals, c=labels, cmap="tab10", s=45, edgecolor="#0f172a")
        if classes is not None:
            handles, _ = scatter.legend_elements()
            legend_labels = [classes[i] for i in range(min(len(classes), len(handles)))]
            ax.legend(handles, legend_labels, title="Classes")
    else:
        values = np.asarray(y)
        scatter = ax.scatter(x_vals, y_vals, z_vals, c=values, cmap="viridis", s=45, edgecolor="#0f172a")
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.7)
        cbar.set_label("Predicted value")

    ax.set_title("3D Feature Space")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_zlabel("Feature 3")
    return create_image_from_figure(fig)


def parse_neurons_per_layer(neurons: str, hidden_layers: int) -> tuple[int, ...]:
    if hidden_layers <= 0:
        return ()

    values = [int(value.strip()) for value in neurons.split(",") if value.strip()]
    if not values:
        raise ValueError("Please enter one or more neuron counts for the hidden layers.")

    if len(values) < hidden_layers:
        values.extend([values[-1]] * (hidden_layers - len(values)))

    return tuple(values[:hidden_layers])


def normalize_activation(activation: str) -> str:
    activation_key = activation.strip().lower()
    if activation_key not in {"sigmoid", "relu", "tanh", "softmax"}:
        raise ValueError("Activation must be Sigmoid, ReLu, Tanh, or Softmax.")
    return activation_key


def build_model(task: str, architecture: str, hidden_layer_sizes: tuple[int, ...], activation: str, learning_rate: float = 0.05, epochs: int = 200, l2: float = 0.0):
    output_activation = None
    if task == "binary":
        output_activation = "sigmoid"
    elif task == "multi":
        output_activation = "softmax"

    if architecture == "historical_perceptron":
        if task in {"binary", "multi"}:
            return PerceptronModel(epochs=epochs, lr=learning_rate, l2=l2)
        return MLPModel(
            layer_sizes=(),
            activation=activation,
            output_activation=None,
            learning_rate=learning_rate,
            epochs=epochs,
            l2=l2,
            batch_size=0,
        )

    if architecture == "single_perceptron":
        return MLPModel(
            layer_sizes=(),
            activation=activation,
            output_activation=output_activation,
            learning_rate=learning_rate,
            epochs=epochs,
            l2=l2,
            batch_size=0,
        )

    if architecture == "sklearn_linear":
        from sklearn.linear_model import LogisticRegression, LinearRegression
        if task == "regression":
            return LinearRegression()
        return LogisticRegression(max_iter=epochs, solver="lbfgs")

    if architecture == "sklearn_mlp":
        from sklearn.neural_network import MLPClassifier, MLPRegressor
        sklearn_activation = activation
        if sklearn_activation == "sigmoid":
            sklearn_activation = "logistic"
        elif sklearn_activation == "softmax":
            sklearn_activation = "relu"
        if task == "regression":
            return MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation=sklearn_activation, max_iter=epochs, early_stopping=True)
        return MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=sklearn_activation, max_iter=epochs, early_stopping=True)

    return MLPModel(
        layer_sizes=hidden_layer_sizes,
        activation=activation,
        output_activation=output_activation,
        learning_rate=learning_rate,
        epochs=epochs,
        l2=l2,
        batch_size=16,
        early_stopping=True,
        patience=20,
    )


def build_task_description(task: str) -> str:
    return {
        "binary": "Binary Classification",
        "multi": "Multi-Class Classification",
        "regression": "Regression",
    }[task]


def build_iris_dataset(task: str) -> dict:
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    df["species"] = df["target"].map(lambda value: iris.target_names[value])

    if task == "binary":
        df = df[df["species"].isin(IRIS_BINARY_SPECIES)].reset_index(drop=True)
        target_column = "species"
    elif task == "multi":
        target_column = "species"
    else:
        target_column = "sepal length (cm)"
        df = df.drop(columns=["target"])
        df = df[["sepal width (cm)", "petal length (cm)", "petal width (cm)", target_column]]
        return {
            "df": df,
            "target_column": target_column,
            "message": "Using remaining numeric iris features to regress sepal length.",
        }

    return {"df": df, "target_column": target_column, "message": "Using Iris dataset for model training."}


def analyze_dataset(
    df: pd.DataFrame,
    task: str,
    architecture: str,
    hidden_layers: int,
    neurons: str,
    activation: str,
    learning_rate: float,
    epochs: int,
    l2: float,
    split_ratio: float,
) -> dict:
    if df.shape[1] < 2:
        raise ValueError("Dataset must have at least one feature column and one target column.")

    X_processed, y_processed, y_labels, classes, target_column = prepare_dataset(df, task)
    hidden_layer_sizes = parse_neurons_per_layer(neurons, hidden_layers)
    activation_key = normalize_activation(activation)
    model = build_model(task, architecture, hidden_layer_sizes, activation_key, learning_rate, epochs, l2)
    evaluation = train_and_evaluate(X_processed, y_processed, y_labels, task, model, split_ratio)
    
    # Generate sklearn comparison
    sklearn_comparison = []
    if task in {"binary", "multi"}:
        from sklearn.neural_network import MLPClassifier as SklearnMLP
        from sklearn.linear_model import Perceptron as SklearnPerceptron
        
        sklearn_model = SklearnMLP(hidden_layer_sizes=(10,) if not hidden_layer_sizes else hidden_layer_sizes, 
                                    max_iter=epochs, random_state=42)
        sklearn_model.fit(X_processed, y_processed)
        sklearn_score = sklearn_model.score(X_processed, y_processed)
        
        sklearn_comparison.append({
            "name": "sklearn MLP",
            "score": f"{sklearn_score:.4f}",
            "details": f"Built-in sklearn MLP with hidden layers {hidden_layer_sizes or (10,)}"
        })
    
    # Generate predictions list for interactive interface
    predictions = []
    y_test = evaluation["y_test"]
    y_pred = evaluation["prediction"]
    correct_count = 0
    incorrect_count = 0
    
    for i in range(min(len(y_test), 50)):  # Limit to 50 predictions
        actual = str(y_test[i])
        predicted = str(y_pred[i])
        is_correct = actual == predicted
        if is_correct:
            correct_count += 1
        else:
            incorrect_count += 1
        
        predictions.append({
            "index": i,
            "actual": actual,
            "predicted": predicted,
            "correct": is_correct
        })
    
    accuracy_percent = (correct_count / len(y_test) * 100) if len(y_test) > 0 else 0
    
    # Generate network visualization data
    network_visualization = {
        "layers": [
            {"neurons": X_processed.shape[1], "type": "input"},
            *([{"neurons": size, "type": "hidden"} for size in hidden_layer_sizes] if hidden_layer_sizes else []),
            {"neurons": len(classes) if classes else 1, "type": "output"}
        ]
    }
    
    return {
        "usable": True,
        "dataset_info": f"{len(df)} rows, {X_processed.shape[1]} features, target '{target_column}'",
        "preprocess_info": "Imputed missing values, encoded categorical features, and standardized numeric inputs.",
        "metric_text": evaluation["metric_text"],
        "details": evaluation["details"],
        "y_test": evaluation["y_test"],
        "prediction": evaluation["prediction"],
        "classes": classes,
        "architecture": architecture,
        "hidden_layers": hidden_layers,
        "neurons": ",".join(str(n) for n in hidden_layer_sizes) if hidden_layer_sizes else "none",
        "activation": activation,
        "history": evaluation["history"],
        "train_actual": evaluation["train_actual"],
        "train_prediction": evaluation["train_prediction"],
        "X_test": evaluation["X_test"],
        "model": evaluation["model"],
        "learning_rate": learning_rate,
        "epochs": epochs,
        "l2": l2,
        "split_ratio": split_ratio,
        "sklearn_comparison": sklearn_comparison,
        "predictions": predictions,
        "correct_count": correct_count,
        "incorrect_count": incorrect_count,
        "accuracy_percent": round(accuracy_percent, 2),
        "network_visualization": network_visualization,
    }


app = FastAPI()
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app = FastAPI(debug=True)

@app.get("/")
async def index(
    request: Request,
    message: str | None = None,
    selected_dataset: str | None = None,
):
    uploaded_files = [p.name for p in UPLOAD_FOLDER.iterdir() if p.is_file()]
    selected_query = (
        f"?selected_dataset={quote_plus(selected_dataset)}" if selected_dataset else ""
    )
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "uploaded_files": uploaded_files,
            "message": message,
            "selected_dataset": selected_dataset,
            "selected_query": selected_query,
        },
    )


@app.get("/course")
async def course(request: Request):
    return templates.TemplateResponse(
        "course.html",
        {"request": request},
    )


@app.post("/")
async def upload(database: UploadFile | None = File(None)):
    if database is None or not database.filename:
        message = quote_plus("No dataset file selected.")
        return RedirectResponse(url=f"/?message={message}", status_code=303)

    if not allowed_file(database.filename):
        message = quote_plus("Only CSV, XLSX, JSON, TXT, or DB files are allowed.")
        return RedirectResponse(url=f"/?message={message}", status_code=303)

    destination = UPLOAD_FOLDER / database.filename
    contents = await database.read()
    destination.write_bytes(contents)

    message = quote_plus(f"Database saved to {destination.name}")
    selected_dataset = quote_plus(destination.name)
    return RedirectResponse(
        url=f"/?message={message}&selected_dataset={selected_dataset}",
        status_code=303,
    )


@app.get("/iris")
async def iris():
    message = quote_plus("Iris dataset selected.")
    selected_dataset = quote_plus("Iris dataset")
    return RedirectResponse(
        url=f"/?message={message}&selected_dataset={selected_dataset}",
        status_code=303,
    )


@app.get("/model")
async def model(
    request: Request,
    selected_dataset: str | None = None,
):
    selected_query = (
        f"?selected_dataset={quote_plus(selected_dataset)}" if selected_dataset else ""
    )
    return templates.TemplateResponse(
        "model.html",
        {
            "request": request,
            "selected_dataset": selected_dataset,
            "selected_query": selected_query,
        },
    )


@app.get("/configure_model")
async def configure_model(
    request: Request,
    task: str,
    selected_dataset: str | None = None,
):
    if not selected_dataset:
        message = quote_plus("Please select a dataset before choosing a model.")
        return RedirectResponse(url=f"/?message={message}", status_code=303)

    return templates.TemplateResponse(
        "configure_model.html",
        {
            "request": request,
            "task": task,
            "selected_dataset": selected_dataset,
            "selected_query": f"?selected_dataset={quote_plus(selected_dataset)}",
        },
    )


@app.get("/run_model")
async def run_model(
    request: Request,
    task: str,
    model_architecture: str,
    hidden_layers: int = 0,
    neurons: str = "",
    activation: str = "Sigmoid",
    experiment: str = "baseline",
    learning_rate: float = 0.05,
    epochs: int = 200,
    l2: float = 0.0,
    split_ratio: float = 0.2,
    selected_dataset: str | None = None,
):
    if not selected_dataset:
        message = quote_plus("Please select a dataset first.")
        return RedirectResponse(url=f"/?message={message}", status_code=303)

    task_key = task.lower()
    if task_key not in {"binary", "multi", "regression"}:
        message = quote_plus("Unknown task selected.")
        return RedirectResponse(url=f"/?message={message}", status_code=303)

    experiment_map = {
        "baseline": "Baseline training",
        "compare_models": "Perceptron vs MLP",
        "layer_effect": "Effect of number of layers",
        "regularization": "Overfitting vs regularization",
    }
    experiment_name = experiment_map.get(experiment, "Baseline training")
    split_ratio = max(0.05, min(0.45, split_ratio))
    selected_query = f"?selected_dataset={quote_plus(selected_dataset)}"
    task_name = build_task_description(task_key)
    task_description = (
        "Binary classification uses two target classes.",
        "Multi-class classification uses three or more target classes.",
        "Regression predicts a numeric value from input features.",
    )[["binary", "multi", "regression"].index(task_key)]

    hidden_layer_sizes = parse_neurons_per_layer(neurons, hidden_layers)
    activation_key = normalize_activation(activation)

    try:
        if selected_dataset == "Iris dataset":
            iris_info = build_iris_dataset(task_key)
            df = iris_info["df"]
            X_processed, y_processed, y_labels, classes, _ = prepare_dataset(df, task_key)
            result = analyze_dataset(
                df,
                task_key,
                model_architecture,
                hidden_layers,
                neurons,
                activation,
                learning_rate,
                epochs,
                l2,
                split_ratio,
            )
            result["details"] = iris_info["message"] + " " + result["details"]
        else:
            df = load_dataset(selected_dataset)
            X_processed, y_processed, y_labels, classes, _ = prepare_dataset(df, task_key)
            result = analyze_dataset(
                df,
                task_key,
                model_architecture,
                hidden_layers,
                neurons,
                activation,
                learning_rate,
                epochs,
                l2,
                split_ratio,
            )

        comparison_results = []
        if result.get("usable") and experiment == "compare_models" and task_key in {"binary", "multi"}:
            perceptron_model = build_model(task_key, "historical_perceptron", (), activation_key, learning_rate, epochs, l2)
            mlp_model = build_model(task_key, "multi_layer_perceptron", hidden_layer_sizes, activation_key, learning_rate, epochs, l2)
            perc_eval = train_and_evaluate(X_processed, y_processed, y_labels, task_key, perceptron_model, split_ratio)
            mlp_eval = train_and_evaluate(X_processed, y_processed, y_labels, task_key, mlp_model, split_ratio)
            comparison_results = [
                {"name": "Historical Perceptron", "test_score": perc_eval["test_score"], "metric_text": perc_eval["metric_text"]},
                {"name": "MLP", "test_score": mlp_eval["test_score"], "metric_text": mlp_eval["metric_text"]},
            ]
        elif result.get("usable") and experiment == "layer_effect" and task_key != "regression":
            layer_comparisons = []
            for depth in [0, 1, 2]:
                if depth == 0:
                    layer_sizes = ()
                elif hidden_layer_sizes:
                    layer_sizes = tuple(hidden_layer_sizes[:depth])
                else:
                    layer_sizes = tuple([10] * depth)
                model_variant = build_model(task_key, "multi_layer_perceptron", layer_sizes, activation_key, learning_rate, epochs, l2)
                eval_variant = train_and_evaluate(X_processed, y_processed, y_labels, task_key, model_variant, split_ratio)
                layer_comparisons.append({"name": f"{depth} hidden", "test_score": eval_variant["test_score"], "metric_text": eval_variant["metric_text"]})
            comparison_results = layer_comparisons
        elif result.get("usable") and experiment == "regularization":
            no_reg_model = build_model(task_key, model_architecture, hidden_layer_sizes, activation_key, learning_rate, epochs, 0.0)
            reg_model = build_model(task_key, model_architecture, hidden_layer_sizes, activation_key, learning_rate, epochs, max(l2, 0.1))
            eval_no_reg = train_and_evaluate(X_processed, y_processed, y_labels, task_key, no_reg_model, split_ratio)
            eval_reg = train_and_evaluate(X_processed, y_processed, y_labels, task_key, reg_model, split_ratio)
            comparison_results = [
                {"name": "No Regularization", "test_score": eval_no_reg["test_score"], "metric_text": eval_no_reg["metric_text"]},
                {"name": "With Regularization", "test_score": eval_reg["test_score"], "metric_text": eval_reg["metric_text"]},
            ]

        if result.get("usable") and experiment == "baseline" and model_architecture in {"sklearn_linear", "sklearn_mlp"}:
            selected_model = build_model(task_key, model_architecture, hidden_layer_sizes, activation_key, learning_rate, epochs, l2)
            baseline_arch = "single_perceptron" if model_architecture == "sklearn_linear" else "multi_layer_perceptron"
            baseline_hidden = hidden_layer_sizes if hidden_layer_sizes else (10,)
            baseline_model = build_model(task_key, baseline_arch, baseline_hidden, activation_key, learning_rate, epochs, l2)
            selected_eval = train_and_evaluate(X_processed, y_processed, y_labels, task_key, selected_model, split_ratio)
            baseline_eval = train_and_evaluate(X_processed, y_processed, y_labels, task_key, baseline_model, split_ratio)
            comparison_results = [
                {"name": "Scikit-learn baseline", "test_score": selected_eval["test_score"], "metric_text": selected_eval["metric_text"]},
                {"name": "Custom baseline", "test_score": baseline_eval["test_score"], "metric_text": baseline_eval["metric_text"]},
            ]

        plot_images = []
        if result.get("usable"):
            plot_images.append({
                "title": "Training curves",
                "src": plot_training_curves(result["history"], task_key),
            })
            plot_images.append({
                "title": "Train vs Test",
                "src": plot_train_test_comparison(
                    task_key,
                    result["train_actual"],
                    result["train_prediction"],
                    result["y_test"],
                    result["prediction"],
                    classes=result.get("classes"),
                ),
            })

            if task_key in {"binary", "multi"}:
                class_labels = result.get("classes") or [str(label) for label in sorted(set(result["y_test"]))]
                plot_images.append({
                    "title": "Class distribution",
                    "src": plot_distribution(result["y_test"], class_labels),
                })
                plot_images.append({
                    "title": "Confusion matrix",
                    "src": plot_confusion_matrix(result["y_test"], result["prediction"], class_labels),
                })
                try:
                    plot_images.append({
                        "title": "Decision boundary",
                        "src": plot_decision_boundary(
                            model=result["model"],
                            X=result["X_test"],
                            y=result["y_test"],
                            task=task_key,
                            classes=class_labels,
                        ),
                    })
                except Exception:
                    pass

                if result["X_test"].shape[1] >= 3:
                    try:
                        plot_images.append({
                            "title": "3D feature space",
                            "src": plot_feature_space_3d(
                                result["X_test"],
                                result["prediction"] if task_key == "regression" else result["y_test"],
                                task_key,
                                classes=class_labels if task_key != "regression" else None,
                            ),
                        })
                    except Exception:
                        pass
            else:
                plot_images.append({
                    "title": "Regression fit",
                    "src": plot_regression_results(result["y_test"], result["prediction"]),
                })

        if comparison_results:
            plot_images.append({
                "title": "Experiment comparison",
                "src": plot_experiment_comparison(comparison_results, experiment_name, task_key),
            })

        return templates.TemplateResponse(
            "model_result.html",
            {
                "request": request,
                "selected_dataset": selected_dataset,
                "selected_query": selected_query,
                "task_name": task_name,
                "task_description": task_description,
                "usable": result["usable"],
                "summary": f"The dataset is usable for {task_name}.",
                "dataset_info": result["dataset_info"],
                "preprocess_info": result["preprocess_info"],
                "metric_text": result["metric_text"],
                "details": result["details"],
                "message": result.get("message", ""),
                "plot_images": plot_images,
                "model_architecture": result.get("architecture", model_architecture),
                "hidden_layers": result.get("hidden_layers", hidden_layers),
                "neurons": result.get("neurons", neurons or "none"),
                "activation": activation,
                "experiment_name": experiment_name,
                "comparison_results": comparison_results,
                "sklearn_comparison": result.get("sklearn_comparison", []),
                "predictions": result.get("predictions", []),
                "correct_count": result.get("correct_count", 0),
                "incorrect_count": result.get("incorrect_count", 0),
                "accuracy_percent": result.get("accuracy_percent", 0),
                "network_visualization": result.get("network_visualization", None),
            },
        )
    except Exception as exc:
        message = str(exc)
        return templates.TemplateResponse(
            "model_result.html",
            {
                "request": request,
                "selected_dataset": selected_dataset,
                "selected_query": selected_query,
                "task_name": task_name,
                "task_description": task_description,
                "usable": False,
                "summary": "",
                "dataset_info": "",
                "preprocess_info": "",
                "metric_text": "",
                "details": "",
                "message": message,
                "model_architecture": model_architecture,
                "hidden_layers": hidden_layers,
                "neurons": neurons or "none",
                "activation": activation,
                "experiment_name": experiment_name,
                "comparison_results": [],
                "sklearn_comparison": [],
                "predictions": [],
                "correct_count": 0,
                "incorrect_count": 0,
                "accuracy_percent": 0,
                "network_visualization": None,
            },
        )
