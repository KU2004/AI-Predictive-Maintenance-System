from src.data_preprocessing import load_data, preprocess_data
from src.model import train_model, predict_failure
from src.evaluate import evaluate_model
from src.visualize import plot_results

def main():
    print("🔄 Loading and processing data...")

    # Load dataset
    data = load_data("data/train_FD001.txt")

    print("✅ Data loaded successfully")

    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(data)

    print("🚀 Training model...")

    # Train model
    model = train_model(X_train, y_train)

    print("📊 Evaluating model...")

    # Evaluate model
    evaluate_model(model, X_test, y_test)

    print("🔮 Making predictions...")

    # Predict
    predictions = predict_failure(model, X_test)

    print("📈 Generating visualization...")

    # Visualize results
    plot_results(y_test, predictions)

    print("✅ Project completed successfully!")
    print("📂 Check 'outputs/' folder for graphs")


if __name__ == "__main__":
    main()