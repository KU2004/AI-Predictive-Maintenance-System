import matplotlib.pyplot as plt

def plot_results(y_test, predictions):
    plt.figure()
    plt.plot(y_test.values, label="Actual")
    plt.plot(predictions, label="Predicted")
    plt.legend()
    plt.title("Failure Prediction")
    plt.savefig("outputs/result.png")
    plt.show()