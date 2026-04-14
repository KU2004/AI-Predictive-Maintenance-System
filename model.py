from sklearn.ensemble import RandomForestClassifier
import joblib

def train_model(X_train, y_train):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    joblib.dump(model, "models/model.pkl")
    return model

def predict_failure(model, X_test):
    return model.predict(X_test)