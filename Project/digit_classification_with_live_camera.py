import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn import metrics
from keras.api.models import Sequential
from keras.api.layers import Dense, Input
from scikeras.wrappers import KerasClassifier
from matplotlib import pyplot as plt
from PIL import Image
import joblib
import cv2

mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(-1, 28 * 28).astype('float32') / 255.0
X_test = X_test.reshape(-1, 28 * 28).astype('float32') / 255.0

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train).astype('float32')
X_test_scaled = scaler.transform(X_test).astype('float32')
joblib.dump(scaler, 'scaler.pkl')
print("Scaler saved as 'scaler.pkl'")

def build_ann():
    model = Sequential([
        Input(shape=(784,)),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

ann = build_ann()
ann.fit(X_train_scaled, y_train, batch_size=32, epochs=5, verbose=1)

rf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
rf.fit(X_train, y_train)

class CustomKerasClassifier(KerasClassifier):
    def predict_proba(self, X):
        proba = super().predict_proba(X)
        if isinstance(proba, list):
            proba = np.array(proba)
        if proba.ndim == 1:
            proba = proba.reshape(1, -1)
        return proba

ann_wrapped = CustomKerasClassifier(model=build_ann, epochs=5, batch_size=32, verbose=0)
ann_wrapped._estimator_type = "classifier"
ann_wrapped.fit(X_train_scaled, y_train)

ensemble_model = VotingClassifier(
    estimators=[('ann', ann_wrapped), ('rf', rf)],
    voting='soft'
)
ensemble_model.fit(X_train_scaled, y_train)
joblib.dump(ensemble_model, 'ensemble_handwritten_digit_model.pkl')
print("Ensemble Model saved as 'ensemble_handwritten_digit_model.pkl'")

def calculate_specificity(cm):
    tn = cm.sum() - cm.sum(axis=1) - cm.sum(axis=0) + np.diag(cm)
    fp = cm.sum(axis=0) - np.diag(cm)
    specificity = tn / (tn + fp)
    return specificity.mean()

models = {
    "ANN": (ann_wrapped, X_test_scaled),
    "Random Forest": (rf, X_test),
    "Ensemble": (ensemble_model, X_test_scaled)
}

for model_name, (model, X) in models.items():
    y_pred = model.predict(X)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred, average='weighted')
    recall = metrics.recall_score(y_test, y_pred, average='weighted')
    f1 = metrics.f1_score(y_test, y_pred, average='weighted')
    cm = metrics.confusion_matrix(y_test, y_pred)
    specificity = calculate_specificity(cm)
    auc = metrics.roc_auc_score(y_test, model.predict_proba(X), multi_class='ovr')

    print(f"\n{model_name} Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")

def preprocess_image(img, invert_colors=False):
    gray = img.convert('L')
    gray_np = np.array(gray)

    if invert_colors:
        gray_np = 255 - gray_np

    _, thresholded = cv2.threshold(gray_np, 127, 255, cv2.THRESH_BINARY)

    resized = cv2.resize(thresholded, (28, 28), interpolation=cv2.INTER_AREA)

    normalized = resized / 255.0

    flattened = normalized.flatten().reshape(1, -1)
    return flattened, gray_np, resized, normalized

def predict_digit():
    print("Choose an input method:")
    print("1. Upload Image")
    print("2. Live Camera")
    choice = input("Enter your choice: ").strip()

    if choice == '1':
        file_path = input("Enter the full path of the image file: ").strip()
        img = Image.open(file_path)
        inverted_choice = input("Is the digit black on white background? (y/n): ").strip().lower()
        invert_colors = True if inverted_choice == 'y' else False
        img_flattened, gray, resized, normalized = preprocess_image(img, invert_colors)

        plt.figure(figsize=(12, 4))
        plt.suptitle("Preprocessing Steps")
        plt.subplot(1, 3, 1)
        plt.title("Grayscale Image")
        plt.imshow(gray, cmap='gray')
        plt.subplot(1, 3, 2)
        plt.title("Resized Image")
        plt.imshow(resized, cmap='gray')
        plt.subplot(1, 3, 3)
        plt.title("Normalized Image")
        plt.imshow(normalized, cmap='gray')
        plt.show()

        img_scaled = scaler.transform(img_flattened)
        prediction = ensemble_model.predict(img_scaled)
        ann_prob = ann_wrapped.predict_proba(img_scaled)
        rf_prob = rf.predict_proba(img_flattened)

        print(f"Predicted Digit: {prediction[0]}")

        digits = range(10)
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.bar(digits, ann_prob[0], color='blue')
        plt.title("ANN Probabilities")
        plt.xlabel("Digit")
        plt.ylabel("Probability")

        plt.subplot(1, 2, 2)
        plt.bar(digits, rf_prob[0], color='green')
        plt.title("Random Forest Probabilities")
        plt.xlabel("Digit")
        plt.ylabel("Probability")
        plt.show()

    elif choice == '2':
        cap = cv2.VideoCapture(0)
        print("Press 'p' to capture and classify a digit, and 'q' to quit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            cv2.imshow("Camera Feed", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('p'):
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, (28, 28))
                inverted_choice = input("Is the digit black on white background? (y/n): ").strip().lower()
                invert_colors = True if inverted_choice == 'y' else False
                if invert_colors:
                    resized = 255 - resized
                normalized = resized / 255.0
                flattened = normalized.flatten().reshape(1, -1)

                plt.figure(figsize=(12, 4))
                plt.suptitle("Live Camera Preprocessing")
                plt.subplot(1, 3, 1)
                plt.title("Grayscale Image")
                plt.imshow(gray, cmap='gray')
                plt.subplot(1, 3, 2)
                plt.title("Resized Image")
                plt.imshow(resized, cmap='gray')
                plt.subplot(1, 3, 3)
                plt.title("Normalized Image")
                plt.imshow(normalized, cmap='gray')
                plt.show()

                img_scaled = scaler.transform(flattened)
                prediction = ensemble_model.predict(img_scaled)
                print(f"Predicted Digit: {prediction[0]}")

            elif key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

predict_digit()
