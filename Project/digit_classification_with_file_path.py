import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

print('\nLoading MNIST Data...')
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], -1) / 255.0
X_test = X_test.reshape(X_test.shape[0], -1) / 255.0

from keras.api.utils import to_categorical

X_train_reshaped = X_train.reshape(-1, 28, 28, 1)
X_test_reshaped = X_test.reshape(-1, 28, 28, 1)
y_train_oh = to_categorical(y_train, 10)
y_test_oh = to_categorical(y_test, 10)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def calculate_metrics(model, X_test, y_test, model_type):
    if model_type == 'ann':
        y_pred = model.predict(X_test).argmax(axis=1)
    else:
        y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred)

    print(f"\nMetrics for {model_type.upper()}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)

    plt.matshow(conf_matrix, cmap='coolwarm')
    plt.title(f'{model_type.upper()} Confusion Matrix')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    return accuracy, precision, recall, f1, conf_matrix

def train_model(model_type):
    if model_type == 'random_forest':
        from sklearn.ensemble import RandomForestClassifier
        print('\nTraining Random Forest Classifier with n_estimators=300 and max_depth=30...')
        model = RandomForestClassifier(n_estimators=300, max_depth=30, n_jobs=-1, random_state=42)
        model.fit(X_train, y_train)
        calculate_metrics(model, X_test, y_test, model_type)

    elif model_type == 'knn':
        from sklearn.neighbors import KNeighborsClassifier
        print('\nTraining KNN Classifier with n_neighbors=5...')
        model = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
        model.fit(X_train, y_train)
        calculate_metrics(model, X_test, y_test, model_type)

    elif model_type == 'ann':
        from keras.api.models import Sequential
        from keras.api.layers import Dense, Flatten
        print('\nTraining ANN Model...')
        model = Sequential([
            Flatten(input_shape=(28, 28, 1)),
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(X_train_reshaped, y_train_oh, epochs=10, batch_size=128, verbose=1)
        calculate_metrics(model, X_test_reshaped, y_test, model_type)
        
    else:
        print('Invalid model type selected. Exiting...')
        sys.exit()
    return model

print("\nSelect the model to train:")
print("1. Random Forest")
print("2. KNN")
print("3. ANN")

model_choice = input("Enter your choice (1/2/3): ").strip()

if model_choice == '1':
    model_type = 'random_forest'
elif model_choice == '2':
    model_type = 'knn'
elif model_choice == '3':
    model_type = 'ann'
else:
    print("Invalid choice. Defaulting to Random Forest.")
    model_type = 'random_forest'

model = train_model(model_type)

if model_type == 'random_forest':
    import pickle
    with open('rf_model.pkl', 'wb') as f:
        pickle.dump(model, f)
elif model_type == 'knn':
    import pickle
    with open('knn_model.pkl', 'wb') as f:
        pickle.dump(model, f)
elif model_type == 'ann':
    model.save('ann_model.h5')

from PIL import Image

def preprocess_image(image_path):
    try:
        print(f"\nPreprocessing image: {image_path}")
        img = Image.open(image_path).convert('L')
        img = img.resize((28, 28))
        img_array = np.array(img)
        img_array = img_array / 255.0
        img_array = img_array.flatten().reshape(1, -1) if model_type != 'ann' else img_array.reshape(1, 28, 28, 1)
        return img_array, img
    except Exception as e:
        print(f"Error processing image: {e}")
        return None, None

def upload_and_predict():
    while True:
        print("\nEnter the full path of the image (png, jpg, jpeg) or type 'exit' to quit:")
        image_path = input("Image Path: ").strip()

        if image_path.lower() == 'exit':
            print("Exiting the prediction loop. Goodbye!")
            break

        processed_img_array, processed_img = preprocess_image(image_path)

        if processed_img_array is not None:
            if model_type == 'ann':
                probabilities = model.predict(processed_img_array)[0]
                predicted_digit = np.argmax(probabilities)
            else:
                probabilities = model.predict_proba(processed_img_array)[0]
                predicted_digit = model.predict(processed_img_array)[0]

            formatted_probabilities = [f"{prob:.4f}" for prob in probabilities]
            print(f"\nThe predicted digit is: {predicted_digit}")
            print(f"Prediction probabilities for each digit: {formatted_probabilities}")

            plt.imshow(processed_img, cmap='gray')
            plt.title(f"Predicted Digit: {predicted_digit}")
            plt.axis('off')
            plt.show()

            plt.bar(range(10), probabilities)
            plt.title("Prediction Probabilities")
            plt.xlabel("Digit")
            plt.ylabel("Probability")
            plt.xticks(range(10))
            plt.show()
        else:
            print("Failed to process the image. Please check the file path and format.")

def visualize_samples():
    print("\nVisualizing some random test samples...")
    indices = np.random.randint(0, len(X_test), 10)
    for i in indices:
        img = X_test[i].reshape(28, 28) * 255.0
        if model_type == 'ann':
            probabilities = model.predict(X_test_reshaped[i].reshape(1, 28, 28, 1))[0]
            predicted = np.argmax(probabilities)
        else:
            probabilities = model.predict_proba(X_test[i].reshape(1, -1))[0]
            predicted = model.predict(X_test[i].reshape(1, -1))[0]

        formatted_probabilities = [f"{prob:.4f}" for prob in probabilities]
        print(f"Actual: {y_test[i]} | Predicted: {predicted} | Probabilities: {formatted_probabilities}")

        plt.title(f"Actual: {y_test[i]} | Predicted: {predicted}")
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        plt.show()

upload_and_predict()
visualize_samples()
