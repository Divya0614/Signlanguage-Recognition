import cv2
import os
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

def load_data(data_dir, img_size=64):
    X, y = [], []
    classes = sorted(os.listdir(data_dir))
    for label in classes:
        folder = os.path.join(data_dir, label)
        if not os.path.isdir(folder):
            continue
        for img_name in os.listdir(folder):
            img_path = os.path.join(folder, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img = cv2.resize(img, (img_size, img_size))
            X.append(img)
            y.append(label)
    return np.array(X), np.array(y)

# Load data
train_path = "sign_language/train"
test_path = "sign_language/test"
X_train, y_train = load_data(train_path)
X_test, y_test = load_data(test_path)
print("Classes detected in training set:", sorted(set(y_train)))


X_train = X_train / 255.0
X_test = X_test / 255.0

# Encode labels
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_train = to_categorical(y_train_encoded)

# Filter out test labels that were not seen during training
X_test_filtered = []
y_test_filtered = []
for x, label in zip(X_test, y_test):
    if label in le.classes_:
        X_test_filtered.append(x)
        y_test_filtered.append(label)

X_test = np.array(X_test_filtered)
y_test_encoded = le.transform(y_test_filtered)
y_test = to_categorical(y_test_encoded)

num_classes = y_train.shape[1]

# Define the model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.25),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
model.save("sign_language_model.h5")

# Plot accuracy
plt.plot(history.history['accuracy'], label="Train Acc")
plt.plot(history.history['val_accuracy'], label="Val Acc")
plt.legend()
plt.title("Model Accuracy")
plt.savefig("accuracy_plot.png")

# Webcam prediction
model = load_model("sign_language_model.h5")
labels = le.classes_
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    roi = cv2.resize(frame, (64, 64))
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    roi = roi / 255.0
    roi = np.expand_dims(roi, axis=0)
    pred = model.predict(roi)
    class_idx = np.argmax(pred)
    label = labels[class_idx]
    cv2.putText(frame, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 3)
    cv2.imshow("Sign Prediction", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
