import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Reshape, Dense, Dropout, Bidirectional, LSTM, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Set paths and parameters
csv_path = 'E:/E2CR/output/labels.csv'
image_dir = 'E:/E2CR/output/'
mask_dir = 'E:/E2CR/output/'
target_height = 32
target_width = 128
batch_size = 32
epochs = 10


# Load CSV
data = pd.read_csv(csv_path)

# Print column names and first few rows to verify
print("Column Names:", data.columns)
print(data.head())

# Preprocessing function
def preprocess_image(image_path, mask_path, target_height, target_width):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    masked_image = cv2.bitwise_and(image, image, mask=mask)

    # Get the current dimensions of the image
    h, w = masked_image.shape

    # Calculate padding
    scale = min(target_width / w, target_height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized_image = cv2.resize(masked_image, (new_w, new_h))

    # Pad the resized image to the target size
    padded_image = np.full((target_height, target_width), 255, dtype=np.uint8)
    padded_image[(target_height - new_h) // 2 : (target_height - new_h) // 2 + new_h,
                 (target_width - new_w) // 2 : (target_width - new_w) // 2 + new_w] = resized_image

    # Normalize the image
    normalized_image = padded_image / 255.0

    return normalized_image

# Preprocess all images and labels
images = []
labels = []

for index, row in data.iterrows():
    print(f"Processing row {index}: {row}")  # Debug print
    image_path = os.path.join(image_dir, row['Image'].strip())
    mask_path = os.path.join(mask_dir, row['Mask'].strip())
    if not os.path.exists(image_path) or not os.path.exists(mask_path):
        print(f"File not found: {image_path} or {mask_path}")
        continue
    processed_image = preprocess_image(image_path, mask_path, target_height, target_width)
    images.append(processed_image)
    labels.append(row['Label'].strip())

images = np.array(images).reshape(-1, target_height, target_width, 1)
labels = np.array(labels)

# Encode labels
label_encoder = LabelEncoder()
label_encoder.fit(list(''.join(labels)))
encoded_labels = [label_encoder.transform(list(label)) for label in labels]

# Padding encoded labels
max_label_length = max([len(label) for label in encoded_labels])
padded_labels = np.zeros((len(labels), max_label_length), dtype=np.int32)
for i, label in enumerate(encoded_labels):
    padded_labels[i, :len(label)] = label

# Define input lengths and label lengths
def get_output_length(input_length, pool_sizes):
    length = input_length
    for pool_size in pool_sizes:
        length = (length + pool_size - 1) // pool_size
    return length

# Define the CRNN model
def build_crnn(input_shape, num_classes):
    inputs = Input(name='input', shape=input_shape, dtype='float32')
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Reshape(target_shape=(get_output_length(input_shape[1], [2, 2]), (input_shape[0] // 4) * 64))(x)
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Dropout(0.25)(x)
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs, outputs)
    return model

# Build the model
num_classes = len(label_encoder.classes_) + 1  # Include a character for blank label in CTC
model = build_crnn((target_height, target_width, 1), num_classes)

# Calculate the output length for the CTC loss input
model_output_length = get_output_length(target_width, [2, 2])
input_length = np.full((len(labels),), model_output_length, dtype=np.int32)
label_length = np.array([len(label) for label in encoded_labels], dtype=np.int32)

# Split data into training and validation sets
X_train, X_val, y_train, y_val, train_input_length, val_input_length, train_label_length, val_label_length = train_test_split(
    images, padded_labels, input_length, label_length, test_size=0.2, random_state=42)

# Summary of the model
model.summary()

# Custom training step with CTC loss
@tf.functionrun
def train_step(images, labels, input_length, label_length):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        labels = tf.sparse.from_dense(tf.convert_to_tensor(labels, dtype=tf.int32))
        predictions = tf.convert_to_tensor(predictions, dtype=tf.float32)
        input_length = tf.convert_to_tensor(input_length, dtype=tf.int32)
        label_length = tf.convert_to_tensor(label_length, dtype=tf.int32)

        loss = tf.nn.ctc_loss(
            labels=labels,
            logits=predictions,
            label_length=label_length,
            logit_length=input_length,
            blank_index=num_classes - 1,
            logits_time_major=False
        )
        loss = tf.reduce_mean(loss)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, predictions

# Custom validation step
@tf.function
def validate_step(images, labels, input_length, label_length):
    predictions = model(images, training=False)
    labels = tf.sparse.from_dense(tf.convert_to_tensor(labels, dtype=tf.int32))
    loss = tf.nn.ctc_loss(
        labels=labels,
        logits=predictions,
        label_length=label_length,
        logit_length=input_length,
        blank_index=num_classes - 1,
        logits_time_major=False
    )
    loss = tf.reduce_mean(loss)
    return loss, predictions

# Decode predictions to compute accuracy
def decode_predictions(predictions, input_length):
    decoded, _ = tf.keras.backend.ctc_decode(predictions, input_length, greedy=True)
    decoded_dense = tf.sparse.to_dense(decoded[0], default_value=-1).numpy()
    return decoded_dense

# Calculate accuracy
def calculate_accuracy(decoded_preds, labels):
    accuracies = []
    for i in range(len(labels)):
        pred = decoded_preds[i]
        label = labels[i]
        pred = pred[pred != -1]  # Remove padding
        pred_length = min(len(pred), len(label))
        if pred_length == 0:
            accuracies.append(0)
            continue
        pred = pred[:pred_length]
        label = label[:pred_length]
        accuracy = np.mean(pred == label)
        accuracies.append(accuracy)
    return np.mean(accuracies)

# Training loop
optimizer = Adam(learning_rate=0.001)
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    
    # Training
    train_loss = 0.0
    train_accuracy = 0.0
    for i in range(0, len(X_train), batch_size):
        batch_images = X_train[i:i + batch_size]
        batch_labels = y_train[i:i + batch_size]
        batch_input_length = train_input_length[i:i + batch_size]
        batch_label_length = train_label_length[i:i + batch_size]

        loss, predictions = train_step(batch_images, batch_labels, batch_input_length, batch_label_length)
        train_loss += loss.numpy()
        
        decoded_preds = decode_predictions(predictions, batch_input_length)
        print(f"Batch {i//batch_size + 1}: Decoded preds length: {len(decoded_preds)}, Batch labels length: {len(batch_labels)}")  # Debug print
        train_accuracy += calculate_accuracy(decoded_preds, batch_labels)

    train_loss /= (len(X_train) // batch_size)
    train_accuracy /= (len(X_train) // batch_size)
    
    # Validation
    val_loss = 0.0
    val_accuracy = 0.0
    for i in range(0, len(X_val), batch_size):
        batch_images = X_val[i:i + batch_size]
        batch_labels = y_val[i:i + batch_size]
        batch_input_length = val_input_length[i:i + batch_size]
        batch_label_length = val_label_length[i:i + batch_size]

        loss, predictions = validate_step(batch_images, batch_labels, batch_input_length, batch_label_length)
        val_loss += loss.numpy()
        
        decoded_preds = decode_predictions(predictions, batch_input_length)
        print(f"Batch {i//batch_size + 1}: Decoded preds length: {len(decoded_preds)}, Batch labels length: {len(batch_labels)}")  # Debug print
        val_accuracy += calculate_accuracy(decoded_preds, batch_labels)

    val_loss /= (len(X_val) // batch_size)
    val_accuracy /= (len(X_val) // batch_size)
    
    print(f"Epoch {epoch + 1}: Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

# Save the model
model.save('crnn_model.keras')

# Example usage:
# image_path = 'path/to/your/image.jpg'
# mask_path = 'path/to/your/mask.png'
# processed_image = preprocess_image(image_path, mask_path, target_height, target_width)
# prediction = model.predict(np.array([processed_image]).reshape(-1, target_height, target_width, 1))
# decoded_prediction = label_encoder.inverse_transform(np.argmax(prediction, axis=-1).flatten())
# print(''.join(decoded_prediction))

