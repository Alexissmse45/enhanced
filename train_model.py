import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model.cnn_model import build_cnn_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# -------------------------------
# Training parameters
# -------------------------------
IMG_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 15
DATASET_PATH = "datasets"

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Dataset folder '{DATASET_PATH}' not found!")

# -------------------------------
# Step 1: Data augmentation
# -------------------------------
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)

# -------------------------------
# Step 2: Training generator (fixed)
# -------------------------------
train_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="sparse",
    subset="training",  # ✅ use training split (not validation)
    shuffle=True,
)

# -------------------------------
# Step 3: Validation generator
# -------------------------------
val_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="sparse",
    subset="validation",
    shuffle=False,
)

# -------------------------------
# Step 4: Get detected classes
# -------------------------------
num_classes = len(train_gen.class_indices)
print("Detected classes:", train_gen.class_indices)

# Save categories for prediction script
with open("categories.txt", "w") as f:
    for label in train_gen.class_indices.keys():
        f.write(label + "\n")

# -------------------------------
# Step 5: Compute class weights (handle imbalance)
# -------------------------------
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_gen.classes),
    y=train_gen.classes,
)
class_weights = dict(enumerate(class_weights))
print("Class Weights:", class_weights)

# -------------------------------
# Step 6: Build enhanced CNN
# -------------------------------
model = build_cnn_model(input_shape=(128, 128, 3), num_classes=num_classes)

# -------------------------------
# Step 7: Train model with validation and class weights
# -------------------------------

callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, verbose=1)
]

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    class_weight=class_weights,  # ✅ prevent class dominance
)

# -------------------------------
# Step 8: Save trained weights
# -------------------------------
model.save("first_aid_cnn_model.h5")
print("✅ Enhanced CNN model saved.")

# -------------------------------
# Step 9: Plot accuracy and loss
# -------------------------------
def plot_training_history(history, save_path="training_accuracy_loss.png"):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Training Accuracy", marker="o")
    plt.plot(epochs_range, val_acc, label="Validation Accuracy", marker="o")
    plt.legend(loc="lower right")
    plt.title("Training & Validation Accuracy")

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Training Loss", marker="o")
    plt.plot(epochs_range, val_loss, label="Validation Loss", marker="o")
    plt.legend(loc="upper right")
    plt.title("Training & Validation Loss")

    plt.tight_layout()
    plt.savefig(save_path)   # ✅ Auto-save the graph
    plt.close()

# -------------------------------
# Step 10: Save training graph
# -------------------------------
plot_training_history(history, save_path="training_accuracy_loss.png")
print("📊 Training graph saved as 'training_accuracy_loss.png'")
