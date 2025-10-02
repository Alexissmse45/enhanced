import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers, models

# ---------------------------
# Spatial Transformer Network (STN)
# ---------------------------
class SpatialTransformer(layers.Layer):
    def __init__(self, input_shape):
        super(SpatialTransformer, self).__init__()
        self.localization = models.Sequential([
            layers.Conv2D(16, (7,7), activation="relu", padding="same", input_shape=input_shape),
            layers.MaxPooling2D(2,2),
            layers.Conv2D(32, (5,5), activation="relu", padding="same"),
            layers.MaxPooling2D(2,2),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            # 6 params for affine transform (2x3 matrix)
            layers.Dense(6, activation=None,
                         kernel_initializer=tf.zeros_initializer(),
                         bias_initializer=tf.constant_initializer([1,0,0,0,1,0]))
        ])

    def call(self, x):
        # Predict transformation parameters
        theta = self.localization(x)  # (batch, 6)

        # Pad to 8 parameters expected by tfa.image.transform
        # [a0, a1, a2, a3, a4, a5, a6, a7] (last row [0,0,1] is implicit)
        batch_size = tf.shape(x)[0]
        theta = tf.concat([theta, tf.zeros((batch_size, 2))], axis=1)  # (batch, 8)

        # Apply transform
        x_transformed = tfa.image.transform(x, theta, interpolation="BILINEAR")
        return x_transformed

# ---------------------------
# Hierarchical Attention
# ---------------------------
class HierarchicalAttention(layers.Layer):
    def __init__(self, units=32):
        super(HierarchicalAttention, self).__init__()
        self.W = layers.Dense(units, activation='tanh')
        self.V = layers.Dense(1)

    def call(self, features):
        # features shape: (batch, time, feature_dim)
        score = self.V(self.W(features))      # (batch, time, 1)
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)  # weighted sum
        return context_vector

# ---------------------------
# Enhanced CNN with STN + Attention
# ---------------------------
def build_cnn_model(input_shape=(128,128,3), num_classes=8):
    inputs = layers.Input(shape=input_shape)

    # 1. Spatial Transformer Network
    x = SpatialTransformer(input_shape)(inputs)

    # 2. Convolutions with BatchNorm
    x = layers.Conv2D(32, (3,3), activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)

    x = layers.Conv2D(64, (3,3), activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)

    x = layers.Conv2D(128, (3,3), activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)

    # 3. Prepare features for Attention
    x = layers.Reshape((-1, x.shape[-1]))(x)

    # 4. Apply Hierarchical Attention
    x = HierarchicalAttention(units=64)(x)

    # 5. Dense Layers
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)

    # ✅ AdamW for better generalization
    optimizer = tfa.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-5)

    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model
