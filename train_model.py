import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import numpy as np

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32")  / 255.0

x_train = x_train.reshape(-1, 28, 28, 1)
x_test  = x_test.reshape(-1, 28, 28, 1)

print("Treino:", x_train.shape, y_train.shape)
print("Teste: ", x_test.shape,  y_test.shape)


model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(8, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(16, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax"),
], name="mnist_edge_cnn")

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()


history = model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.1,
    verbose=1,
)

loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
per_class = [(y_pred[y_test == c] == c).mean() for c in range(10)]

print("\n" + "=" * 55)
print("         RESULTADO FINAL DO MODELO")
print("=" * 55)
print(f"  Acurácia no teste  : {accuracy * 100:.2f}%")
print(f"  Loss no teste      : {loss:.4f}")
print("-" * 55)
print("  Acurácia por dígito:")
for digit, acc in enumerate(per_class):
    bar = "█" * int(acc * 20)
    print(f"    [{digit}] {bar:<20} {acc * 100:.1f}%")
print("=" * 55)

total_params = model.count_params()
print(f"\n  Parâmetros totais  : {total_params:,}")
print(f"  Modelo adequado para Edge AI: sim" if total_params < 50_000
      else f"  Atenção: {total_params:,} parâmetros podem ser pesados para Edge AI")
print()


model.save("model.h5")
print("Modelo salvo em: model.h5")