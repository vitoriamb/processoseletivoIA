import tensorflow as tf
import os

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("model.h5")
print("Modelo carregado com sucesso!")
model.summary()


converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_dynamic = converter.convert()

with open("model.tflite", "wb") as f:
    f.write(tflite_dynamic)

size_dynamic = len(tflite_dynamic) / 1024

converter2 = tf.lite.TFLiteConverter.from_keras_model(model)
converter2.optimizations = [tf.lite.Optimize.DEFAULT]
converter2.target_spec.supported_types = [tf.float16]
tflite_fp16 = converter2.convert()

with open("model_fp16.tflite", "wb") as f:
    f.write(tflite_fp16)

size_fp16 = len(tflite_fp16) / 1024

print("\n" + "=" * 55)
print("       COMPARAÇÃO DE TÉCNICAS DE OTIMIZAÇÃO")
print("=" * 55)
print(f"  Dynamic Range Quant : {size_dynamic:.2f} KB  → model.tflite")
print(f"  Float16 Quant       : {size_fp16:.2f} KB  → model_fp16.tflite")
print("-" * 55)
print("  Trade-off:")
print("    Dynamic Range → menor tamanho, CPU-friendly")
print("    Float16       → equilíbrio tamanho/precisão, GPU-friendly")
print("=" * 55)