import tensorflow as tf

MODEL_PATH = "skin_classifier_fixed.keras"

print("TensorFlow:", tf.__version__)

print("\nLoading model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("\n✅ MODEL LOADED SUCCESSFULLY!")
print("Input shape:", model.input_shape)
print("Output shape:", model.output_shape)

CLASS_NAMES = [
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc"
]

print("\n✅ Classes:")
for i, name in enumerate(CLASS_NAMES):
    print(f"{i}: {name}")

if model.output_shape[-1] == 7:
    print("\n🎉 SUCCESS — 7-class classifier is ready!")
else:
    print("\n❌ ERROR — model does not have 7 outputs.")