import argparse
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

IMAGE_SIZE = (128, 128)

def make_datasets(train_dir, test_dir, batch_size):
    train = tf.keras.utils.image_dataset_from_directory(train_dir, image_size=IMAGE_SIZE, batch_size=batch_size, shuffle=True, seed=42, label_mode="int")
    test = tf.keras.utils.image_dataset_from_directory(test_dir, image_size=IMAGE_SIZE, batch_size=batch_size, shuffle=False, label_mode="int", class_names=train.class_names)
    augmentation = tf.keras.Sequential([tf.keras.layers.RandomFlip("horizontal"), tf.keras.layers.RandomRotation(0.05), tf.keras.layers.RandomZoom(0.1), tf.keras.layers.RandomContrast(0.1)], name="augmentation")
    return train, test, train.class_names, augmentation

def build_model(num_classes, augmentation):
    base = tf.keras.applications.VGG16(input_shape=(*IMAGE_SIZE, 3), include_top=False, weights="imagenet")
    base.trainable = False
    for layer in base.layers[-4:]:
        layer.trainable = True
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    x = augmentation(inputs)
    x = tf.keras.applications.vgg16.preprocess_input(x)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--output", default="models/brain_tumor_vgg16.keras")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=20)
    args = parser.parse_args()
    train, test, class_names, augmentation = make_datasets(args.train_dir, args.test_dir, args.batch_size)
    model = build_model(len(class_names), augmentation)
    model.fit(train, validation_data=test, epochs=args.epochs)
    y_true = np.concatenate([y.numpy() for _, y in test])
    y_pred = np.argmax(model.predict(test, verbose=0), axis=1)
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    model.save(output)
    output.with_suffix(".labels.json").write_text(json.dumps(class_names, indent=2))
    print(f"Saved model: {output}\nLabels: {class_names}")

if __name__ == "__main__":
    main()
