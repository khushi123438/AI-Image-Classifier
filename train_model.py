import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

train_data = datagen.flow_from_directory(
    "dataset",
    target_size=(128,128),
    batch_size=8,
    class_mode="categorical",
    subset="training"
)


val_data = datagen.flow_from_directory(
    "dataset",
    target_size=(128,128),
    batch_size=8,
    class_mode="categorical",
    subset="validation"
)


model = tf.keras.Sequential([

    tf.keras.layers.Input(shape=(128,128,3)),

    tf.keras.layers.Conv2D(32,(3,3),activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64,(3,3),activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128,activation="relu"),

    tf.keras.layers.Dense(
        train_data.num_classes,
        activation="softmax"
    )
])


model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)


model.save("model.h5")

print("Model saved successfully")