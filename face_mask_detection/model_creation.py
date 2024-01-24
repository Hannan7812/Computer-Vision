import tensorflow as tf


def main():
    # Create the model
    model = create_model()

    # Train the model
    history = train_model(model)

    # Save the model
    save_model(model)
    print(history)


def create_model():
    # Create the model
    model = tf.keras.models.Sequential([
        # Note the input shape is the desired size of the image 224x224 with 3 bytes color
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(244, 244, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        # Flatten the results to feed into a DNN
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),

        # 3 output neurons for 3 classes with the softmax activation
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model):
    # Create an ImageDataGenerator and take images from the dataset folder
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)

    # Flow training images in batches of 32 using train_datagen generator
    train_generator = train_datagen.flow_from_directory(
        'dataset',
        target_size=(244, 244),
        batch_size=32,
        class_mode='sparse'
    )
    history = model.fit_generator(train_generator, epochs=4, verbose=1)
    return history 

def save_model(model):
    # Save the model
    model.save('model.h5') 

if __name__ == '__main__':
    main()