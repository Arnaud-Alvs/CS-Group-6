# This script trains a ML image recognition model 
# Using Tensorflow and Keras, it classifies and recognises 13 different waste types
# We first import the necessary libraries 
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator # used to handle the training of the model
from tensorflow.keras.models import Sequential # used to handle the training of the model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense # used to handle the training of the model

# 1. Points to the desktop to search for the folder
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
train_dir = os.path.join(desktop_path, "waste_image_dataset")  # 1. points to this exact folder on the desktop

# 2. points to the exact subfoler names inside the folder waste_image_dataset
folder_names = [
    "Aluminium", "Cans", "Cardboard", "Foam packaging", "Glass", "Green waste",
    "Hazardous", "Household", "Metal", "Oil", "Paper", "Plastic", "Textiles"
]

# 3. creation of a tool datagen that will help us rescale the images and split them into training and validation sets
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# 3. loads the images from the directors and resizes them, then categorizes them into the classes defined in folder_names and only takes the training subset
train_gen = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training',
    classes=folder_names
)

# 3. loads the images from the directors and resizes them, then categorizes them into the classes defined in folder_names and only takes the validation subset
val_gen = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    classes=folder_names
)

# 4. creation of a sequential model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(folder_names), activation='softmax')
])

# 5. compiles the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 6. trains the model
model.fit(train_gen, validation_data=val_gen, epochs=10)

# 7. saves the model
model.save("waste_image_classifier.h5")
print("✅ Training complete! Model saved as waste_image_classifier.h5")

# With support from ChatGPT (OpenAI), consulted for debugging and resolving initial implementation errors - Arnaud Butty
