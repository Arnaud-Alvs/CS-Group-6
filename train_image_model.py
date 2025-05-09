import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# === STEP 1: Point to your folder on Desktop
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
train_dir = os.path.join(desktop_path, "waste_image_dataset")  # <- Make sure folder is named like this

# === STEP 2: Exact folder names (must match your subfolder names inside the dataset)
folder_names = [
    "Aluminium", "Cans", "Cardboard", "Foam packaging", "Glass", "Green waste",
    "Hazardous", "Household", "Metal", "Oil", "Paper", "Plastic", "Textiles"
]

# === STEP 3: Image generators for training and validation
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training',
    classes=folder_names
)

val_gen = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    classes=folder_names
)

# === STEP 4: Build CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(folder_names), activation='softmax')
])

# === STEP 5: Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# === STEP 6: Train the model
model.fit(train_gen, validation_data=val_gen, epochs=10)

# === STEP 7: Save the model
model.save("waste_image_classifier.h5")
print("✅ Training complete! Model saved as waste_image_classifier.h5")
