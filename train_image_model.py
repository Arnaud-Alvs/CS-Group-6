import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# === STEP 1: Point to your folder on Desktop
# Automatically finds your Desktop folder
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
train_dir = os.path.join(desktop_path, "waste_image_dataset")  # <- Name of the folder you have

# === STEP 2: These must match your actual folder names exactly
folder_names = [
    "Aluminium", "Cans", "Cardboard", "Foam packaging", "Glass", "Green waste",
    "Hazardous", "Household", "Metal", "Oil", "Paper", "Plastic", "Textiles"
]

# === STEP 3: Corresponding labels for the app (used later in the app logic)
emoji_labels = [
    "Aluminium 🧴", "Cans 🥫", "Cardboard 📦", "Foam packaging ☁", "Glass 🍾", "Green waste 🌿",
    "Hazardous waste ⚠", "Household waste 🗑", "Metal 🪙", "Oil 🛢", "Paper 📄", "Plastic", "Textiles 👕"
]

# === STEP 4: Load and preprocess images
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

# === STEP 5: Define your CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(folder_names), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# === STEP 6: Train the model
model.fit(train_gen, validation_data=val_gen, epochs=10)

# === STEP 7: Save the model
model.save("waste_image_classifier.h5")
print("✅ Training complete! Model saved as waste_image_classifier.h5")
