import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.applications import InceptionV3  # Import GoogleNet (InceptionV3)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths
train_dir = r"C:\Natan\Data Science\52025 - Adv computational learning and data analysis\Final Project\data\OCT2017\train"  # Update with your path
val_dir = r"C:\Natan\Data Science\52025 - Adv computational learning and data analysis\Final Project\data\OCT2017\test"  # Update with your path
save_dir = r"C:\Natan\Data Science\52025 - Adv computational learning and data analysis\Final Project\result\fine_tuned_googlenet.h5"

# 📌 Step 1: Load Pretrained InceptionV3 Model (Without Top Layers)
base_model = InceptionV3(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# 📌 Step 2: Freeze Base Model Layers
base_model.trainable = False  # Keeps pretrained weights unchanged

# 📌 Step 3: Add Custom Classification Layers
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Prevents too many parameters
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)  # Prevents overfitting
output_layer = Dense(4, activation="softmax")(x)  # Adjust number of classes

# Create new model
model = Model(inputs=base_model.input, outputs=output_layer)

# 📌 Step 4: Compile the Model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# 📌 Step 5: Prepare the Data
train_datagen = ImageDataGenerator(
    rescale=1./255, rotation_range=20, width_shift_range=0.2,
    height_shift_range=0.2, horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_dir, target_size=(224, 224), batch_size=32, class_mode="categorical"
)

val_data = val_datagen.flow_from_directory(
    val_dir, target_size=(224, 224), batch_size=32, class_mode="categorical"
)

# 📌 Step 6: Train the Model and Store History
history = model.fit(train_data, validation_data=val_data, epochs=3)

# 📌 Step 7 (Optional): Fine-Tune GoogleNet by Unfreezing Some Layers
for layer in base_model.layers[-30:]:  # Unfreeze last 30 layers
    layer.trainable = True

# Recompile with a lower learning rate for fine-tuning
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# Train again for fine-tuning and store history
history_finetune = model.fit(train_data, validation_data=val_data, epochs=3)

# 📌 Step 8: Save the Trained Model
model.save(save_dir)

# 📌 Step 9: Plot Training and Validation Accuracy
# Combine the histories from both training and fine-tuning
acc = history.history['accuracy'] + history_finetune.history['accuracy']
val_acc = history.history['val_accuracy'] + history_finetune.history['val_accuracy']

# Plot the accuracy
plt.figure(figsize=(10, 6))
plt.plot(acc, label="Training Accuracy")
plt.plot(val_acc, label="Validation Accuracy")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()
