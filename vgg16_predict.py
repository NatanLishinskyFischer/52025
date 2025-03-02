import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, f1_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the trained model
model = tf.keras.models.load_model(r"C:\Natan\Data Science\52025 - Adv computational learning and data analysis\Final Project\result\fine_tuned_vgg16.h5")

# Test directory
test_dir = r"C:\Natan\Data Science\52025 - Adv computational learning and data analysis\Final Project\data\OCT2017\test"

# Data Generator for test images
test_datagen = ImageDataGenerator(rescale=1./255)
test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    shuffle=False  # Keep the order for accurate evaluation
)

# Get true labels and predictions
y_true = test_data.classes  # True labels
y_pred_probs = model.predict(test_data)  # Predicted probabilities
y_pred = np.argmax(y_pred_probs, axis=1)  # Convert probabilities to class labels

# Compute F1-score
f1 = f1_score(y_true, y_pred, average="weighted")
print(f"Weighted F1-score: {f1:.4f}")

# Compute ROC curve and AUC for each class
n_classes = len(test_data.class_indices)
fpr = {}
tpr = {}
roc_auc = {}

plt.figure(figsize=(8, 6))
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true == i, y_pred_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    plt.plot(fpr[i], tpr[i], label=f"Class {i} (AUC = {roc_auc[i]:.4f})")

# Plot AUC curve
plt.plot([0, 1], [0, 1], "k--")  # Diagonal line (random classifier)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"AUC-ROC Curve\nWeighted F1-score: {f1:.4f}")  # Add F1-score to title
plt.legend(loc="lower right")
plt.show()
