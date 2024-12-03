import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

# Example data: Replace these with your actual labels and predictions
y_true = [0, 1, 2, 2, 0, 1, 2, 0, 1, 2]  # Ground truth labels
y_pred = [0, 0, 2, 2, 0, 1, 2, 0, 2, 2]  # Predicted labels by the model

# 1. Calculate the Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred)

# 2. Display Confusion Matrix
print("Confusion Matrix:")
print(conf_matrix)

# 3. Classification Metrics
accuracy = accuracy_score(y_true, y_pred)
# Weighted for multiclass
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

print("\nPerformance Metrics:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")

# Detailed Classification Report
print("\nClassification Report:")
print(classification_report(y_true, y_pred,
      target_names=["Class 0", "Class 1", "Class 2"]))

# 4. Visualize the Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Class 0", "Class 1", "Class 2"],
            yticklabels=["Class 0", "Class 1", "Class 2"])
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix Heatmap")
plt.show()
