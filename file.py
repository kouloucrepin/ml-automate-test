import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 1. Générer des données simulées
X, y = make_classification(n_samples=500, n_features=10, n_informative=5, n_classes=3, random_state=42)

# 2. Séparer en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Entraîner un modèle
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 4. Prédictions et matrice de confusion
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)

# 5. Afficher et sauvegarder la matrice de confusion
fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(ax=ax)
plt.title("Matrice de confusion")
plt.tight_layout()

# 6. Exporter dans le même dossier
output_path = os.path.join(os.getcwd(), "matrice_confusion.jpg")
plt.savefig(output_path)
plt.close()

print(f"Matrice de confusion enregistrée dans : {output_path}")
