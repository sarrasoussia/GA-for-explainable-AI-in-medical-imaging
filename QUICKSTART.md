# Guide de Démarrage Rapide

## Installation Rapide

```bash
# 1. Installer les dépendances
pip install -r requirements.txt

# 2. Tester avec un dataset factice
python example_usage.py
```

## Utilisation Basique

### 1. Entraîner un modèle

```bash
# Avec dataset factice (pour tester)
python -m ga_medical_imaging train --num_epochs 20

# Avec vos propres données
python -m ga_medical_imaging train \
    --data_dir data \
    --num_epochs 50 \
    --batch_size 16 \
    --learning_rate 0.001
```

### 2. Évaluer et expliquer

```bash
python -m ga_medical_imaging evaluate_and_explain \
    --checkpoint checkpoints/best_model.pth \
    --image path/to/your/image.png \
    --output_dir explanations
```

## Structure des Données

Pour utiliser vos propres images, organisez-les ainsi :

```
data/
├── sain/
│   ├── image1.png
│   ├── image2.png
│   └── ...
└── tumeur/
    ├── image1.png
    ├── image2.png
    └── ...
```

## Exemple de Code Python

```python
import torch
from ga_medical_imaging.model import GAMedicalClassifier
from ga_medical_imaging.explainability import GAExplainabilityAnalyzer
from torchvision import transforms
from PIL import Image

# Charger le modèle
model = GAMedicalClassifier(num_classes=2, device='cpu')
model.load_state_dict(torch.load('checkpoints/best_model.pth'))
model.eval()

# Charger une image
image = Image.open('path/to/image.png').convert('L')
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
image_tensor = transform(image).unsqueeze(0)

# Prédiction
with torch.no_grad():
    logits = model(image_tensor)
    probs = torch.softmax(logits, dim=1)
    pred = torch.argmax(probs, dim=1).item()

print(f"Prédiction: {'Tumeur' if pred == 1 else 'Sain'}")
print(f"Confiance: {probs[0, pred].item():.3f}")

# Analyse d'explicabilité
analyzer = GAExplainabilityAnalyzer(model, device='cpu')
analysis = analyzer.analyze_geometric_components(image_tensor)
report = analyzer.generate_explanation_report(image_tensor)
print(report)
```

## Commandes Utiles

```bash
# Voir toutes les options d'entraînement
python -m ga_medical_imaging.train --help

# Voir toutes les options d'évaluation
python -m ga_medical_imaging.evaluate_and_explain --help

# Utiliser GPU si disponible
python -m ga_medical_imaging train --device cuda

# Ajuster la taille des images
python -m ga_medical_imaging train --image_size 256 256
```

## Résultats

Après l'entraînement, vous trouverez :
- `checkpoints/best_model.pth` : Meilleur modèle
- `checkpoints/logs/` : Logs TensorBoard (si activé)

Après l'évaluation :
- `explanations/explanation_*.png` : Visualisations des explications

## Problèmes Courants

### Erreur: "No module named 'ga_medical_imaging'"
Solution: Assurez-vous d'être dans le répertoire racine du projet.

### Erreur: "CUDA out of memory"
Solution: Réduisez la taille du batch avec `--batch_size 8` ou `--batch_size 4`.

### Images non chargées
Solution: Vérifiez que les images sont aux formats supportés (PNG, JPG) et que les chemins sont corrects.

