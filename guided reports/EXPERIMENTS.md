# Plan d'Expérimentation et Évaluation

Ce document décrit le plan d'expérimentation pour valider les contributions de ce travail de recherche.

## 1. Objectifs Expérimentaux

### 1.1 Objectifs Principaux

1. **Validation de Performance** : Démontrer que le modèle GA atteint des performances comparables aux méthodes de l'état de l'art
2. **Validation d'Explicabilité** : Prouver que les explications GA sont plus interprétables que les méthodes post-hoc
3. **Analyse des Composantes** : Quantifier la contribution de chaque composante géométrique
4. **Validation Clinique** : Obtenir des retours d'experts sur l'utilité des explications

### 1.2 Hypothèses de Recherche

- **H1** : La représentation multivecteur capture efficacement les caractéristiques géométriques importantes pour la classification
- **H2** : Les explications basées sur les composantes GA sont plus interprétables que les méthodes de salience standard
- **H3** : Différentes pathologies se caractérisent par des patterns différents de contributions géométriques

## 2. Datasets et Préparation

### 2.1 Datasets Recommandés

#### Option 1 : Datasets Publics
- **ISIC** (International Skin Imaging Collaboration) : Classification de lésions cutanées
- **Chest X-Ray** : Pneumonie vs normal
- **Brain Tumor MRI** : Tumeurs cérébrales
- **Retinal Fundus Images** : Détection de rétinopathie

#### Option 2 : Dataset Propriétaire
- Collaboration avec un hôpital ou centre de recherche
- Validation éthique nécessaire
- Annotations d'experts requises

### 2.2 Préparation des Données

```python
# Structure recommandée
data/
├── train/
│   ├── sain/
│   └── tumeur/
├── val/
│   ├── sain/
│   └── tumeur/
└── test/
    ├── sain/
    └── tumeur/
```

## 3. Expériences Planifiées

### 3.1 Expérience 1 : Performance de Classification

**Objectif** : Comparer les performances du modèle GA avec des baselines

**Métriques** :
- Accuracy (Précision globale)
- Sensitivity (Rappel pour la classe positive)
- Specificity (Rappel pour la classe négative)
- F1-Score
- AUC-ROC
- AUC-PR

**Baselines à comparer** :
- ResNet standard
- EfficientNet
- Vision Transformer
- Modèles avec attention

**Script** :
```bash
python experiments/compare_baselines.py \
    --data_dir data \
    --models ga resnet efficientnet \
    --num_epochs 50
```

### 3.2 Expérience 2 : Analyse d'Explicabilité

**Objectif** : Comparer la qualité des explications

**Métriques qualitatives** :
- Cohérence avec les annotations d'experts
- Utilité perçue par les cliniciens
- Clarté des visualisations

**Métriques quantitatives** :
- Intersection avec les régions annotées (IoU)
- Fidélité des explications (perturbation)
- Stabilité des explications

**Méthodes de comparaison** :
- Grad-CAM
- LIME
- SHAP
- Attention maps

**Script** :
```bash
python experiments/compare_explanations.py \
    --model_path checkpoints/best_model.pth \
    --test_images data/test \
    --annotations data/annotations \
    --methods ga gradcam lime shap
```

### 3.3 Expérience 3 : Analyse des Composantes Géométriques

**Objectif** : Comprendre quelles composantes sont importantes pour quels types d'images

**Analyses** :
- Distribution des contributions par composante
- Corrélation entre composantes et types de pathologies
- Analyse statistique des patterns

**Script** :
```bash
python experiments/analyze_components.py \
    --model_path checkpoints/best_model.pth \
    --dataset data/test \
    --output_dir results/component_analysis
```

### 3.4 Expérience 4 : Ablation Study

**Objectif** : Comprendre l'importance de chaque composante du modèle

**Variantes à tester** :
- Sans scalaires (uniquement vecteurs, bivecteurs, trivecteurs)
- Sans vecteurs
- Sans bivecteurs
- Sans trivecteurs
- Architecture standard (sans GA)

**Script** :
```bash
python experiments/ablation_study.py \
    --data_dir data \
    --variants all no_scalars no_vectors no_bivectors no_trivector baseline
```

### 3.5 Expérience 5 : Validation Clinique

**Objectif** : Obtenir des retours d'experts médicaux

**Questionnaire pour experts** :
1. Les explications sont-elles compréhensibles ?
2. Les régions identifiées correspondent-elles à votre analyse ?
3. Les composantes géométriques sont-elles pertinentes cliniquement ?
4. Utiliseriez-vous ce système en pratique ?

**Script** :
```bash
python experiments/clinical_validation.py \
    --model_path checkpoints/best_model.pth \
    --test_images data/test \
    --output_dir results/clinical_validation \
    --survey_mode interactive
```

## 4. Métriques et Évaluations

### 4.1 Métriques de Performance

```python
# Exemple de calcul des métriques
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def evaluate_model(model, dataloader, device):
    all_preds = []
    all_labels = []
    all_probs = []
    
    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds),
        'recall': recall_score(all_labels, all_preds),
        'f1': f1_score(all_labels, all_preds),
        'auc_roc': roc_auc_score(all_labels, all_probs)
    }
    
    return metrics
```

### 4.2 Métriques d'Explicabilité

```python
# Intersection avec annotations d'experts
def compute_explanation_iou(explanation_map, expert_annotation):
    intersection = np.logical_and(explanation_map > threshold, expert_annotation)
    union = np.logical_or(explanation_map > threshold, expert_annotation)
    iou = np.sum(intersection) / np.sum(union)
    return iou

# Fidélité (perturbation)
def compute_fidelity(model, image, explanation, num_perturbations=100):
    # Masquer les régions importantes et mesurer la baisse de confiance
    # Plus la baisse est grande, plus l'explication est fidèle
    pass
```

## 5. Résultats Attendus

### 5.1 Tableaux de Résultats

**Tableau 1 : Performance de Classification**

| Modèle | Accuracy | Sensitivity | Specificity | F1-Score | AUC-ROC |
|--------|----------|-------------|------------|----------|---------|
| ResNet-50 | ... | ... | ... | ... | ... |
| EfficientNet | ... | ... | ... | ... | ... |
| **GA Model** | ... | ... | ... | ... | ... |

**Tableau 2 : Contribution des Composantes**

| Type d'Image | Scalaires | Vecteurs | Bivecteurs | Trivecteurs |
|--------------|-----------|----------|------------|-------------|
| Sain | ... | ... | ... | ... |
| Tumeur | ... | ... | ... | ... |

### 5.2 Visualisations

- Courbes ROC comparatives
- Graphiques de contribution des composantes
- Exemples d'explications pour différents cas
- Cartes d'importance spatiale

## 6. Analyse Statistique

### 6.1 Tests Statistiques

- Tests de significativité pour comparer les performances
- Analyse de variance (ANOVA) pour les contributions des composantes
- Corrélations entre métriques

### 6.2 Représentativité

- Validation croisée k-fold
- Tests sur plusieurs datasets
- Analyse de la généralisation

## 7. Scripts d'Expérimentation

Créer un dossier `experiments/` avec :

```
experiments/
├── compare_baselines.py
├── compare_explanations.py
├── analyze_components.py
├── ablation_study.py
├── clinical_validation.py
└── utils/
    ├── metrics.py
    └── visualization.py
```

## 8. Calendrier d'Expérimentation

1. **Semaine 1-2** : Préparation des données et baselines
2. **Semaine 3-4** : Entraînement des modèles
3. **Semaine 5-6** : Expériences d'explicabilité
4. **Semaine 7-8** : Analyse des composantes et ablation
5. **Semaine 9-10** : Validation clinique
6. **Semaine 11-12** : Analyse des résultats et rédaction

## 9. Documentation des Résultats

Pour chaque expérience, documenter :
- Configuration exacte (hyperparamètres, seed, etc.)
- Résultats bruts (métriques, visualisations)
- Analyse et interprétation
- Limitations observées

---

**Note** : Ce plan doit être adapté selon vos ressources, datasets disponibles, et contraintes temporelles.

