# GA pour l'IA Explicable en Imagerie Médicale

Ce projet implémente un système d'intelligence artificielle basé sur l'**Algèbre Géométrique (Geometric Algebra)** pour la classification et l'explication de décisions en imagerie médicale. Le système permet non seulement de classer les images (tissu sain vs tumeur), mais aussi d'identifier quelles composantes géométriques influencent le diagnostic, offrant ainsi une couche d'interprétabilité très recherchée.

## 🎯 Objectifs

- **Classification**: Distinguer les tissus sains des tumeurs dans les images médicales
- **Explicabilité**: Identifier quelles composantes géométriques (scalaires, vecteurs, bivecteurs, trivecteurs) influencent les décisions
- **Interprétabilité**: Fournir des visualisations et rapports détaillés sur les décisions du modèle

## 📚 Concepts de l'Algèbre Géométrique

Le système utilise l'algèbre de Clifford pour représenter les images médicales comme des **multivecteurs** :

- **Scalaires (Grade 0)**: Intensités de pixels
- **Vecteurs (Grade 1)**: Gradients spatiaux (dx, dy)
- **Bivecteurs (Grade 2)**: Orientations et textures
- **Trivecteurs (Grade 3)**: Relations géométriques complexes

Cette représentation permet de capturer des caractéristiques géométriques riches qui sont naturellement interprétables.

## 🏗️ Structure du Projet

```
MastersGA/
├── ga_medical_imaging/
│   ├── __init__.py
│   ├── ga_representation.py      # Conversion images → multivecteurs
│   ├── model.py                    # Modèles de classification GA
│   ├── explainability.py          # Module d'explicabilité
│   ├── data_utils.py              # Utilitaires pour les données
│   ├── train.py                   # Script d'entraînement
│   └── evaluate_and_explain.py   # Évaluation et explications
├── example_usage.py               # Exemples d'utilisation
├── requirements.txt               # Dépendances Python
└── README.md                      # Ce fichier
```

## 🚀 Installation

### Option 1 : Google Colab (Recommandé pour débuter)

Le moyen le plus simple de tester le projet est d'utiliser le notebook Colab :

1. Ouvrez **[GA_Medical_Imaging_Colab.ipynb](GA_Medical_Imaging_Colab.ipynb)** sur [Google Colab](https://colab.research.google.com/)
2. Exécutez les cellules dans l'ordre
3. Le notebook contient tout le code nécessaire (version simplifiée)

Voir **[COLAB_SETUP.md](COLAB_SETUP.md)** pour plus de détails.

### Option 2 : Installation Locale

#### Prérequis

- Python 3.8+
- PyTorch 2.0+
- CUDA (optionnel, pour GPU)

#### Installation des dépendances

```bash
pip install -r requirements.txt
```

## 💻 Utilisation

### 1. Entraînement du modèle

#### Avec vos propres données

Organisez vos images dans la structure suivante :
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

Puis lancez l'entraînement :
```bash
python -m ga_medical_imaging.train --data_dir data --num_epochs 50 --batch_size 16
```

#### Avec un dataset factice (pour tester)

Le système peut créer automatiquement un dataset factice :
```bash
python -m ga_medical_imaging.train --num_epochs 20
```

### 2. Évaluation et génération d'explications

```bash
python -m ga_medical_imaging.evaluate_and_explain \
    --checkpoint checkpoints/best_model.pth \
    --image path/to/image.png \
    --output_dir explanations
```

### 3. Exemples d'utilisation

Pour voir des exemples complets :
```bash
python example_usage.py
```

## 📊 Fonctionnalités d'Explicabilité

Le module d'explicabilité fournit :

1. **Analyse des composantes géométriques** : Contribution relative de chaque grade (scalaires, vecteurs, bivecteurs, trivecteurs)

2. **Cartes d'importance spatiale** : Visualisation des régions les plus importantes pour la décision

3. **Rapports textuels** : Explications détaillées en langage naturel

4. **Visualisations** : Graphiques montrant :
   - L'image originale
   - La carte d'importance spatiale
   - Les contributions des différentes composantes
   - Les visualisations des scalaires, vecteurs et bivecteurs

## 🔬 Architecture du Modèle

### GAMedicalClassifier

Le modèle principal comprend :

1. **GeometricAlgebraRepresentation** : Convertit les images en multivecteurs
2. **GAFeatureExtractor** : Extrait des caractéristiques géométriques via des couches GA
3. **Classifier** : Couches de classification finales

### GAMultivectorLayer

Couche personnalisée qui opère sur les multivecteurs en utilisant le produit géométrique, permettant au modèle d'apprendre des relations géométriques complexes.

## 📈 Métriques et Évaluation

Le système suit :
- **Précision d'entraînement et de validation**
- **Perte d'entraînement et de validation**
- **Contributions des composantes géométriques**
- **Cartes d'attention spatiale**

Les résultats sont sauvegardés dans TensorBoard (si activé) et dans les checkpoints.

## 🎨 Visualisations

Les visualisations générées incluent :

- **Image originale** : L'image médicale d'entrée
- **Carte d'importance** : Régions importantes pour la décision
- **Graphique de contributions** : Barres montrant l'importance de chaque composante
- **Composantes individuelles** : Visualisations des scalaires, vecteurs et bivecteurs
- **Probabilités de prédiction** : Confiance du modèle pour chaque classe

## 🔧 Paramètres Configurables

### Entraînement
- `--num_epochs` : Nombre d'époques (défaut: 50)
- `--batch_size` : Taille du batch (défaut: 16)
- `--learning_rate` : Taux d'apprentissage (défaut: 0.001)
- `--image_size` : Taille des images (défaut: 224 224)

### Modèle
- `multivector_dim` : Dimension des multivecteurs (8 pour GA 3D)
- `feature_dim` : Dimension des caractéristiques extraites (128)
- `num_classes` : Nombre de classes (2 pour binaire)

## 📝 Exemple de Rapport d'Explication

```
=== RAPPORT D'EXPLICATION - DIAGNOSTIC MÉDICAL ===

PRÉDICTION:
  Classe prédite: Tumeur
  Confiance: 87.3%
  
CONTRIBUTION DES COMPOSANTES GÉOMÉTRIQUES:

1. Scalaires (Intensités de pixels):
   Contribution: 25.3%
   
2. Vecteurs (Gradients spatiaux):
   Contribution: 30.1%
   
3. Bivecteurs (Orientations et textures):
   Contribution: 35.2%
   
4. Trivecteur (Relations complexes):
   Contribution: 9.4%

ANALYSE:
La composante la plus influente est les orientations et textures 
(35.2% de la contribution totale).
```

## 🧪 Tests et Validation

Pour tester le système avec des données factices :

```python
from ga_medical_imaging.data_utils import create_dummy_dataset
from ga_medical_imaging.model import GAMedicalClassifier

# Créer un dataset factice
image_paths, labels = create_dummy_dataset(num_samples=100)

# Créer et tester le modèle
model = GAMedicalClassifier(num_classes=2, device='cpu')
# ... entraînement et évaluation
```

## 🔬 Contributions de Recherche

Ce projet présente plusieurs contributions originales dans le domaine de l'IA explicable en imagerie médicale :

### Contributions Principales

1. **Représentation Multivecteur pour Images Médicales** : Développement d'un schéma de conversion d'images médicales en représentations multivecteurs GA qui capture explicitement différentes dimensions géométriques (scalaires, vecteurs, bivecteurs, trivecteurs).

2. **Couches Neuronales sur Multivecteurs** : Implémentation de couches spécialisées (`GAMultivectorLayer`) opérant directement sur les multivecteurs avec produits géométriques adaptés.

3. **Explicabilité Intrinsèque** : Système d'explication basé sur les composantes géométriques, fournissant des explications structurelles plutôt que post-hoc.

4. **Analyse de Contribution des Composantes** : Méthode pour quantifier la contribution relative de chaque grade géométrique dans les décisions de classification.

5. **Architecture End-to-End Explicable** : Conception d'une architecture complète qui maintient l'interprétabilité à chaque étape du pipeline.

Pour plus de détails sur les contributions, voir **[CONTRIBUTIONS.md](CONTRIBUTIONS.md)**.

Pour le plan d'expérimentation, voir **[EXPERIMENTS.md](EXPERIMENTS.md)**.

## 📚 Références

Ce projet s'inspire de :
- Geometric Algebra pour la représentation des données
- Explainable AI (XAI) pour l'interprétabilité
- Medical Image Analysis pour l'application

## 🤝 Contribution

Pour contribuer au projet :
1. Fork le repository
2. Créez une branche pour votre fonctionnalité
3. Committez vos changements
4. Poussez vers la branche
5. Ouvrez une Pull Request

## 📄 Licence

Ce projet est destiné à des fins de recherche et d'éducation.

## 📖 Documentation de Recherche

Ce projet fait partie d'un travail de recherche de master. La documentation complète inclut :

- **[RESEARCH_REPORT.md](RESEARCH_REPORT.md)** : **Rapport de recherche complet** (méthodologie, contributions, cadre expérimental)
- **[CONTRIBUTIONS.md](CONTRIBUTIONS.md)** : Contributions détaillées de ce travail
- **[CONTRIBUTIONS_SUMMARY.md](CONTRIBUTIONS_SUMMARY.md)** : Résumé concis des contributions
- **[EXPERIMENTS.md](EXPERIMENTS.md)** : Plan d'expérimentation et évaluation
- **[RESEARCH_PAPER_OUTLINE.md](RESEARCH_PAPER_OUTLINE.md)** : Plan de rédaction du mémoire
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** : Guide de référence rapide

Pour générer un résumé automatique des contributions :
```bash
python scripts/generate_contribution_summary.py
```

## 🔮 Améliorations Futures

- [ ] Support pour images 3D (volumes médicaux)
- [ ] Intégration avec d'autres architectures (Transformers GA)
- [ ] Métriques d'explicabilité quantitatives
- [ ] Interface web pour la visualisation interactive
- [ ] Support pour multi-classes (plusieurs types de tumeurs)

## 📧 Contact

Pour toute question ou suggestion, n'hésitez pas à ouvrir une issue.

---

**Note**: Ce système est conçu pour la recherche et l'éducation. Pour une utilisation clinique réelle, des validations supplémentaires et des certifications appropriées sont nécessaires.

# GA-for-explainable-AI-in-medical-imaging
