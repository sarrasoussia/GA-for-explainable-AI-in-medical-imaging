"""
Script pour générer automatiquement un résumé des contributions
basé sur l'analyse du code.
"""

import os
import ast
import re
from pathlib import Path
from typing import List, Dict


def analyze_code_contributions(project_root: str = ".") -> Dict:
    """
    Analyse le code pour identifier les contributions techniques.
    """
    contributions = {
        'classes': [],
        'functions': [],
        'modules': [],
        'innovations': []
    }
    
    ga_dir = Path(project_root) / "ga_medical_imaging"
    
    if not ga_dir.exists():
        return contributions
    
    # Analyser chaque module Python
    for py_file in ga_dir.glob("*.py"):
        if py_file.name == "__init__.py":
            continue
            
        with open(py_file, 'r', encoding='utf-8') as f:
            content = f.read()
            tree = ast.parse(content)
            
            # Extraire les classes
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    docstring = ast.get_docstring(node)
                    contributions['classes'].append({
                        'name': node.name,
                        'file': py_file.name,
                        'docstring': docstring or "No docstring"
                    })
                
                # Extraire les fonctions importantes
                if isinstance(node, ast.FunctionDef):
                    if not node.name.startswith('_'):
                        docstring = ast.get_docstring(node)
                        contributions['functions'].append({
                            'name': node.name,
                            'file': py_file.name,
                            'docstring': docstring or "No docstring"
                        })
    
    return contributions


def generate_summary_markdown(contributions: Dict, output_path: str = "CONTRIBUTION_SUMMARY.md"):
    """
    Génère un résumé markdown des contributions.
    """
    md_content = """# Résumé Automatique des Contributions

Ce document est généré automatiquement à partir de l'analyse du code.

## Classes Principales

"""
    
    # Grouper les classes par fichier
    classes_by_file = {}
    for cls in contributions['classes']:
        file = cls['file']
        if file not in classes_by_file:
            classes_by_file[file] = []
        classes_by_file[file].append(cls)
    
    for file, classes in classes_by_file.items():
        md_content += f"\n### {file}\n\n"
        for cls in classes:
            md_content += f"#### `{cls['name']}`\n\n"
            md_content += f"{cls['docstring']}\n\n"
    
    md_content += "\n## Fonctions Clés\n\n"
    
    # Fonctions importantes par fichier
    functions_by_file = {}
    for func in contributions['functions']:
        file = func['file']
        if file not in functions_by_file:
            functions_by_file[file] = []
        functions_by_file[file].append(func)
    
    for file, functions in functions_by_file.items():
        md_content += f"\n### {file}\n\n"
        for func in functions[:10]:  # Limiter à 10 par fichier
            md_content += f"- `{func['name']}`: {func['docstring'].split('.')[0] if func['docstring'] else 'No description'}\n"
    
    md_content += """

## Métriques du Code

"""
    
    # Compter les lignes de code
    total_lines = 0
    total_files = 0
    
    ga_dir = Path("ga_medical_imaging")
    if ga_dir.exists():
        for py_file in ga_dir.glob("*.py"):
            if py_file.name != "__init__.py":
                with open(py_file, 'r', encoding='utf-8') as f:
                    lines = len([l for l in f.readlines() if l.strip() and not l.strip().startswith('#')])
                    total_lines += lines
                    total_files += 1
    
    md_content += f"""
- **Nombre de fichiers Python**: {total_files}
- **Lignes de code (approximatif)**: {total_lines}
- **Classes principales**: {len(contributions['classes'])}
- **Fonctions publiques**: {len(contributions['functions'])}

## Structure des Contributions

### 1. Représentation GA
- `GeometricAlgebraRepresentation`: Conversion images → multivecteurs
- `GAMultivectorLayer`: Couches neuronales sur multivecteurs

### 2. Modèles
- `GAMedicalClassifier`: Classificateur principal
- `GAMedicalClassifierWithAttention`: Variante avec attention

### 3. Explicabilité
- `GAExplainabilityAnalyzer`: Analyseur d'explicabilité
- Méthodes d'analyse des composantes géométriques
- Génération de rapports et visualisations

### 4. Utilitaires
- Chargement et préprocessing de données médicales
- Scripts d'entraînement et d'évaluation
- Génération de datasets synthétiques

---

*Généré automatiquement - Voir CONTRIBUTIONS.md pour les détails complets*
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    print(f"Résumé généré dans {output_path}")


def main():
    """Fonction principale."""
    print("Analyse du code pour identifier les contributions...")
    contributions = analyze_code_contributions()
    
    print(f"Trouvé {len(contributions['classes'])} classes")
    print(f"Trouvé {len(contributions['functions'])} fonctions")
    
    print("\nGénération du résumé...")
    generate_summary_markdown(contributions)
    
    print("\nAnalyse terminée!")


if __name__ == '__main__':
    main()

