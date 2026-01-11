"""
Script to automatically generate a summary of contributions
based on code analysis.

Updated to reflect the complete Geometric Algebra for Explainable AI
in Medical Imaging project, including:
- GA-based classification models
- Intrinsic explainability framework
- Baseline comparison framework
- cGAN-based data augmentation
- 5-fold cross-validation evaluation
"""

import os
import ast
import re
from pathlib import Path
from typing import List, Dict


def analyze_code_contributions(project_root: str = ".") -> Dict:
    """
    Analyze code to identify technical contributions.
    """
    contributions = {
        'classes': [],
        'functions': [],
        'modules': [],
        'innovations': [],
        'evaluation_scripts': [],
        'data_scripts': []
    }
    
    ga_dir = Path(project_root) / "ga_medical_imaging"
    
    if not ga_dir.exists():
        return contributions
    
    # Analyze each Python module
    for py_file in ga_dir.glob("*.py"):
        if py_file.name == "__init__.py" or py_file.name == "__main__.py":
            continue
        
        module_type = "core"
        if "evaluate" in py_file.name or "compare" in py_file.name:
            module_type = "evaluation"
        elif "data" in py_file.name or "organize" in py_file.name or "cgan" in py_file.name:
            module_type = "data"
        elif "train" in py_file.name:
            module_type = "training"
            
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
                tree = ast.parse(content)
                
                # Extract classes
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        docstring = ast.get_docstring(node)
                        class_info = {
                            'name': node.name,
                            'file': py_file.name,
                            'module_type': module_type,
                            'docstring': docstring or "No docstring"
                        }
                        contributions['classes'].append(class_info)
                        
                        # Identify key innovations
                        if 'GA' in node.name or 'Geometric' in node.name:
                            contributions['innovations'].append({
                                'type': 'GA_Component',
                                'name': node.name,
                                'file': py_file.name
                            })
                    
                    # Extract important functions
                    if isinstance(node, ast.FunctionDef):
                        if not node.name.startswith('_'):
                            docstring = ast.get_docstring(node)
                            func_info = {
                                'name': node.name,
                                'file': py_file.name,
                                'module_type': module_type,
                                'docstring': docstring or "No docstring"
                            }
                            contributions['functions'].append(func_info)
                            
                            # Track evaluation and data functions
                            if module_type == "evaluation":
                                contributions['evaluation_scripts'].append({
                                    'function': node.name,
                                    'file': py_file.name
                                })
                            elif module_type == "data":
                                contributions['data_scripts'].append({
                                    'function': node.name,
                                    'file': py_file.name
                                })
        except Exception as e:
            print(f"Warning: Could not parse {py_file.name}: {e}")
            continue
    
    return contributions


def generate_summary_markdown(contributions: Dict, output_path: str = "CONTRIBUTION_SUMMARY.md"):
    """
    Generate markdown summary of contributions.
    """
    md_content = """# Automatic Contribution Summary

This document is automatically generated from code analysis.

**Project**: Geometric Algebra for Explainable AI in Medical Imaging  
**Application**: COVID-19 Detection from Chest X-ray Images  
**Date**: Generated automatically

## Overview

This project implements a comprehensive Geometric Algebra-based framework for explainable medical image classification, with specific application to COVID-19 detection.

### Key Features

1. **Geometric Algebra Representation**: Multivector-based image encoding
2. **Intrinsic Explainability**: Geometric component analysis
3. **Baseline Comparison**: Framework for comparing with state-of-the-art methods
4. **cGAN Augmentation**: Conditional GAN for class balancing
5. **5-Fold Cross-Validation**: Comprehensive evaluation protocol

## Core Classes

"""
    
    # Group classes by module type and file
    classes_by_type = {
        'core': [],
        'evaluation': [],
        'data': [],
        'training': []
    }
    
    for cls in contributions['classes']:
        module_type = cls.get('module_type', 'core')
        if module_type in classes_by_type:
            classes_by_type[module_type].append(cls)
    
    # Core GA Components
    md_content += "\n### Core GA Components\n\n"
    for cls in classes_by_type['core']:
        if 'GA' in cls['name'] or 'Geometric' in cls['name']:
            md_content += f"#### `{cls['name']}` ({cls['file']})\n\n"
            md_content += f"{cls['docstring']}\n\n"
    
    # Evaluation Components
    if classes_by_type['evaluation']:
        md_content += "\n### Evaluation & Comparison\n\n"
        for cls in classes_by_type['evaluation']:
            md_content += f"#### `{cls['name']}` ({cls['file']})\n\n"
            md_content += f"{cls['docstring']}\n\n"
    
    # Data & Augmentation Components
    if classes_by_type['data']:
        md_content += "\n### Data Processing & Augmentation\n\n"
        for cls in classes_by_type['data']:
            md_content += f"#### `{cls['name']}` ({cls['file']})\n\n"
            md_content += f"{cls['docstring']}\n\n"
    
    md_content += "\n## Key Functions\n\n"
    
    # Group functions by importance and module type
    key_functions = {
        'GA_Representation': [],
        'Model_Training': [],
        'Evaluation': [],
        'Explainability': [],
        'Data_Processing': []
    }
    
    for func in contributions['functions']:
        file = func['file']
        name = func['name']
        
        # Categorize functions
        if 'multivector' in name.lower() or 'ga' in name.lower() or 'geometric' in name.lower():
            key_functions['GA_Representation'].append(func)
        elif 'train' in name.lower() or 'fit' in name.lower():
            key_functions['Model_Training'].append(func)
        elif 'evaluate' in name.lower() or 'metric' in name.lower() or 'compare' in name.lower():
            key_functions['Evaluation'].append(func)
        elif 'explain' in name.lower() or 'analyze' in name.lower() or 'component' in name.lower():
            key_functions['Explainability'].append(func)
        elif 'load' in name.lower() or 'dataset' in name.lower() or 'organize' in name.lower() or 'generate' in name.lower():
            key_functions['Data_Processing'].append(func)
    
    # Write categorized functions
    for category, funcs in key_functions.items():
        if funcs:
            md_content += f"\n### {category.replace('_', ' ')}\n\n"
            for func in funcs[:15]:  # Limit to 15 per category
                desc = func['docstring'].split('.')[0] if func['docstring'] else 'No description'
                md_content += f"- `{func['name']}` ({func['file']}): {desc}\n"
    
    md_content += """

## Code Metrics

"""
    
    # Count lines of code
    total_lines = 0
    total_files = 0
    code_by_type = {'core': 0, 'evaluation': 0, 'data': 0, 'training': 0}
    
    ga_dir = Path("ga_medical_imaging")
    if ga_dir.exists():
        for py_file in ga_dir.glob("*.py"):
            if py_file.name not in ["__init__.py", "__main__.py"]:
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        lines = len([l for l in f.readlines() if l.strip() and not l.strip().startswith('#')])
                        total_lines += lines
                        total_files += 1
                        
                        # Categorize by file name
                        if "evaluate" in py_file.name or "compare" in py_file.name:
                            code_by_type['evaluation'] += lines
                        elif "data" in py_file.name or "organize" in py_file.name or "cgan" in py_file.name:
                            code_by_type['data'] += lines
                        elif "train" in py_file.name:
                            code_by_type['training'] += lines
                        else:
                            code_by_type['core'] += lines
                except:
                    pass
    
    md_content += f"""
- **Total Python files**: {total_files}
- **Total lines of code (approx.)**: {total_lines}
- **Core classes**: {len([c for c in contributions['classes'] if c.get('module_type') == 'core'])}
- **Public functions**: {len(contributions['functions'])}

### Code Distribution by Category
- **Core GA components**: {code_by_type['core']} lines
- **Evaluation & comparison**: {code_by_type['evaluation']} lines
- **Data processing & augmentation**: {code_by_type['data']} lines
- **Training scripts**: {code_by_type['training']} lines

## Contribution Structure

### 1. Geometric Algebra Representation
- `GeometricAlgebraRepresentation`: Converts images → multivectors (8 components)
- `GAMultivectorLayer`: Neural layers operating on multivectors
- `GAFeatureExtractor`: Extracts geometric features via GA layers

### 2. Classification Models
- `GAMedicalClassifier`: Main GA-based classifier for medical images
- `GAMedicalClassifierWithAttention`: Variant with spatial attention mechanism

### 3. Intrinsic Explainability
- `GAExplainabilityAnalyzer`: Analyzer for geometric component contributions
- `analyze_geometric_components`: Quantifies contribution of each geometric grade
- `visualize_explanations`: Generates visual explanations
- `generate_explanation_report`: Creates textual explanation reports

### 4. Evaluation Framework
- `evaluate_5fold_cv`: 5-fold cross-validation matching baseline protocols
- `compare_with_baselines`: Comparison with DarkCovidNet, VGG-19, VGG16+cGAN
- `compare_models`: Direct comparison between GA model and traditional CNN
- `calculate_metrics_comprehensive`: Computes all evaluation metrics (accuracy, sensitivity, specificity, precision, F1, ROC AUC)

### 5. Data Processing & Augmentation
- `organize_covid_dataset`: Organizes COVID-19 dataset for binary classification
- `ConditionalGAN`: cGAN for generating synthetic COVID-19 images
- `generate_synthetic_dataset`: Creates balanced dataset using cGAN
- `load_dataset_from_directory`: Loads medical image datasets with flexible naming

### 6. Training & Pipeline Scripts
- `train.py`: Main training script with TensorBoard logging
- `train_cgan.py`: Trains conditional GAN for data augmentation
- `run_baseline_comparison.py`: Complete evaluation and comparison pipeline
- `run_cgan_pipeline.py`: Complete cGAN augmentation and evaluation pipeline

## Key Innovations

### 1. Multivector Representation
**Innovation**: Explicit geometric structure encoding (scalars, vectors, bivectors, trivectors)  
**Code**: `ga_representation.py::GeometricAlgebraRepresentation`

### 2. Intrinsic Explainability
**Innovation**: Explanations based on model structure, not post-hoc approximations  
**Code**: `explainability.py::GAExplainabilityAnalyzer`

### 3. Baseline Comparison Framework
**Innovation**: Comprehensive comparison with state-of-the-art methods using same evaluation protocol  
**Code**: `compare_with_baselines.py`, `evaluate_5fold_cv.py`

### 4. cGAN-Based Augmentation
**Innovation**: Conditional GAN for class balancing (matching Electronics 2022 approach)  
**Code**: `cgan_generator.py::ConditionalGAN`

### 5. End-to-End Explainable Pipeline
**Innovation**: Maintains interpretability at every stage from image to explanation  
**Code**: `model.py::GAMedicalClassifier`

## Application Domain

**Primary Application**: COVID-19 detection from chest X-ray images  
**Dataset**: ieee8023/covid-chestxray-dataset  
**Task**: Binary classification (COVID-19 vs no-findings)  
**Evaluation**: 5-fold Cross-Validation

## Baseline Comparisons

The framework includes comparison with:
1. **DarkCovidNet**: Accuracy ≈ 98%, Sensitivity ≈ 95%, Specificity ≈ 95%, Precision ≈ 98%, F1 ≈ 0.97
2. **VGG-19**: Accuracy ≈ 98.75%
3. **VGG16 + cGAN**: Accuracy ≈ 99.76% (Electronics 2022)

---

*Automatically generated - See CONTRIBUTIONS.md for complete details*
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    print(f"✓ Summary generated in {output_path}")


def main():
    """Main function."""
    print("Analyzing code to identify contributions...")
    contributions = analyze_code_contributions()
    
    print(f"Found {len(contributions['classes'])} classes")
    print(f"Found {len(contributions['functions'])} functions")
    print(f"Found {len(contributions['innovations'])} GA innovations")
    print(f"Found {len(contributions['evaluation_scripts'])} evaluation functions")
    print(f"Found {len(contributions['data_scripts'])} data processing functions")
    
    print("\nGenerating summary...")
    generate_summary_markdown(contributions)
    
    print("\n✓ Analysis complete!")
    print(f"✓ Summary saved to CONTRIBUTION_SUMMARY.md")


if __name__ == '__main__':
    main()

