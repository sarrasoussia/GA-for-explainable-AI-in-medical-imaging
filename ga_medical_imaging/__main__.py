"""
Main entry point for running package scripts.

Usage:
    python -m ga_medical_imaging train [options]
    python -m ga_medical_imaging evaluate [options]
    python -m ga_medical_imaging train_cgan [options]
    python -m ga_medical_imaging evaluate_5fold_cv [options]
    python -m ga_medical_imaging compare_models [options]
    python -m ga_medical_imaging generate_synthetic_dataset [options]
    python -m ga_medical_imaging organize_covid_dataset [options]
"""

import sys
import argparse

def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("="*70)
        print("Geometric Algebra for Representation, Robustness, and Explainability")
        print("="*70)
        print("\nAvailable commands:")
            print("  train                    - Train GA-based classifier (basic)")
            print("  train_professional      - Train GA-based classifier (professional)")
            print("  evaluate                 - Evaluate model and generate explanations")
        print("  train_cgan               - Train conditional GAN for data augmentation")
        print("  evaluate_5fold_cv        - Run 5-fold cross-validation")
        print("  compare_models           - Compare GA model with CNN baseline")
        print("  generate_synthetic_dataset - Generate synthetic images using cGAN")
        print("  organize_covid_dataset   - Organize COVID-19 dataset")
        print("\nExample:")
        print("  python -m ga_medical_imaging train --data_dir data/covid_chestxray")
        print("="*70)
        sys.exit(1)
    
    command = sys.argv[1]
    
    try:
        if command == 'train':
            from .train import main as train_main
            sys.argv = sys.argv[1:]  # Remove 'train' from argv
            train_main()
        elif command == 'train_professional':
            from .train_professional import main as train_professional_main
            sys.argv = sys.argv[1:]  # Remove 'train_professional' from argv
            train_professional_main()
        elif command == 'evaluate':
            from .evaluate_and_explain import main as eval_main
            sys.argv = sys.argv[1:]  # Remove 'evaluate' from argv
            eval_main()
        elif command == 'train_cgan':
            from .train_cgan import main as train_cgan_main
            sys.argv = sys.argv[1:]  # Remove 'train_cgan' from argv
            train_cgan_main()
        elif command == 'evaluate_5fold_cv':
            from .evaluate_5fold_cv import main as cv_main
            sys.argv = sys.argv[1:]  # Remove 'evaluate_5fold_cv' from argv
            cv_main()
        elif command == 'compare_models':
            from .compare_models import main as compare_main
            sys.argv = sys.argv[1:]  # Remove 'compare_models' from argv
            compare_main()
        elif command == 'generate_synthetic_dataset':
            from .generate_synthetic_dataset import main as gen_main
            sys.argv = sys.argv[1:]  # Remove 'generate_synthetic_dataset' from argv
            gen_main()
        elif command == 'organize_covid_dataset':
            from .organize_covid_dataset import main as org_main
            sys.argv = sys.argv[1:]  # Remove 'organize_covid_dataset' from argv
            org_main()
        else:
            print(f"❌ Unknown command: {command}")
            print("\nAvailable commands:")
            print("  train, train_professional, evaluate, train_cgan, evaluate_5fold_cv,")
            print("  compare_models, generate_synthetic_dataset, organize_covid_dataset")
            sys.exit(1)
    except ImportError as e:
        print(f"❌ Error importing module for command '{command}': {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error running command '{command}': {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()

