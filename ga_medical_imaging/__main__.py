"""
Point d'entrée principal pour exécuter les scripts du package.
"""

import sys
import argparse

def main():
    """Point d'entrée principal."""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python -m ga_medical_imaging train [options]")
        print("  python -m ga_medical_imaging evaluate [options]")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == 'train':
        from .train import main as train_main
        sys.argv = sys.argv[1:]  # Remove 'train' from argv
        train_main()
    elif command == 'evaluate':
        from .evaluate_and_explain import main as eval_main
        sys.argv = sys.argv[1:]  # Remove 'evaluate' from argv
        eval_main()
    else:
        print(f"Commande inconnue: {command}")
        print("Commandes disponibles: train, evaluate")
        sys.exit(1)

if __name__ == '__main__':
    main()

