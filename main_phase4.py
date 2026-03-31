
"""
Main Script - Phase 4 (OPTIMIZED + DEBUG VERSION)
================================================
CGNN training with debug logs, GPU support, and stability fixes
"""

import sys
from pathlib import Path
import time

# Add src to path
sys.path.append(str(Path(__file__).parent))

import torch
from src.neural_network import (
    CGNNDataLoader,
    CGNN,
    CGNNTrainer,
    CGNNEvaluator,
    set_seed,
    count_parameters,
    get_device,
    create_directories,
    load_config,
    print_model_summary
)

def main():
    """Main execution function"""

    start_time = time.time()

    print("\n" + "="*70)
    print("🚀 CGNN - PHASE 4: MODEL TRAINING (OPTIMIZED)")
    print("="*70 + "\n")

    # ========================================
    # CONFIG + SETUP
    # ========================================
    print("⚙️ Loading configuration...")
    config = load_config('configs/phase4_config.yaml')

    print("🎲 Setting random seed...")
    set_seed(config['data']['random_seed'])

    print("🖥️ Checking device...")
    device = get_device(config)
    print(f"🔥 Using device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("📁 Creating directories...")
    create_directories(config)

    # ========================================
    # STEP 1: Load and prepare data
    # ========================================
    print("\n" + "="*70)
    print("STEP 1: DATA LOADING")
    print("="*70)

    step_start = time.time()

    print("⏳ Initializing data loader...")
    data_loader = CGNNDataLoader(config)

    print("⏳ Calling prepare_data()... (this may take time)")
    train_data, val_data, test_data, feature_names = data_loader.prepare_data()

    print(f"✅ Data loaded in {time.time() - step_start:.2f} sec")

    # Save scaler
    scaler_path = Path(config['output']['results_dir']) / 'scaler.pkl'
    data_loader.save_scaler(scaler_path)
    print(f"📦 Scaler saved to {scaler_path}")

    # ========================================
    # STEP 2: Create model
    # ========================================
    print("\n" + "="*70)
    print("STEP 2: MODEL CREATION")
    print("="*70)

    print("🧠 Initializing model...")
    model = CGNN(config)

    print("📊 Model Summary:")
    print_model_summary(model, train_data)

    print(f"🔢 Total parameters: {count_parameters(model):,}")

    # ========================================
    # STEP 3: Train model
    # ========================================
    print("\n" + "="*70)
    print("STEP 3: MODEL TRAINING")
    print("="*70)

    step_start = time.time()

    trainer = CGNNTrainer(model, config, device)

    print("🏋️ Starting training...")
    best_metrics = trainer.train(train_data, val_data)

    print(f"✅ Training completed in {time.time() - step_start:.2f} sec")

    # Load best model
    print("📥 Loading best model...")
    trainer.load_best_model()

    # ========================================
    # STEP 4: Evaluate model
    # ========================================
    print("\n" + "="*70)
    print("STEP 4: MODEL EVALUATION")
    print("="*70)

    evaluator = CGNNEvaluator(model, config, device)

    print("🔍 Evaluating model...")
    results, y_true, y_pred, y_prob = evaluator.evaluate(test_data)

    print("📊 Results:")
    evaluator.print_results(results)

    # Save results
    results_path = Path(config['output']['results_dir']) / 'test_results.json'
    evaluator.save_results(results, results_path)
    print(f"📁 Results saved to {results_path}")

    # Plots
    cm_path = Path(config['output']['plots_dir']) / 'confusion_matrix.png'
    evaluator.plot_confusion_matrix(results['confusion_matrix'], cm_path)

    dist_path = Path(config['output']['plots_dir']) / 'class_distribution.png'
    evaluator.plot_class_distribution(y_true, y_pred, dist_path)

    print("📈 Plots saved")

    # ========================================
    # STEP 5: Save final model
    # ========================================
    print("\n" + "="*70)
    print("STEP 5: SAVING MODEL")
    print("="*70)

    model_save_path = config['output']['model_save_path']
    Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'feature_names': feature_names,
        'test_results': results
    }, model_save_path)

    print(f"✅ Model saved to {model_save_path}")

    # ========================================
    # DONE
    # ========================================
    total_time = time.time() - start_time

    print("\n" + "="*70)
    print("🎉 PHASE 4 COMPLETE!")
    print("="*70)
    print(f"⏱️ Total Time: {total_time:.2f} sec")
    print(f"🎯 Accuracy: {results['accuracy']:.4f}")
    print(f"📊 F1 Score: {results['f1']:.4f}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
