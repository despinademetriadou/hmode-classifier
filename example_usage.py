#!/usr/bin/env python3
"""
Example Usage of H-mode Classifier
=================================

This script demonstrates various ways to use the H-mode classification model.
"""

from hmode_classifier import HModeClassifier
import numpy as np
import matplotlib.pyplot as plt


def basic_usage_example():
    """Basic usage: train model and make predictions."""
    print("=== BASIC USAGE EXAMPLE ===")
    
    # Initialize classifier
    classifier = HModeClassifier('OMFIT_Hmode_Studies.pkl')
    
    # Load data and build dataset
    classifier.load_data()
    dataset = classifier.build_dataset()
    
    # Train model
    results = classifier.train_final_model(dataset['features'], dataset['labels'])
    
    # Make a prediction
    shot_ids = list(classifier.DD_full.keys())
    example_shot = shot_ids[0]
    prediction = classifier.predict(classifier.DD_full, example_shot, 0)
    
    if prediction:
        print(f"\nPrediction for shot {example_shot}:")
        print(f"  Mode: {prediction['prediction']}")
        print(f"  Confidence: {prediction['confidence']:.4f}")
        
    return classifier, results


def batch_prediction_example(classifier):
    """Example of making predictions on multiple shots."""
    print("\n=== BATCH PREDICTION EXAMPLE ===")
    
    shot_ids = list(classifier.DD_full.keys())[:5]  # First 5 shots
    
    predictions = []
    for shot_id in shot_ids:
        # Get number of time slices for this shot
        n_times = len(classifier.DD_full[shot_id]['Times'])
        
        for t_idx in range(min(3, n_times)):  # Max 3 time slices per shot
            pred = classifier.predict(classifier.DD_full, shot_id, t_idx)
            if pred:
                predictions.append({
                    'shot': shot_id,
                    'time_index': t_idx,
                    'prediction': pred['prediction'],
                    'confidence': pred['confidence']
                })
    
    # Display results
    print(f"Batch predictions for {len(predictions)} samples:")
    for pred in predictions[:10]:  # Show first 10
        print(f"  Shot {pred['shot']:>6}, t={pred['time_index']}: "
              f"{pred['prediction']:>7} (conf: {pred['confidence']:.3f})")


def model_analysis_example(classifier, results):
    """Analyze model performance and features."""
    print("\n=== MODEL ANALYSIS EXAMPLE ===")
    
    # Feature importance analysis
    print("Top 10 most important features:")
    importances = classifier.model.feature_importances_
    feature_names = classifier.feature_names
    
    # Sort by importance
    sorted_idx = np.argsort(importances)[::-1]
    
    for i, idx in enumerate(sorted_idx[:10]):
        print(f"  {i+1:2d}. {feature_names[idx]:20s}: {importances[idx]:.4f}")
    
    # Performance metrics
    print(f"\nModel Performance:")
    print(f"  Training accuracy: {results['train_accuracy']:.4f}")
    print(f"  Test accuracy:     {results['test_accuracy']:.4f}")
    print(f"  CV mean ± std:     {results['cv_mean']:.4f} ± {results['cv_std']:.4f}")


def feature_extraction_example(classifier):
    """Show how features are extracted for a single shot."""
    print("\n=== FEATURE EXTRACTION EXAMPLE ===")
    
    shot_ids = list(classifier.DD_full.keys())
    example_shot = shot_ids[0]
    
    # Extract features manually
    features = classifier.compute_enhanced_features(
        classifier.DD_full, example_shot, 0
    )
    
    if features is not None:
        print(f"Features for shot {example_shot}, time index 0:")
        
        feature_names = classifier.get_feature_names()
        
        # Group features by category
        categories = {
            'Core Gradients': slice(0, 3),
            'Pedestal Structure': slice(3, 9), 
            'Profile Shape': slice(9, 15),
            'Curvature': slice(15, 18),
            'Multi-field Coupling': slice(18, 21),
            'OMFIT Comparison': slice(21, 24)
        }
        
        for category, idx_slice in categories.items():
            print(f"\n  {category}:")
            for i in range(idx_slice.start, min(idx_slice.stop, len(features))):
                if i < len(feature_names):
                    print(f"    {feature_names[i]:20s}: {features[i]:8.4f}")


def save_load_example(classifier):
    """Demonstrate model saving and loading."""
    print("\n=== SAVE/LOAD EXAMPLE ===")
    
    # Save the trained model
    classifier.save_model('example_model.pkl')
    
    # Create a new classifier instance and load the model
    new_classifier = HModeClassifier()
    new_classifier.load_model('example_model.pkl')
    
    print("Model successfully saved and loaded!")
    
    # Verify the loaded model works
    shot_ids = list(classifier.DD_full.keys())
    example_shot = shot_ids[0]
    
    # Make predictions with both models
    pred1 = classifier.predict(classifier.DD_full, example_shot, 0)
    
    # Note: new_classifier doesn't have DD_full loaded, so we need to pass the data
    new_classifier.DD_full = classifier.DD_full
    pred2 = new_classifier.predict(new_classifier.DD_full, example_shot, 0)
    
    if pred1 and pred2:
        print(f"Original model prediction: {pred1['prediction']} ({pred1['confidence']:.4f})")
        print(f"Loaded model prediction:   {pred2['prediction']} ({pred2['confidence']:.4f})")
        print("✓ Predictions match!" if pred1['prediction'] == pred2['prediction'] else "❌ Mismatch!")


def plot_performance_example(results):
    """Create performance visualization."""
    print("\n=== VISUALIZATION EXAMPLE ===")
    
    # Confusion matrix visualization
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    
    y_true = results['y_test']
    y_pred = results['y_pred']
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix - H-mode Classification')
    plt.colorbar()
    
    # Add labels
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['L-mode', 'H-mode'])
    plt.yticks(tick_marks, ['L-mode', 'H-mode'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ Confusion matrix saved as 'confusion_matrix.png'")


def main():
    """Run all examples."""
    print("H-MODE CLASSIFIER - EXAMPLE USAGE")
    print("=" * 50)
    
    try:
        # Basic usage
        classifier, results = basic_usage_example()
        
        # Additional examples
        batch_prediction_example(classifier)
        model_analysis_example(classifier, results)
        feature_extraction_example(classifier)
        save_load_example(classifier)
        plot_performance_example(results)
        
        print(f"\n{'='*50}")
        print("✓ All examples completed successfully!")
        print("✓ Model achieves {:.2f}% accuracy on test data".format(
            results['test_accuracy'] * 100))
        
    except FileNotFoundError:
        print("❌ Error: OMFIT_Hmode_Studies.pkl not found!")
        print("Please ensure the data file is in the current directory.")
    except Exception as e:
        print(f"❌ Error running examples: {e}")


if __name__ == "__main__":
    main()