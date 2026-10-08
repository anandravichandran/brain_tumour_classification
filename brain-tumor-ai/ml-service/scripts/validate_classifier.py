"""
Validate the classifier on actual dataset images.
"""
import os
import glob
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import cv2

# Import production preprocessing
from app.preprocessing.preprocess import preprocess_for_classifier
from app.inference.classifier_inference import run_classifier
from app.core.config import settings
from app.models.model_loader import model_store

def main():
    model_store.load_all()
    dataset_dir = "/home/tech_polymath/Downloads/brain_tumor_dataset"
    
    classes = {"no": 0, "yes": 1}
    
    test_results = []
    
    # We will test up to 20 images from each class
    for cls_name, cls_idx in classes.items():
        cls_dir = os.path.join(dataset_dir, cls_name)
        if not os.path.isdir(cls_dir):
            print(f"Directory {cls_dir} not found.")
            continue
            
        images = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir) if f.endswith(('.jpg', '.jpeg', '.png', '.JPG'))]
        images = images[:20]  # Take 20 images
        
        for img_path in images:
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                continue
                
            # Run production preprocessing
            clf_batch = preprocess_for_classifier(img_bgr)
            
            # Run inference
            result = run_classifier(clf_batch)
            
            predicted_class = result["predicted_class_idx"]
            raw = result["raw_score"]
            threshold = settings.classifier_threshold
            pred_label = result["prediction"]
            expected_label = "tumor" if cls_idx == 1 else "no_tumor"
            
            passwd = "PASS" if expected_label == pred_label else "FAIL"
            
            print(f"{cls_name}/{os.path.basename(img_path)}")
            print(f"expected={expected_label}")
            print(f"raw={raw:.8f}")
            print(f"threshold={threshold}")
            print(f"predicted={pred_label}")
            print(passwd)
            print("-" * 20)
            
            test_results.append({
                "y_true": cls_idx,
                "y_pred": predicted_class
            })
            
    if not test_results:
        print("No test results. Dataset might be empty.")
        return
        
    y_true = [r["y_true"] for r in test_results]
    y_pred = [r["y_pred"] for r in test_results]
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Specificity calculation
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print("TEST RESULTS:")
    print(f"overall accuracy: {acc:.4f}")
    print(f"precision: {prec:.4f}")
    print(f"recall: {rec:.4f}")
    print(f"specificity: {specificity:.4f}")
    print(f"F1: {f1:.4f}")
    print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

if __name__ == "__main__":
    main()
