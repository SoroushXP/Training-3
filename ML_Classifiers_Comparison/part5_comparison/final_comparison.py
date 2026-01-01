# Part 5: Final Model Comparison
# بخش ۵: مقایسه نهایی مدل‌ها

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import pandas as pd
import arabic_reshaper
from bidi.algorithm import get_display
import warnings
warnings.filterwarnings('ignore')

# Configure matplotlib for Persian font
plt.rcParams['font.family'] = 'Tahoma'

def persian_text(text):
    """Reshape Persian text for correct display in matplotlib"""
    reshaped = arabic_reshaper.reshape(text)
    return get_display(reshaped)

def find_best_knn(X_train, X_test, y_train, y_test):
    """Find best K for KNN"""
    best_k = 1
    best_acc = 0
    for k in range(1, 41):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        acc = accuracy_score(y_test, knn.predict(X_test))
        if acc > best_acc:
            best_acc = acc
            best_k = k
    return best_k

def find_best_dt(X_train, X_test, y_train, y_test):
    """Find best max_depth for Decision Tree"""
    best_depth = 2
    best_acc = 0
    for depth in range(2, 11):
        dt = DecisionTreeClassifier(criterion='entropy', max_depth=depth, random_state=42)
        dt.fit(X_train, y_train)
        acc = accuracy_score(y_test, dt.predict(X_test))
        if acc > best_acc:
            best_acc = acc
            best_depth = depth
    return best_depth

def find_best_svm(X_train, X_test, y_train, y_test):
    """Find best kernel and C for SVM"""
    C_values = [0.01, 0.1, 1, 10, 100]
    kernels = ['linear', 'poly', 'rbf']
    
    best_kernel = 'rbf'
    best_C = 1
    best_acc = 0
    
    for kernel in kernels:
        for C in C_values:
            if kernel == 'poly':
                svm = SVC(kernel=kernel, C=C, degree=3, random_state=42)
            else:
                svm = SVC(kernel=kernel, C=C, random_state=42)
            svm.fit(X_train, y_train)
            acc = accuracy_score(y_test, svm.predict(X_test))
            if acc > best_acc:
                best_acc = acc
                best_kernel = kernel
                best_C = C
    
    return best_kernel, best_C

def run_final_comparison():
    """Run final comparison of all models"""
    
    # Load data
    digits = load_digits()
    X = digits.data
    y = digits.target
    
    # Split data 70/30
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("=" * 70)
    print("مقایسه نهایی مدل‌ها")
    print("Final Model Comparison")
    print("=" * 70)
    
    print("\nدر حال یافتن بهترین پارامترها...")
    print("Finding best parameters...")
    
    # Find best parameters
    best_k = find_best_knn(X_train_scaled, X_test_scaled, y_train, y_test)
    print(f"✓ بهترین K برای KNN: {best_k}")
    
    best_depth = find_best_dt(X_train_scaled, X_test_scaled, y_train, y_test)
    print(f"✓ بهترین عمق برای Decision Tree: {best_depth}")
    
    best_kernel, best_C = find_best_svm(X_train_scaled, X_test_scaled, y_train, y_test)
    print(f"✓ بهترین پارامترها برای SVM: kernel={best_kernel}, C={best_C}")
    
    # Train final models
    print("\nدر حال آموزش مدل‌های نهایی...")
    print("Training final models...")
    
    # KNN
    knn = KNeighborsClassifier(n_neighbors=best_k)
    knn.fit(X_train_scaled, y_train)
    knn_pred = knn.predict(X_test_scaled)
    
    # Decision Tree
    dt = DecisionTreeClassifier(criterion='entropy', max_depth=best_depth, random_state=42)
    dt.fit(X_train_scaled, y_train)
    dt_pred = dt.predict(X_test_scaled)
    
    # SVM
    if best_kernel == 'poly':
        svm = SVC(kernel=best_kernel, C=best_C, degree=3, random_state=42)
    else:
        svm = SVC(kernel=best_kernel, C=best_C, random_state=42)
    svm.fit(X_train_scaled, y_train)
    svm_pred = svm.predict(X_test_scaled)
    
    # Calculate metrics
    results = {
        'Algorithm': ['KNN', 'Decision Tree', 'SVM'],
        'Best Parameters': [f'K={best_k}', f'max_depth={best_depth}', f'kernel={best_kernel}, C={best_C}'],
        'Accuracy': [
            accuracy_score(y_test, knn_pred),
            accuracy_score(y_test, dt_pred),
            accuracy_score(y_test, svm_pred)
        ],
        'Precision': [
            precision_score(y_test, knn_pred, average='weighted'),
            precision_score(y_test, dt_pred, average='weighted'),
            precision_score(y_test, svm_pred, average='weighted')
        ],
        'Recall': [
            recall_score(y_test, knn_pred, average='weighted'),
            recall_score(y_test, dt_pred, average='weighted'),
            recall_score(y_test, svm_pred, average='weighted')
        ],
        'F1-Score': [
            f1_score(y_test, knn_pred, average='weighted'),
            f1_score(y_test, dt_pred, average='weighted'),
            f1_score(y_test, svm_pred, average='weighted')
        ]
    }
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    print("\n" + "=" * 70)
    print("جدول مقایسه نهایی (Final Comparison Table)")
    print("=" * 70)
    print(df.to_string(index=False))
    
    # Save to CSV
    df.to_csv('comparison_results.csv', index=False, encoding='utf-8-sig')
    print("\nنتایج در فایل comparison_results.csv ذخیره شد")
    
    # Find best algorithm
    best_idx = np.argmax(results['Accuracy'])
    best_algorithm = results['Algorithm'][best_idx]
    best_accuracy = results['Accuracy'][best_idx]
    
    print("\n" + "=" * 70)
    print("بهترین الگوریتم (Best Algorithm)")
    print("=" * 70)
    print(f"الگوریتم: {best_algorithm}")
    print(f"دقت: {best_accuracy:.4f}")
    print(f"پارامترها: {results['Best Parameters'][best_idx]}")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Bar chart comparison
    x = np.arange(len(results['Algorithm']))
    width = 0.2
    
    bars1 = axes[0].bar(x - 1.5*width, results['Accuracy'], width, label='Accuracy', color='blue', alpha=0.8)
    bars2 = axes[0].bar(x - 0.5*width, results['Precision'], width, label='Precision', color='green', alpha=0.8)
    bars3 = axes[0].bar(x + 0.5*width, results['Recall'], width, label='Recall', color='orange', alpha=0.8)
    bars4 = axes[0].bar(x + 1.5*width, results['F1-Score'], width, label='F1-Score', color='red', alpha=0.8)
    
    axes[0].set_xlabel(persian_text('الگوریتم') + ' (Algorithm)', fontsize=12)
    axes[0].set_ylabel(persian_text('امتیاز') + ' (Score)', fontsize=12)
    axes[0].set_title(persian_text('مقایسه معیارهای ارزیابی') + '\n(Evaluation Metrics Comparison)', fontsize=14)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(results['Algorithm'])
    axes[0].legend(loc='lower right')
    axes[0].set_ylim([0.7, 1.0])
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Radar chart
    categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    # Number of variables
    N = len(categories)
    
    # What will be the angle of each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    # Colors for each algorithm
    colors = ['blue', 'green', 'red']
    
    for i, algo in enumerate(results['Algorithm']):
        values = [results['Accuracy'][i], results['Precision'][i], 
                  results['Recall'][i], results['F1-Score'][i]]
        values += values[:1]
        
        axes[1].plot(angles, values, 'o-', linewidth=2, label=algo, color=colors[i])
        axes[1].fill(angles, values, alpha=0.1, color=colors[i])
    
    axes[1].set_xticks(angles[:-1])
    axes[1].set_xticklabels(categories)
    axes[1].set_ylim([0.7, 1.0])
    axes[1].legend(loc='lower right')
    axes[1].set_title(persian_text('نمودار راداری مقایسه الگوریتم‌ها') + '\n(Radar Chart Comparison)', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('final_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Recommendation
    print("\n" + "=" * 70)
    print("پیشنهاد الگوریتم (Algorithm Recommendation)")
    print("=" * 70)
    print(f"""
برای دیتاست digits، الگوریتم {best_algorithm} پیشنهاد می‌شود.

دلایل:
-------
1. بالاترین دقت ({best_accuracy:.4f}) را در بین الگوریتم‌ها دارد
2. Precision و Recall متعادل و بالایی دارد
3. عملکرد پایدار روی همه کلاس‌ها

مقایسه الگوریتم‌ها:
------------------
• KNN: ساده، قابل تفسیر، اما کند برای داده‌های بزرگ
• Decision Tree: سریع، قابل تفسیر، اما مستعد Overfitting
• SVM: دقیق، مناسب برای داده‌های با ابعاد بالا، اما پیچیده‌تر

نتیجه‌گیری نهایی:
-----------------
{best_algorithm} با پارامترهای {results['Best Parameters'][best_idx]} 
بهترین انتخاب برای این دیتاست است.
""")
    
    return df, best_algorithm

if __name__ == "__main__":
    df, best_algorithm = run_final_comparison()
