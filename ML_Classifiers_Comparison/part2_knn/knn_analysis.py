# Part 2: KNN Analysis
# بخش ۲: تحلیل الگوریتم KNN

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
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

def run_knn_analysis():
    """Run KNN analysis for k=1 to 40"""
    
    # Load data
    digits = load_digits()
    X = digits.data
    y = digits.target
    
    # Split data 70/30
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Prepare scaled data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    k_values = range(1, 41)
    
    # Results storage
    results = {
        'without_norm': {'accuracy': [], 'error': []},
        'with_norm': {'accuracy': [], 'error': []}
    }
    
    print("=" * 60)
    print("تحلیل الگوریتم KNN برای مقادیر مختلف k")
    print("KNN Analysis for Different k Values")
    print("=" * 60)
    
    # Without normalization
    print("\n--- بدون نرمال‌سازی (Without Normalization) ---")
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        error = 1 - acc
        results['without_norm']['accuracy'].append(acc)
        results['without_norm']['error'].append(error)
        if k % 10 == 0 or k == 1:
            print(f"k={k:2d}: Accuracy={acc:.4f}, Error={error:.4f}")
    
    # With normalization
    print("\n--- با StandardScaler (With StandardScaler) ---")
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_scaled, y_train)
        y_pred = knn.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        error = 1 - acc
        results['with_norm']['accuracy'].append(acc)
        results['with_norm']['error'].append(error)
        if k % 10 == 0 or k == 1:
            print(f"k={k:2d}: Accuracy={acc:.4f}, Error={error:.4f}")
    
    # Find optimal k
    best_k_without = k_values[np.argmax(results['without_norm']['accuracy'])]
    best_acc_without = max(results['without_norm']['accuracy'])
    
    best_k_with = k_values[np.argmax(results['with_norm']['accuracy'])]
    best_acc_with = max(results['with_norm']['accuracy'])
    
    print("\n" + "=" * 60)
    print("نتایج بهینه (Optimal Results)")
    print("=" * 60)
    print(f"بدون نرمال‌سازی - بهترین k: {best_k_without}, دقت: {best_acc_without:.4f}")
    print(f"با نرمال‌سازی - بهترین k: {best_k_with}, دقت: {best_acc_with:.4f}")
    
    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Accuracy vs k
    axes[0].plot(k_values, results['without_norm']['accuracy'], 'b-o', 
                 label=persian_text('بدون نرمال‌سازی') + ' (Without Normalization)', markersize=4)
    axes[0].plot(k_values, results['with_norm']['accuracy'], 'g-s', 
                 label=persian_text('با') + ' StandardScaler (With StandardScaler)', markersize=4)
    axes[0].axvline(x=best_k_without, color='blue', linestyle='--', alpha=0.5)
    axes[0].axvline(x=best_k_with, color='green', linestyle='--', alpha=0.5)
    axes[0].set_xlabel(persian_text('مقدار') + ' k (k Value)', fontsize=12)
    axes[0].set_ylabel(persian_text('دقت') + ' (Accuracy)', fontsize=12)
    axes[0].set_title(persian_text('نمودار دقت بر حسب') + ' k\n(Accuracy vs k)', fontsize=14)
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(range(0, 41, 5))
    
    # Plot 2: Error vs k
    axes[1].plot(k_values, results['without_norm']['error'], 'r-o', 
                 label=persian_text('بدون نرمال‌سازی') + ' (Without Normalization)', markersize=4)
    axes[1].plot(k_values, results['with_norm']['error'], 'm-s', 
                 label=persian_text('با') + ' StandardScaler (With StandardScaler)', markersize=4)
    axes[1].set_xlabel(persian_text('مقدار') + ' k (k Value)', fontsize=12)
    axes[1].set_ylabel(persian_text('خطا') + ' (Error)', fontsize=12)
    axes[1].set_title(persian_text('نمودار خطا بر حسب') + ' k\n(Error vs k)', fontsize=14)
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(range(0, 41, 5))
    
    plt.tight_layout()
    plt.savefig('knn_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Analysis
    print("\n" + "=" * 60)
    print("تحلیل نتایج (Analysis)")
    print("=" * 60)
    print("""
تحلیل اثر مقدار k:
-----------------
• k کوچک (مثلاً k=1):
  - مدل بسیار حساس به نویز و داده‌های پرت است
  - احتمال Overfitting بالا
  - واریانس بالا، بایاس پایین
  
• k بزرگ (مثلاً k=40):
  - مدل بیش از حد ساده می‌شود
  - احتمال Underfitting بالا
  - واریانس پایین، بایاس بالا
  
• k بهینه بین این دو حالت قرار دارد

تفاوت نتایج با و بدون نرمال‌سازی:
---------------------------------
• KNN از فاصله اقلیدسی استفاده می‌کند
• بدون نرمال‌سازی، ویژگی‌های با مقیاس بزرگتر تأثیر بیشتری دارند
• نرمال‌سازی باعث می‌شود همه ویژگی‌ها تأثیر برابر داشته باشند
• در دیتاست digits که همه پیکسل‌ها در بازه [0,16] هستند، 
  تفاوت چندانی مشاهده نمی‌شود
""")
    
    return results, best_k_without, best_k_with

if __name__ == "__main__":
    results, best_k_without, best_k_with = run_knn_analysis()
