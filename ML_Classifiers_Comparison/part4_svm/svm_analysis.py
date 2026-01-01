# Part 4: SVM Analysis
# بخش ۴: تحلیل ماشین بردار پشتیبان (SVM)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
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

def run_svm_analysis():
    """Run SVM analysis with different kernels and C values"""
    
    # Load data
    digits = load_digits()
    X = digits.data
    y = digits.target
    
    # Split data 70/30
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Scale data (important for SVM)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    C_values = [0.01, 0.1, 1, 10, 100]
    kernels = ['linear', 'poly', 'rbf']
    
    print("=" * 70)
    print("تحلیل SVM با کرنل‌ها و مقادیر C مختلف")
    print("SVM Analysis with Different Kernels and C Values")
    print("=" * 70)
    
    # Results storage
    results = {}
    
    # Part 1: Linear SVM with different C values
    print("\n" + "=" * 70)
    print("بخش اول: SVM خطی با مقادیر مختلف C")
    print("Part 1: Linear SVM with Different C Values")
    print("=" * 70)
    
    linear_results = {'C': [], 'accuracy': [], 'model': []}
    
    print(f"\n{'C Value':<12} {'Accuracy':<12}")
    print("-" * 24)
    
    for C in C_values:
        svm = SVC(kernel='linear', C=C, random_state=42)
        svm.fit(X_train_scaled, y_train)
        y_pred = svm.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        
        linear_results['C'].append(C)
        linear_results['accuracy'].append(acc)
        linear_results['model'].append(svm)
        
        print(f"{C:<12} {acc:<12.4f}")
    
    # Part 2: Different Kernels with best C
    print("\n" + "=" * 70)
    print("بخش دوم: مقایسه کرنل‌های مختلف")
    print("Part 2: Comparing Different Kernels")
    print("=" * 70)
    
    kernel_results = {}
    
    for kernel in kernels:
        kernel_results[kernel] = {'C': [], 'accuracy': [], 'best_model': None, 'best_acc': 0}
        
        print(f"\n--- Kernel: {kernel.upper()} ---")
        print(f"{'C Value':<12} {'Accuracy':<12}")
        print("-" * 24)
        
        for C in C_values:
            if kernel == 'poly':
                svm = SVC(kernel=kernel, C=C, degree=3, random_state=42)
            else:
                svm = SVC(kernel=kernel, C=C, random_state=42)
            
            svm.fit(X_train_scaled, y_train)
            y_pred = svm.predict(X_test_scaled)
            acc = accuracy_score(y_test, y_pred)
            
            kernel_results[kernel]['C'].append(C)
            kernel_results[kernel]['accuracy'].append(acc)
            
            if acc > kernel_results[kernel]['best_acc']:
                kernel_results[kernel]['best_acc'] = acc
                kernel_results[kernel]['best_model'] = svm
                kernel_results[kernel]['best_C'] = C
            
            print(f"{C:<12} {acc:<12.4f}")
    
    results['linear'] = linear_results
    results['kernels'] = kernel_results
    
    # Find best overall model
    best_kernel = max(kernel_results.keys(), key=lambda k: kernel_results[k]['best_acc'])
    best_model = kernel_results[best_kernel]['best_model']
    best_C = kernel_results[best_kernel]['best_C']
    best_acc = kernel_results[best_kernel]['best_acc']
    
    print("\n" + "=" * 70)
    print("بهترین مدل (Best Model)")
    print("=" * 70)
    print(f"Kernel: {best_kernel}")
    print(f"C: {best_C}")
    print(f"Accuracy: {best_acc:.4f}")
    
    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Accuracy vs C for different kernels
    colors = {'linear': 'blue', 'poly': 'green', 'rbf': 'red'}
    markers = {'linear': 'o', 'poly': 's', 'rbf': '^'}
    
    for kernel in kernels:
        axes[0].semilogx(kernel_results[kernel]['C'], kernel_results[kernel]['accuracy'],
                        marker=markers[kernel], color=colors[kernel], linestyle='-',
                        label=f'{kernel.upper()}', markersize=10, linewidth=2)
    
    axes[0].set_xlabel(persian_text('مقدار') + ' C (C Value)', fontsize=12)
    axes[0].set_ylabel(persian_text('دقت') + ' (Accuracy)', fontsize=12)
    axes[0].set_title(persian_text('دقت بر حسب') + ' C ' + persian_text('برای کرنل‌های مختلف') + '\n(Accuracy vs C for Different Kernels)', fontsize=14)
    axes[0].legend(loc='best', fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Bar chart comparing best accuracy of each kernel
    kernel_names = [k.upper() for k in kernels]
    best_accs = [kernel_results[k]['best_acc'] for k in kernels]
    best_Cs = [kernel_results[k]['best_C'] for k in kernels]
    
    bars = axes[1].bar(kernel_names, best_accs, color=['blue', 'green', 'red'], alpha=0.7)
    axes[1].set_xlabel(persian_text('کرنل') + ' (Kernel)', fontsize=12)
    axes[1].set_ylabel(persian_text('بهترین دقت') + ' (Best Accuracy)', fontsize=12)
    axes[1].set_title(persian_text('مقایسه بهترین دقت کرنل‌ها') + '\n(Best Accuracy Comparison)', fontsize=14)
    axes[1].set_ylim([0.9, 1.0])
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add C values on bars
    for bar, c in zip(bars, best_Cs):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, 
                    f'C={c}', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('svm_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Confusion matrices for best model of each kernel
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, kernel in enumerate(kernels):
        model = kernel_results[kernel]['best_model']
        y_pred = model.predict(X_test_scaled)
        cm = confusion_matrix(y_test, y_pred)
        
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=digits.target_names)
        disp.plot(ax=axes[idx], cmap='Blues', colorbar=False)
        axes[idx].set_title(f'{kernel.upper()} (C={kernel_results[kernel]["best_C"]})\nAccuracy: {kernel_results[kernel]["best_acc"]:.4f}')
    
    plt.suptitle(persian_text('ماتریس درهم‌ریختگی برای بهترین مدل هر کرنل') + '\n(Confusion Matrix for Best Model of Each Kernel)', 
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('svm_confusion_matrices.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Analysis
    print("\n" + "=" * 70)
    print("تحلیل اثر پارامتر C روی Margin و Generalization")
    print("Analysis of C Parameter Effect on Margin and Generalization")
    print("=" * 70)
    print("""
اثر پارامتر C:
--------------
• C کوچک (مثلاً 0.01):
  - Margin بزرگ‌تر (Soft Margin)
  - تحمل بیشتر برای خطاهای طبقه‌بندی
  - ممکن است منجر به Underfitting شود
  - Generalization بهتر ولی دقت آموزش کمتر

• C بزرگ (مثلاً 100):
  - Margin کوچک‌تر (Hard Margin)
  - تحمل کمتر برای خطاها
  - ممکن است منجر به Overfitting شود
  - دقت آموزش بالا ولی Generalization ضعیف‌تر

• C بهینه:
  - تعادل بین Margin و دقت طبقه‌بندی
  - بهترین Generalization روی داده‌های جدید

مقایسه کرنل‌ها:
---------------
• Linear: مناسب داده‌های خطی جدایی‌پذیر
• Polynomial (degree=3): می‌تواند روابط غیرخطی را مدل کند
• RBF: انعطاف‌پذیرترین، مناسب اکثر مسائل غیرخطی

بهترین عملکرد:
--------------
""")
    print(f"کرنل {best_kernel.upper()} با C={best_C} بهترین دقت {best_acc:.4f} را دارد")
    
    return results, best_kernel, best_C

if __name__ == "__main__":
    results, best_kernel, best_C = run_svm_analysis()
