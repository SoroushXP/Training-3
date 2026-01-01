# Part 3: Decision Tree Analysis
# بخش ۳: تحلیل درخت تصمیم

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
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

def run_decision_tree_analysis():
    """Run Decision Tree analysis for max_depth 2 to 10"""
    
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
    
    depths = range(2, 11)
    
    # Results storage
    results = {
        'without_norm': {'train_acc': [], 'test_acc': []},
        'with_norm': {'train_acc': [], 'test_acc': []}
    }
    
    print("=" * 60)
    print("تحلیل درخت تصمیم برای عمق‌های مختلف")
    print("Decision Tree Analysis for Different Depths")
    print("=" * 60)
    
    # Without normalization
    print("\n--- بدون نرمال‌سازی (Without Normalization) ---")
    print(f"{'Depth':<8} {'Train Acc':<12} {'Test Acc':<12}")
    print("-" * 32)
    
    for depth in depths:
        dt = DecisionTreeClassifier(criterion='entropy', max_depth=depth, random_state=42)
        dt.fit(X_train, y_train)
        
        train_pred = dt.predict(X_train)
        test_pred = dt.predict(X_test)
        
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        
        results['without_norm']['train_acc'].append(train_acc)
        results['without_norm']['test_acc'].append(test_acc)
        
        print(f"{depth:<8} {train_acc:<12.4f} {test_acc:<12.4f}")
    
    # With normalization
    print("\n--- با StandardScaler (With StandardScaler) ---")
    print(f"{'Depth':<8} {'Train Acc':<12} {'Test Acc':<12}")
    print("-" * 32)
    
    for depth in depths:
        dt = DecisionTreeClassifier(criterion='entropy', max_depth=depth, random_state=42)
        dt.fit(X_train_scaled, y_train)
        
        train_pred = dt.predict(X_train_scaled)
        test_pred = dt.predict(X_test_scaled)
        
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        
        results['with_norm']['train_acc'].append(train_acc)
        results['with_norm']['test_acc'].append(test_acc)
        
        print(f"{depth:<8} {train_acc:<12.4f} {test_acc:<12.4f}")
    
    # Find overfitting point
    test_accs = results['without_norm']['test_acc']
    train_accs = results['without_norm']['train_acc']
    
    # Overfitting starts when test accuracy starts decreasing while train increases
    overfitting_depth = None
    for i in range(1, len(test_accs)):
        if test_accs[i] < test_accs[i-1] and train_accs[i] > train_accs[i-1]:
            overfitting_depth = depths[i]
            break
    
    # Best depth based on test accuracy
    best_depth = depths[np.argmax(test_accs)]
    best_test_acc = max(test_accs)
    
    print("\n" + "=" * 60)
    print("نتایج تحلیل (Analysis Results)")
    print("=" * 60)
    print(f"بهترین عمق (Best Depth): {best_depth}")
    print(f"بهترین دقت آزمون (Best Test Accuracy): {best_test_acc:.4f}")
    if overfitting_depth:
        print(f"شروع Overfitting از عمق (Overfitting starts at depth): {overfitting_depth}")
    
    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Without normalization
    axes[0].plot(depths, results['without_norm']['train_acc'], 'b-o', 
                 label=persian_text('دقت آموزش') + ' (Train Accuracy)', markersize=8, linewidth=2)
    axes[0].plot(depths, results['without_norm']['test_acc'], 'r-s', 
                 label=persian_text('دقت آزمون') + ' (Test Accuracy)', markersize=8, linewidth=2)
    axes[0].axvline(x=best_depth, color='green', linestyle='--', 
                    label=persian_text('بهترین عمق') + f' = {best_depth}', alpha=0.7)
    axes[0].fill_between(depths, results['without_norm']['train_acc'], 
                         results['without_norm']['test_acc'], alpha=0.2, color='purple')
    axes[0].set_xlabel(persian_text('عمق درخت') + ' (Tree Depth)', fontsize=12)
    axes[0].set_ylabel(persian_text('دقت') + ' (Accuracy)', fontsize=12)
    axes[0].set_title(persian_text('بدون نرمال‌سازی') + '\n(Without Normalization)', fontsize=14)
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(depths)
    
    # Plot 2: With normalization
    axes[1].plot(depths, results['with_norm']['train_acc'], 'b-o', 
                 label=persian_text('دقت آموزش') + ' (Train Accuracy)', markersize=8, linewidth=2)
    axes[1].plot(depths, results['with_norm']['test_acc'], 'r-s', 
                 label=persian_text('دقت آزمون') + ' (Test Accuracy)', markersize=8, linewidth=2)
    axes[1].fill_between(depths, results['with_norm']['train_acc'], 
                         results['with_norm']['test_acc'], alpha=0.2, color='purple')
    axes[1].set_xlabel(persian_text('عمق درخت') + ' (Tree Depth)', fontsize=12)
    axes[1].set_ylabel(persian_text('دقت') + ' (Accuracy)', fontsize=12)
    axes[1].set_title(persian_text('با') + ' StandardScaler\n(With StandardScaler)', fontsize=14)
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(depths)
    
    plt.tight_layout()
    plt.savefig('decision_tree_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Overfitting visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    
    gap = np.array(results['without_norm']['train_acc']) - np.array(results['without_norm']['test_acc'])
    ax.bar(depths, gap, color='orange', alpha=0.7)
    ax.set_xlabel(persian_text('عمق درخت') + ' (Tree Depth)', fontsize=12)
    ax.set_ylabel(persian_text('شکاف دقت آموزش و آزمون') + '\n(Train-Test Accuracy Gap)', fontsize=12)
    ax.set_title(persian_text('شاخص') + ' Overfitting ' + persian_text('(افزایش شکاف = بیشتر)') + '\n(Overfitting Indicator)', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticks(depths)
    
    plt.tight_layout()
    plt.savefig('overfitting_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Analysis
    print("\n" + "=" * 60)
    print("تحلیل نتایج (Analysis)")
    print("=" * 60)
    print("""
آیا افزایش عمق باعث بهبود مدل می‌شود؟
--------------------------------------
• در ابتدا، افزایش عمق باعث بهبود هر دو دقت آموزش و آزمون می‌شود
• پس از یک نقطه بهینه، افزایش عمق:
  - دقت آموزش همچنان افزایش می‌یابد
  - دقت آزمون ثابت می‌ماند یا کاهش می‌یابد
  - این نشانه Overfitting است

نقطه شروع Overfitting:
-----------------------
• زمانی که شکاف بین دقت آموزش و آزمون زیاد می‌شود
• مدل داده‌های آموزش را "حفظ" می‌کند به جای "یادگیری"
• ناحیه رنگی بین دو منحنی نشان‌دهنده این شکاف است

نتیجه‌گیری:
-----------
• عمق بهینه تعادلی بین پیچیدگی مدل و قابلیت تعمیم ایجاد می‌کند
• عمق بیش از حد = Overfitting
• عمق کم = Underfitting
""")
    
    return results, best_depth

if __name__ == "__main__":
    results, best_depth = run_decision_tree_analysis()
