# Part 1: Data Preparation
# بخش ۱: آماده‌سازی داده‌ها

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import arabic_reshaper
from bidi.algorithm import get_display

# Configure matplotlib for Persian font
plt.rcParams['font.family'] = 'Tahoma'

def persian_text(text):
    """Reshape Persian text for correct display in matplotlib"""
    reshaped = arabic_reshaper.reshape(text)
    return get_display(reshaped)

def load_and_prepare_data():
    """Load digits dataset and prepare it for analysis"""
    
    # Load the digits dataset
    digits = load_digits()
    X = digits.data
    y = digits.target
    
    # Report dataset information
    print("=" * 50)
    print("گزارش اطلاعات دیتاست (Dataset Information Report)")
    print("=" * 50)
    print(f"تعداد نمونه‌ها (Number of samples): {X.shape[0]}")
    print(f"تعداد ویژگی‌ها (Number of features): {X.shape[1]}")
    print(f"تعداد کلاس‌ها (Number of classes): {len(np.unique(y))}")
    print(f"کلاس‌ها (Classes): {np.unique(y)}")
    print("=" * 50)
    
    # Split data 70% train, 30% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"\nتعداد نمونه‌های آموزش (Training samples): {X_train.shape[0]}")
    print(f"تعداد نمونه‌های آزمون (Test samples): {X_test.shape[0]}")
    
    # Case 1: Without normalization
    data_without_norm = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }
    
    # Case 2: With StandardScaler normalization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    data_with_norm = {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler
    }
    
    # Visualize some sample digits
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    fig.suptitle(persian_text('نمونه‌ای از ارقام دیتاست') + ' (Sample Digits from Dataset)', fontsize=14)
    
    for i, ax in enumerate(axes.flat):
        ax.imshow(digits.images[i], cmap='gray')
        ax.set_title(f'Label: {digits.target[i]}')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_digits.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Compare data distribution before and after normalization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Before normalization
    axes[0].hist(X_train.flatten(), bins=50, alpha=0.7, color='blue')
    axes[0].set_title(persian_text('توزیع داده‌ها بدون نرمال‌سازی') + '\n(Data Distribution Without Normalization)')
    axes[0].set_xlabel(persian_text('مقدار پیکسل') + ' (Pixel Value)')
    axes[0].set_ylabel(persian_text('فراوانی') + ' (Frequency)')
    
    # After normalization
    axes[1].hist(X_train_scaled.flatten(), bins=50, alpha=0.7, color='green')
    axes[1].set_title(persian_text('توزیع داده‌ها با') + ' StandardScaler\n(Data Distribution With StandardScaler)')
    axes[1].set_xlabel(persian_text('مقدار نرمال‌شده') + ' (Normalized Value)')
    axes[1].set_ylabel(persian_text('فراوانی') + ' (Frequency)')
    
    plt.tight_layout()
    plt.savefig('normalization_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n" + "=" * 50)
    print("آماده‌سازی داده‌ها با موفقیت انجام شد!")
    print("Data preparation completed successfully!")
    print("=" * 50)
    
    return data_without_norm, data_with_norm, digits

if __name__ == "__main__":
    data_without_norm, data_with_norm, digits = load_and_prepare_data()
