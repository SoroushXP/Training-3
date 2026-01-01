# مقایسه و تحلیل دسته‌بندهای متفاوت

## به نام خدا

## هدف پروژه
در این پروژه، رفتار الگوریتم‌های مختلف یادگیری ماشین را روی دیتاست `digits` بررسی و مقایسه می‌کنیم.

## دیتاست
- **نام**: Digits Dataset از sklearn
- **تعداد نمونه**: ۱۷۹۷
- **تعداد ویژگی**: ۶۴ (تصاویر ۸×۸)
- **تعداد کلاس**: ۱۰ (ارقام ۰ تا ۹)

## ساختار پروژه

```
ML_Classifiers_Comparison/
│
├── README.md                    # این فایل
├── requirements.txt             # پیش‌نیازها
│
├── part1_data_preparation/      # بخش ۱: آماده‌سازی داده‌ها
│   ├── data_preparation.py
│   └── README.md
│
├── part2_knn/                   # بخش ۲: تحلیل KNN
│   ├── knn_analysis.py
│   └── README.md
│
├── part3_decision_tree/         # بخش ۳: تحلیل درخت تصمیم
│   ├── decision_tree_analysis.py
│   └── README.md
│
├── part4_svm/                   # بخش ۴: تحلیل SVM
│   ├── svm_analysis.py
│   └── README.md
│
└── part5_comparison/            # بخش ۵: مقایسه نهایی
    ├── final_comparison.py
    └── README.md
```

## بخش‌های پروژه

### بخش ۱: آماده‌سازی داده‌ها
- بارگذاری دیتاست digits
- تقسیم به آموزش (۷۰٪) و آزمون (۳۰٪)
- دو حالت: با و بدون نرمال‌سازی StandardScaler
- گزارش تعداد نمونه‌ها، ویژگی‌ها و کلاس‌ها

### بخش ۲: تحلیل KNN
- اجرای KNN برای k=1 تا k=40
- محاسبه دقت و خطا
- رسم نمودار دقت بر حسب k
- یافتن k بهینه
- تحلیل Overfitting و Underfitting

### بخش ۳: تحلیل درخت تصمیم
- ساخت درخت با معیار entropy
- آزمایش عمق ۲ تا ۱۰
- رسم نمودار دقت آموزش و آزمون
- شناسایی نقطه شروع Overfitting

### بخش ۴: تحلیل SVM
- SVM خطی با C = [0.01, 0.1, 1, 10, 100]
- کرنل‌های Linear, Polynomial (degree=3), RBF
- ماتریس درهم‌ریختگی
- تحلیل اثر C روی Margin و Generalization

### بخش ۵: مقایسه نهایی
- جدول مقایسه با بهترین پارامترها
- معیارهای Accuracy, Precision, Recall
- پیشنهاد بهترین الگوریتم

## پیش‌نیازها
```
numpy
scikit-learn
matplotlib
pandas
```

## نصب پیش‌نیازها
```bash
pip install -r requirements.txt
```

## نحوه اجرا

### اجرای هر بخش به صورت جداگانه:
```bash
# بخش ۱
cd part1_data_preparation
python data_preparation.py

# بخش ۲
cd part2_knn
python knn_analysis.py

# بخش ۳
cd part3_decision_tree
python decision_tree_analysis.py

# بخش ۴
cd part4_svm
python svm_analysis.py

# بخش ۵
cd part5_comparison
python final_comparison.py
```

## خروجی‌ها
هر بخش نمودارها و گزارش‌های مربوطه را تولید می‌کند:
- فایل‌های PNG برای نمودارها
- خروجی متنی در ترمینال
- فایل CSV برای جدول مقایسه نهایی

## نتایج مورد انتظار
- **KNN**: بهترین k معمولاً بین ۱ تا ۱۰
- **Decision Tree**: بهترین عمق معمولاً بین ۵ تا ۸
- **SVM**: کرنل RBF با C مناسب بهترین نتیجه

## نکات مهم
1. نرمال‌سازی برای KNN و SVM مهم است
2. افزایش پیچیدگی مدل = خطر Overfitting
3. انتخاب پارامتر مناسب نیازمند آزمایش است

## نویسنده
این پروژه برای تمرین مقایسه الگوریتم‌های دسته‌بندی یادگیری ماشین تهیه شده است.
