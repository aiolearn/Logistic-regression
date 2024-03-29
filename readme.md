<h1 align="center">Welcome to Logistic regression Project 👋</h1>

# Logistic regression Project

In this project, we wrote a number recognition program with logistic regression

## Modules

We need the following modules to implement projects

```python
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import linear_model
import sklearn.metrics as sm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
```

## Usage

Import handwritten numbers dataset

```python
from sklearn import datasets
digits = datasets.load_digits() 
```

Filter the dataset to include only 0s and 1s
```python
mask = (digits.target == 0) | (digits.target == 1)
filtered_data = digits.data[mask]
filtered_target = digits.target[mask]
digits.data.shape
```

The label corresponding to the second example in the digits dataset

```python
digits.target[1]
```

Show pictures corresponding to numbers

```python
plt.imshow(digits.images[1])
```

x: data attributes from the digits dataset
y: labels corresponding to data from the digits dataset

```python
x=digits.data
y=digits.target
```

Divide the data into training and test data

```python
X_train , X_test , y_train , y_test = train_test_split(x, y , test_size=0.2)
```

Training dataset for features
```python
X_train
```
The dimensions of the training dataset for features (number of samples and number of features

```python
X_train.shape
```

Create a logistic regression model
```python
model=linear_model.LogisticRegression()
```

Model training with training data
```python
model.fit(X_train,y_train)
```

Output prediction with test data
```python 
out=model.predict(X_test)
out
y_test
```
Determining the correctness of the model prediction
```python
er=y_test - out
er
```

Calculate the number of correct and incorrect predictions
```python 
correct=0
incorrect=0
for i in er:
  if (i==0 or i==1):
    correct=correct+1
  else:
    incorrect=incorrect+1

print("correct: ",correct)
print("incoorect: ",incorrect)
```

## Result

This project was written by Majid Tajanjari and the Aiolearn team, and we need your support!❤️

# پروژه لاجستیک رگریشن

در این پروژه یک برنامه تشخیص اعداد با رگرسیون لجستیک نوشتیم

## ماژول ها

برای اجرای پروژه ها به ماژول های زیر نیاز داریم

```python
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import linear_model
import sklearn.metrics as sm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
```

## نحوه استفاده

وارد کردن مجموعه داده های اعداد دست نویس
```python
from sklearn import datasets
digits = datasets.load_digits() 
```

مجموعه داده را فیلتر کنید تا فقط شامل 0 و 1 باشد
```python
mask = (digits.target == 0) | (digits.target == 1)
filtered_data = digits.data[mask]
filtered_target = digits.target[mask]
digits.data.shape
```

برچسب مربوط به مثال دوم در مجموعه داده اعداد

```python
digits.target[1]
```

نمایش تصاویر مربوط به اعداد

```python
plt.imshow(digits.images[1])
```

x: ویژگی های داده از مجموعه داده اعداد
y: برچسب های مربوط به داده های مجموعه اعداد

```python
x=digits.data
y=digits.target
```

داده ها را به داده های آموزشی و آزمایشی تقسیم کنید

```python
X_train , X_test , y_train , y_test = train_test_split(x, y , test_size=0.2)
```

مجموعه داده های آموزشی برای ویژگی ها
```python
X_train
```
ابعاد مجموعه داده آموزشی برای ویژگی ها (تعداد نمونه ها و تعداد ویژگی ها

```python
X_train.shape
```

یک مدل رگرسیون لجستیک ایجاد کنید
```python
model=linear_model.LogisticRegression()
```

آموزش مدل با داده های آموزشی
```python
model.fit(X_train,y_train)
```

پیش بینی خروجی با داده های تست
```python 
out=model.predict(X_test)
out
y_test
```
تعیین صحت پیش‌بینی مدل
```python
er=y_test - out
er
```

تعداد پیش بینی های صحیح و نادرست را محاسبه کنید
```python 
correct=0
incorrect=0
for i in er:
  if (i==0 or i==1):
    correct=correct+1
  else:
    incorrect=incorrect+1

print("correct: ",correct)
print("incoorect: ",incorrect)
```

## نتیجه
این پروژه توسط مجید تجن جاری و تیم Aiolearn نوشته شده است و ما به حمایت شما نیازمندیم!❤️