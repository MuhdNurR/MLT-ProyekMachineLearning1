# -*- coding: utf-8 -*-
"""HearthAttack.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1WlzO6jUz6wg-6m7xkX2rcJX1PQY6wPRV

Nama : Muhammad Nur Rachman N. S

Dataset : https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset?resource=download
"""

# Commented out IPython magic to ensure Python compatibility.
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable
import seaborn as sns
import numpy as np
import pandas as pd
import os

# %matplotlib inline

!wget --no-check-certificate \
    "https://storage.googleapis.com/kaggle-data-sets/1226038/2047221/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20240221%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20240221T025041Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=10a136e04c9ae5eed22a6185fa4b796777c1ff366f6722640ca91ed63985fdee49c0e40404d3cff19947c5abaf334b344c2235bec9a1eb25b8b0d69ab25baae47d6d1e8a675c8e24ff950fa36b7ceeae5e92219bd18ec494397ecd7901ba50bfaa1b1b53fd19b21460332a363b22ba8e1b0689933cd113e9bd9f1dd5d2c6a902b27d67cdaba10e06cc35cb16bd9d012e3105700fddc3af1bc60168144f53fe78a8b6dadfb31c18b280bf2523d935d40eb1ec7a5b244144ff19c083385c57b752b89ab277e42d1a78e3aa0198d02d527a4ca8355f360f7020d1360d74dbd2f1bb9aeb5fed6d44b5d8303a9ad819c19d3b63b3ed517302f1072443293ceb273ae9" \
    -O "/content/archive.zip"

local_zip = '/content/archive.zip'
zip_ref   = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/content/Heart-Attack-Analysis-Prediction/')
zip_ref.close()

"""# Data Understanding"""

df = pd.read_csv('/content/Heart-Attack-Analysis-Prediction/heart.csv')
df.head()

"""* Age : Age of the patient
* Sex : Sex of the patient
* exang: exercise induced angina (1 = yes; 0 = no)
* ca: number of major vessels (0-3)
* cp : Chest Pain type chest pain type
 * Value 1: typical angina
 * Value 2: atypical angina
 * Value 3: non-anginal pain
* Value 4: asymptomatic
*trtbps : resting blood pressure (in mm Hg)
* chol : cholestoral in mg/dl fetched via BMI sensor
* fbs : (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
* rest_ecg : resting electrocardiographic results
 * Value 0: normal
 * Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
 * Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria
* thalach : maximum heart rate achieved
* target : 0 = less chance of heart attack 1 = more chance of heart attack

## Mengecek data yang kosong
"""

df.info()

print('Total missing value in the dataframe:', df.isnull().sum().sum(), 'records')

"""Setelah melakukan pengecekan data, didapati bahwa sudah dalam kondisi bersih dan tidak perlu adanya penghapusan nilai"""

sns.countplot(x='output', data=df, palette=['#3366cc', '#993399'])
plt.title('Distribution of Heart Disease (0: No, 1: Yes)')
plt.show()

"""Mengetahui berapa orang yang terkena penyakit jantung dan tidak terkena

# Correlation Matrix

## Correlation plot
"""

correlation_matrix = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='BuPu', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

"""Mengetahui hubungan antar fitur numerik dalam dataset

## Feature Distribution
"""

df.hist(figsize=(12, 10))
plt.suptitle('Feature Distributions', x=0.5, y=1.02, ha='center',fontsize='large')
plt.show()

"""Memahami distribusi data dalam analisis data dan pemilihan model pembelajaran mesin yang tepat."""

categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exng', 'slp', 'caa', 'thall']
num_rows = 4
num_cols = 2

fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 10))

for i, feature in enumerate(categorical_features):
    row_index = i // num_cols
    col_index = i % num_cols

    sns.countplot(x=feature, hue='output', data=df, palette=['#3366cc', '#993399'], ax=axes[row_index, col_index])
    axes[row_index, col_index].set_title(f'Distribution of {feature} by Heart Disease')

plt.tight_layout()
plt.show()

"""## Age distribution"""

plt.figure(figsize=(10, 6))
sns.histplot(x='age', data=df, bins=20, kde=True, hue='output', multiple='stack', palette=['#3366cc', '#993399'])
plt.title('Distribution of Age by Heart Disease (0: No, 1: Yes)')
plt.legend(title='Heart Disease', labels=['No', 'Yes'])
plt.show()

"""## Noise identification"""

numerical_features = ['age', 'trtbps', 'chol', 'thalachh', 'oldpeak']

plt.figure(figsize=(12, 8))
for i, feature in enumerate(numerical_features, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(x='output', y=feature, data=df, palette=['#3366cc', '#993399'])
    plt.title(f'Box Plot of {feature} by Heart Disease')

plt.tight_layout()
plt.show()

df = df[df['chol'] <= 500]

"""menghapus outlier pada chol atau kalori karena jauh dari mayoritas nilai"""

plt.figure(figsize=(6, 4))
sns.boxplot(x='output', y='chol', data=df, palette=['#3366cc', '#993399'])
plt.title('Box Plot of Cholesterol by Heart Disease')
plt.show()

X = df.drop('output', axis=1)
y = df['output']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f'Total # of sample in whole dataset: {len(X)}')
print(f'Total # of sample in train dataset: {len(X_train)}')
print(f'Total # of sample in test dataset: {len(X_test)}')

"""# Models

Beberapa jenis algoritma Machine Learning yang akan ditest:

* XGBoost: Algoritma boosting yang populer untuk klasifikasi dan regresi dengan performa tinggi dan efisiensi komputasi.
* Neural Network: Algoritma yang terinspirasi dari struktur otak manusia, mampu mempelajari pola kompleks dan membuat prediksi akurat, seperti convolutional neural networks (CNN) untuk pengenalan gambar dan recurrent neural networks (RNN) untuk pemrosesan bahasa alami.
* AdaBoost: Algoritma boosting yang menggabungkan beberapa model klasifikasi lemah untuk menghasilkan model yang lebih kuat.
* Naive Bayes: Algoritma klasifikasi sederhana dan mudah dipahami berdasarkan teorema Bayes, sering digunakan dalam aplikasi teks dan email spam filtering.
* KNN (K-Nearest Neighbors): Algoritma klasifikasi dan regresi sederhana yang memprediksi label data berdasarkan K data terdekat.
* SVM (Support Vector Machine): Algoritma klasifikasi dan regresi yang kuat dengan margin maksimal, mampu bekerja dengan data dimensi tinggi dan non-linear, sering digunakan untuk klasifikasi gambar dan teks.
* Random Forest: Algoritma ensemble learning yang menggabungkan beberapa decision tree untuk menghasilkan prediksi yang lebih akurat, mampu menangani overfitting dan meningkatkan stabilitas model.
* Logistic Regression: Algoritma klasifikasi probabilistik untuk memprediksi probabilitas suatu kejadian, sederhana dan mudah dipahami, sering digunakan dalam klasifikasi spam, prediksi risiko kredit, dan klasifikasi sentiment.

## Logistic Regression
"""

model_lr = LogisticRegression()
model_lr.fit(X_train_scaled, y_train)

y_pred = model_lr.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print(f'Logistic Regression Accuracy: {accuracy:.2f}')
print('Classification Report:\n', classification_report(y_test, y_pred))

plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap=sns.color_palette(['#3366cc', '#993399']))
plt.title('Confusion Matrix - Logistic Regression')
plt.show()

"""## Random Forest"""

model_rf = RandomForestClassifier(random_state=42)
model_rf.fit(X_train_scaled, y_train)

y_pred_rf = model_rf.predict(X_test_scaled)

accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f'Random Forest Accuracy: {accuracy_rf:.2f}')
print('Classification Report:\n', classification_report(y_test, y_pred_rf))

plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', cmap=sns.color_palette(['#3366cc', '#993399']))
plt.title('Confusion Matrix - Random Forest')
plt.show()

"""## Support Vector Machine"""

model_svm = SVC(kernel='linear', random_state=42)
model_svm.fit(X_train_scaled, y_train)
y_pred_svm = model_svm.predict(X_test_scaled)

accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f'SVM Accuracy: {accuracy_svm:.2f}')
print('Classification Report:\n', classification_report(y_test, y_pred_svm))

plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred_svm), annot=True, fmt='d', cmap=sns.color_palette(['#3366cc', '#993399']))
plt.title('Confusion Matrix - SVM')
plt.show()

"""## KNN"""

model_knn = KNeighborsClassifier()
model_knn.fit(X_train_scaled, y_train)

y_pred_knn = model_knn.predict(X_test_scaled)

accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f'KNN Accuracy: {accuracy_knn:.2f}')
print('Classification Report:\n', classification_report(y_test, y_pred_knn))

plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred_knn), annot=True, fmt='d', cmap=sns.color_palette(['#3366cc', '#993399']))
plt.title('Confusion Matrix - KNN')
plt.show()

"""## Naive Bayes"""

model_nb = GaussianNB()
model_nb.fit(X_train_scaled, y_train)

y_pred_nb = model_nb.predict(X_test_scaled)

accuracy_nb = accuracy_score(y_test, y_pred_nb)
print(f'Naive Bayes Accuracy: {accuracy_nb:.2f}')
print('Classification Report:\n', classification_report(y_test, y_pred_nb))


plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred_nb), annot=True, fmt='d', cmap=sns.color_palette(['#3366cc', '#993399']))
plt.title('Confusion Matrix - Naive Bayes')
plt.show()

"""## Ada Boost Classifier"""

model_adaboost = AdaBoostClassifier(random_state=42)
model_adaboost.fit(X_train_scaled, y_train)


y_pred_adaboost = model_adaboost.predict(X_test_scaled)


accuracy_adaboost = accuracy_score(y_test, y_pred_adaboost)
print(f'AdaBoost Accuracy: {accuracy_adaboost:.2f}')
print('Classification Report:\n', classification_report(y_test, y_pred_adaboost))

plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred_adaboost), annot=True, fmt='d', cmap=sns.color_palette(['#3366cc', '#993399']))
plt.title('Confusion Matrix - AdaBoost')
plt.show()

"""## XGBOOST"""

model_xgb = XGBClassifier(random_state=42)
model_xgb.fit(X_train_scaled, y_train)

y_pred_xgb = model_xgb.predict(X_test_scaled)

accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
print(f'XGBoost Accuracy: {accuracy_xgb:.2f}')
print('Classification Report:\n', classification_report(y_test, y_pred_xgb))

plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred_xgb), annot=True, fmt='d', cmap=sns.color_palette(['#3366cc', '#993399']))
plt.title('Confusion Matrix - XGBoost')
plt.show()

"""## Multi Layer Perceptron"""

model_nn = MLPClassifier(random_state=42, max_iter=500)
model_nn.fit(X_train_scaled, y_train)

y_pred_nn = model_nn.predict(X_test_scaled)

accuracy_nn = accuracy_score(y_test, y_pred_nn)
print(f'Neural Network Accuracy: {accuracy_nn:.2f}')
print('Classification Report:\n', classification_report(y_test, y_pred_nn))

plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred_nn), annot=True, fmt='d', cmap=sns.color_palette(['#3366cc', '#993399']))
plt.title('Confusion Matrix - Neural Network')
plt.show()

"""# Models Summary

Dapat diketahui bahwa model dengan akurasi tinggi yaitu Random Forest dalam kasus ini
"""

from sklearn.metrics import mean_squared_error

# Mean squared error dari model
mse = pd.DataFrame(columns=['train', 'test'], index=['XGBoost', 'Neural Network', 'AdaBoost', 'Naive Bayes', 'KNN', 'SVM', 'Random Forest', 'Logistic Regression'])

model_dict = {'Logistic Regression': model_lr, 'RF': model_rf, 'SVM': model_svm, 'KNN' : model_knn, 'Naive Bayes' : model_nb, 'AdaBoost' : model_adaboost, 'XGBoost' : model_xgb, 'Neural Network' : model_nn}

for name, model in model_dict.items():
    mse.loc[name, 'train'] = mean_squared_error(y_true=y_train, y_pred=model.predict(X_train))/1e3
    mse.loc[name, 'test'] = mean_squared_error(y_true=y_test, y_pred=model.predict(X_test))/1e3

mse

fig, ax = plt.subplots()
mse.sort_values(by='test', ascending=False).plot(kind='barh', ax=ax, zorder=3)
ax.grid(zorder=0)

models = ['XGBoost', 'Neural Network', 'AdaBoost', 'Naive Bayes', 'KNN', 'SVM', 'Random Forest', 'Logistic Regression']
accuracies = [accuracy_xgb, accuracy_nn, accuracy_adaboost, accuracy_nb, accuracy_knn, accuracy_svm, accuracy_rf, accuracy]

table = PrettyTable()
table.field_names = ["Model", "Accuracy"]

for model, accuracy in zip(models, accuracies):

    colored_model = f'{model}'
    colored_accuracy = f'{accuracy:.2%}'
    table.add_row([colored_model, colored_accuracy])

print(table)