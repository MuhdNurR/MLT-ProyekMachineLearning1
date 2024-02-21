# Laporan Proyek Machine Learning - Muhammad Nur Rachman Nidhi Suryono

Membuat Model Predictive Analytics. Dalam proyek pertama ini, Anda akan mengimplementasikan apa yang telah Anda pelajari di seluruh modul untuk membuat model machine learning dan menulis laporan proyek. 

## Domain proyek
![Hearth!](https://d2jx2rerrg6sh3.cloudfront.net/image-handler/picture/2021/1/shutterstock_1576424071.jpg "Hearth")

### Latar Belakang
Penyakit jantung merupakan salah satu masalah kesehatan utama di dunia. Menurut data terbaru dari World Health Organization (WHO) pada tahun 2021, penyakit jantung iskemik adalah penyebab kematian nomor satu di dunia, merenggut sekitar 9 juta jiwa setiap tahun. Di Indonesia sendiri, penyakit jantung menempati posisi kedua sebagai penyebab kematian terbanyak setelah stroke.

Deteksi dini dan pencegahan risiko serangan jantung sangat penting untuk menyelamatkan jiwa dan meningkatkan kualitas hidup pasien. Data diagnostik yang kaya, seperti data elektrokardiogram (EKG), data tekanan darah, dan riwayat kesehatan pasien, dapat membantu dokter dalam menilai risiko pasien dan membuat keputusan pengobatan yang tepat.

### Peran Pembelajaran Mesin

Pembelajaran mesin (ML) dapat memainkan peran penting dalam meningkatkan praktik medis dalam mendeteksi dan mencegah risiko serangan jantung. Berikut beberapa contohnya:

* Analisis data diagnostik: ML dapat digunakan untuk menganalisis data diagnostik, seperti EKG dan data tekanan darah, untuk mengidentifikasi pasien yang berisiko tinggi terkena serangan jantung.
* Pengembangan model prediksi: ML dapat digunakan untuk mengembangkan model prediksi yang dapat memperkirakan kemungkinan pasien terkena serangan jantung di masa depan.
* Personalisasi pengobatan: ML dapat digunakan untuk mempersonalisasi pengobatan untuk pasien dengan penyakit jantung, dengan mempertimbangkan faktor-faktor risiko individu pasien.
* Pengembangan alat skrining: ML dapat digunakan untuk mengembangkan alat skrining yang dapat membantu mengidentifikasi pasien yang berisiko tinggi terkena penyakit jantung, sehingga mereka dapat menerima pengobatan pencegahan yang tepat.

## Business Understanding
Industri kesehatan membutuhkan alat dan teknik yang canggih untuk menganalisis data pasien secara efektif dan akurat. Rumah sakit, klinik, dan penyedia layanan kesehatan lainnya dapat memanfaatkan model pembelajaran mesin untuk memprediksi risiko serangan jantung pada pasien berdasarkan data medis mereka. Ini berpotensi:
* Mengurangi biaya perawatan kesehatan dengan mengidentifikasi pasien berisiko tinggi dan memberikan intervensi pencegahan dini.
* Meningkatkan akurasi diagnosis dan keputusan pengobatan.
* Meningkatkan kepuasan pasien dengan perawatan yang lebih personal dan tepat sasaran.

### Problem Statements (pernyataan masalah)
### Goals (tujuan)
* Mengembangkan model pembelajaran mesin yang akurat untuk memprediksi risiko serangan jantung pada pasien.
* Mengidentifikasi faktor risiko kunci yang berkontribusi terhadap peningkatan risiko serangan jantung.
* Mensegmentasi pasien ke dalam kelompok risiko yang berbeda untuk mengalokasikan sumber daya perawatan secara lebih efektif.
* Menyediakan wawasan yang dapat ditindaklanjuti kepada dokter dan penyedia layanan kesehatan lainnya untuk membantu mereka dalam membuat keputusan pengobatan yang lebih baik.

## Data Understanding
### Sumber Data
https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset/data

### Variabel
| Label | Description | Data Type | Values/Ranges | Missing Values Allowed? |
|---|---|---|---|---|
| Age | Age of the patient | Integer | Any positive integer | No |
| Sex | Sex of the patient | Categorical | Male, Female | No |
| exang | Exercise-induced angina (yes/no) | Binary | 1 (yes), 0 (no) | No |
| ca | Number of major vessels (0-3) | Integer | 0, 1, 2, 3 | No |
| cp | Chest pain type | Categorical | 1 (typical angina), 2 (atypical angina), 3 (non-anginal pain), 4 (asymptomatic) | No |
| trtbps | Resting blood pressure (mm Hg) | Continuous | Any positive integer | No |
| chol | Cholesterol (mg/dl) | Continuous | Any positive integer | No |
| fbs | Fasting blood sugar (>120 mg/dl) | Binary | 1 (true), 0 (false) | No |
| rest_ecg | Resting electrocardiogram results | Categorical | 0 (normal), 1 (abnormality), 2 (hypertrophy) | No |
| thalach | Maximum heart rate achieved | Continuous | Any positive integer | No |
| target | Risk of heart attack (low/high) | Binary | 0 (low chance), 1 (high chance) | No |

## Data Preparation
### Splitting Dataset
Train test split adalah proses membagi data menjadi data train dan test. Data trainakan digunakan untuk membangun model, sedangkan data test akan digunakan untuk menguji performa model. Pada proyek ini dataset sebesar dibagi menjadi 416 untuk data train dan 105 untuk data uji.

### Proses Mengatasi Outlier
Outlier adalah data yang memiliki nilai yang sangat jauh dari nilai rata-rata data lainnya. Outlier dapat menyebabkan model menjadi tidak akurat. Berikut beberapa cara untuk mengatasi outlier:

* Membuang outlier: Cara ini dapat dilakukan jika outlier hanya sedikit dan tidak mewakili populasi data.
* Mengubah nilai outlier: Cara ini dapat dilakukan dengan mengganti nilai outlier dengan nilai rata-rata, median, atau nilai yang terdekat dengan nilai outlier.
* Menormalisasi data: Cara ini dapat dilakukan dengan mengubah skala data sehingga outlier tidak terlalu berpengaruh pada model.
* Proses Mengatasi Missing Value

### Proses Mengatasi Missing value
Missing value adalah data yang hilang. Missing value dapat menyebabkan model menjadi tidak akurat. Berikut beberapa cara untuk mengatasi missing value:

* Menghapus data yang memiliki missing value: Cara ini dapat dilakukan jika data yang memiliki missing value hanya sedikit.
* Mengisi missing value dengan nilai rata-rata, median, atau nilai yang terdekat: Cara ini dapat dilakukan jika missing value tidak terlalu banyak dan data terdistribusi secara normal.
* Menggunakan metode imputasi: Ada beberapa metode imputasi yang dapat digunakan untuk mengisi missing value, seperti metode K-Nearest Neighbors (KNN), metode Expectation-Maximization (EM), dan metode Multiple Imputation.

### Algoritma Yang Dipakai
1. XGBoost
XGBoost adalah algoritma ensemble learning yang menggabungkan beberapa decision tree. XGBoost dikenal dengan kemampuannya yang handal dalam berbagai masalah klasifikasi dan regresi. XGBoost terkenal dengan performanya yang baik dan sering menjadi pilihan utama dalam berbagai kompetisi machine learning.

2. Neural Network
Neural Network adalah algoritma yang terinspirasi dari struktur otak manusia. Neural Network terdiri dari beberapa layer neuron yang saling terhubung. Neural Network mampu belajar dari data dan membuat prediksi dengan cara yang kompleks dan non-linear. Neural Network sangat populer untuk berbagai tugas, termasuk klasifikasi gambar, pengenalan suara, dan pemrosesan bahasa alami.

3. AdaBoost
AdaBoost adalah algoritma ensemble learning yang menggabungkan beberapa weak learner. AdaBoost bekerja dengan memberikan bobot yang lebih besar kepada weak learner yang berkinerja baik pada data training. AdaBoost dikenal dengan kemampuannya untuk meningkatkan performa model klasifikasi dengan menggabungkan beberapa model yang sederhana.

4. Naive Bayes
Naive Bayes adalah algoritma klasifikasi yang didasarkan pada teorema Bayes. Naive Bayes mengasumsikan bahwa semua fitur data bersifat independen satu sama lain. Naive Bayes adalah algoritma yang sederhana dan mudah dipahami, tetapi performanya bisa kurang optimal dibandingkan dengan algoritma lain pada beberapa kasus.

5. KNN (K-Nearest Neighbors)
KNN adalah algoritma klasifikasi yang bekerja dengan mencari K data terdekat dengan data yang ingin diklasifikasikan. KNN kemudian akan mengklasifikasikan data berdasarkan kelas mayoritas dari K data terdekat tersebut. KNN adalah algoritma yang mudah dipahami dan diimplementasikan, tetapi performanya bisa sensitif terhadap nilai K yang dipilih.

6. SVM (Support Vector Machine)
SVM adalah algoritma klasifikasi yang bekerja dengan mencari hyperplane yang memisahkan dua kelas data dengan jarak yang paling besar. SVM dikenal dengan kemampuannya yang handal dalam menangani data dengan dimensi yang tinggi.

7. Random Forest
Random Forest adalah algoritma ensemble learning yang menggabungkan beberapa decision tree. Random Forest bekerja dengan membangun beberapa decision tree dengan menggunakan subset data yang berbeda. Random Forest dikenal dengan kemampuannya yang handal dalam menangani data dengan noise dan outlier.

8. Logistic Regression
Logistic Regression adalah algoritma klasifikasi yang digunakan untuk memprediksi probabilitas suatu data。 Logistic Regression adalah algoritma yang sederhana dan mudah dipahami, tetapi performanya bisa kurang optimal dibandingkan dengan algoritma lain pada beberapa kasus.

# Modeling
Beberapa jenis algoritma Machine Learning yang akan ditest:

* XGBoost: Algoritma boosting yang populer untuk klasifikasi dan regresi dengan performa tinggi dan efisiensi komputasi.
* Neural Network: Algoritma yang terinspirasi dari struktur otak manusia, mampu mempelajari pola kompleks dan membuat prediksi akurat, seperti convolutional neural networks (CNN) untuk pengenalan gambar dan recurrent neural networks (RNN) untuk pemrosesan bahasa alami.
* AdaBoost: Algoritma boosting yang menggabungkan beberapa model klasifikasi lemah untuk menghasilkan model yang lebih kuat.
* Naive Bayes: Algoritma klasifikasi sederhana dan mudah dipahami berdasarkan teorema Bayes, sering digunakan dalam aplikasi teks dan email spam filtering.
* KNN (K-Nearest Neighbors): Algoritma klasifikasi dan regresi sederhana yang memprediksi label data berdasarkan K data terdekat.
* SVM (Support Vector Machine): Algoritma klasifikasi dan regresi yang kuat dengan margin maksimal, mampu bekerja dengan data dimensi tinggi dan non-linear, sering digunakan untuk klasifikasi gambar dan teks.
* Random Forest: Algoritma ensemble learning yang menggabungkan beberapa decision tree untuk menghasilkan prediksi yang lebih akurat, mampu menangani overfitting dan meningkatkan stabilitas model.
* Logistic Regression: Algoritma klasifikasi probabilistik untuk memprediksi probabilitas suatu kejadian, sederhana dan mudah dipahami, sering digunakan dalam klasifikasi spam, prediksi risiko kredit, dan klasifikasi sentiment.

# Evaluation
Akurasi:

Akurasi mengukur seberapa dekat hasil prediksi model dengan nilai sebenarnya. Nilai akurasi dihitung dengan membagi jumlah prediksi yang benar dengan total data yang diuji (y_test). Semakin tinggi nilai akurasi (mendekati 1), semakin baik performa model dalam memprediksi nilai yang sebenarnya.

|        Model        | Accuracy |
|---------------------|----------|
|       XGBoost       |  91.43%  |
|    Neural Network   |  88.57%  |
|       AdaBoost      |  90.48%  |
|     Naive Bayes     |  84.76%  |
|         KNN         |  87.62%  |
|         SVM         |  88.57%  |
|    Random Forest    |  94.29%  |
| Logistic Regression |  88.57%  |

Mean Squared Error (MSE):

MSE mengukur rata-rata error kuadrat dari hasil prediksi model. Nilai MSE dihitung dengan menghitung rata-rata dari selisih kuadrat antara hasil aktual dan hasil prediksi. Semakin kecil nilai MSE (mendekati 0), semakin kecil error model dan semakin baik performa model dalam memprediksi nilai yang sebenarnya.

Formula MSE:


![MSE!](https://miro.medium.com/v2/resize:fit:700/1*BtVajQNj29LkVySEWR_4ww.png "MSE")

|	|train|	test|
|-----|-----|----|
|XGBoost	|0.175481	|0.219048|
|Neural Network	|0.365385|	0.390476|
|AdaBoost	|0.242788	|0.285714|
|Naive Bayes	|0.627404	|0.619048|
|KNN	|0.382212	|0.419048|
|SVM	|0.634615	|0.609524|
|Random Forest	|NaN |	NaN|
|Logistic Regression |	0.620192	|0.590476
RF	|0.401442|	0.4 | 
