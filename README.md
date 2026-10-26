# Jenis Pembelajaran, Metode, Algoritma, dan Model: Dataset Zomato

**Dataset:** Restaurant Market Analysis: Ratings, Prices (Kaggle), 9.551 restoran Zomato. Yang ditebak adalah kategori rating restoran (`Rating text`): Poor, Average, Good, Very Good, Excellent.

| Komponen | Yang digunakan |
|---|---|
| Jenis pembelajaran | **Supervised learning** |
| Metode | **Klasifikasi** |
| Algoritma | **Random Forest** |
| Model | **Random Forest Classifier** |

---

## 1. Jenis Pembelajaran: Supervised Learning

**Apa:** Supervised learning adalah cara belajar mesin dari data yang **sudah punya jawaban** (label). Model mempelajari hubungan antara ciri-ciri data dan jawabannya, lalu memakai pola itu untuk menebak data baru.

| | Supervised learning | Unsupervised learning |
|---|---|---|
| Data | Punya jawaban (label) | Tidak punya jawaban |
| Tujuan | Menebak jawaban untuk data baru | Mencari kelompok atau pola sendiri |
| Hasil bisa dinilai benar/salah? | Ya, dicocokkan dengan jawaban asli | Tidak ada jawaban untuk dicocokkan |
| Contoh | Klasifikasi, regresi | Clustering (misalnya K-Means) |

**Mengapa memilih supervised learning:**
- Setiap restoran di dataset **sudah punya jawaban**, yaitu kategori rating (`Rating text`).
- Tujuannya adalah **menebak kategori rating tertentu** (Poor sampai Excellent), bukan sekadar mengelompokkan restoran.
- Karena ada jawaban asli, hasil tebakan model **bisa diukur**: berapa yang benar dan berapa yang salah.

---

## 2. Metode: Klasifikasi

**Apa:** Klasifikasi adalah metode untuk menebak **kategori** sebuah data berdasarkan ciri-cirinya. Di sini, yang ditebak adalah kategori rating restoran dari karakteristik restorannya.

**Mengapa memilih klasifikasi:**
- Yang ingin ditebak berupa **kategori** (5 kelas rating), bukan angka.
- Klasifikasi adalah salah satu metode dalam supervised learning. Metode lainnya, regresi, dipakai untuk menebak angka.
- Penelitian sebelumnya pada data Zomato juga memakai klasifikasi dengan kategori rating yang sama, yaitu Poor sampai Excellent (Farid & Fitrianah, 2021).

---

## 3. Algoritma: Random Forest

**Apa:** Random Forest adalah algoritma yang membuat **banyak pohon keputusan** (*decision tree*). Setiap pohon menebak kategori rating, lalu **suara terbanyak** menjadi tebakan akhir.

**Mengapa memilih Random Forest:**
1. **Lebih akurat dan stabil daripada satu pohon keputusan (Decision Tree).** Satu pohon mudah terlalu hafal data latih, sehingga buruk saat menebak data baru. Gabungan banyak pohon mengurangi masalah ini. Beberapa penelitian menunjukkan Random Forest lebih akurat daripada Decision Tree (Riansyah dkk., 2025; Hajar & Nurdin, 2026; Putra dkk., 2025). Hal ini juga terbukti pada dataset ini (lihat bagian 5).
2. **Terbukti baik pada data Zomato.** Random Forest memberi hasil baik pada data Zomato (Jonathan dkk., 2019), dan menjadi yang terbaik dari 7 algoritma yang dibandingkan (Vaidhy dkk., 2025).
3. **Pengembangan dari penelitian sebelumnya.** Farid & Fitrianah (2021) memakai C4.5, yaitu satu pohon keputusan, pada data Zomato. Random Forest mengembangkannya dengan menggabungkan banyak pohon.
4. **Bisa menangani kelas yang tidak seimbang.** Data rating didominasi kelas Average, sedangkan kelas Poor dan Excellent sedikit. Random Forest bisa dipadukan dengan teknik SMOTE untuk mengatasinya (Anjani dkk., 2023).

---

## 4. Model: Random Forest Classifier

**Apa:** Model adalah **hasil training** algoritma Random Forest pada data Zomato. Model inilah yang dipakai untuk menebak kategori rating restoran.

| Keterangan | Isi |
|---|---|
| Model | Random Forest Classifier (library scikit-learn), 200 pohon |
| Input (fitur) | Votes, price range, biaya makan berdua, kota, negara, table booking, online delivery, jenis masakan |
| Output (target) | `Rating text`: Poor, Average, Good, Very Good, Excellent |
| Data latih | Restoran yang sudah punya rating (restoran "Not rated" tidak dipakai) |

**Mengapa memakai model ini:** model ini adalah penerapan langsung dari algoritma Random Forest untuk masalah klasifikasi, sesuai metode dan algoritma yang dipilih di atas.

---

## 5. Perbandingan Algoritma

Untuk membuktikan pilihan Random Forest, tiga algoritma diuji dengan kondisi yang **sama persis**:

| Pengaturan | Isi |
|---|---|
| Data | 7.403 restoran yang punya rating |
| Pembagian data | 80% training (5.922 restoran), 20% testing (1.481 restoran) |
| Fitur | 28 fitur yang sama untuk ketiga algoritma |
| Pengacakan | `random_state = 42` |

Satu-satunya yang berbeda adalah **algoritmanya**:
- **Logistic Regression** sebagai pembanding dasar (baseline), algoritma paling sederhana.
- **Decision Tree**, yaitu satu pohon keputusan.
- **Random Forest**, yaitu 200 pohon keputusan.

### 5.1 Hasil training dan testing

| Algoritma | Akurasi training | Akurasi testing | Tebakan benar (testing) | F1 macro |
|---|---|---|---|---|
| Logistic Regression | 68,1% | 67,0% | 992 / 1.481 | 0,400 |
| Decision Tree | 99,7% | 57,9% | 857 / 1.481 | 0,370 |
| **Random Forest** | 99,7% | 65,7% | 973 / 1.481 | **0,429** |

- **Akurasi training:** model menebak ulang 5.922 restoran training yang **sudah dipelajarinya**, lalu dicocokkan dengan rating aslinya.
- **Akurasi testing:** model menebak 1.481 restoran yang **belum pernah dilihatnya**.

Akurasi testing adalah ukuran kemampuan model yang sebenarnya. Akurasi training dipakai untuk mendeteksi *overfitting* (model terlalu hafal data latih). Makin besar selisih akurasi training dan testing, makin besar overfitting-nya.

### 5.2 Tebakan benar per kategori (data testing)

| Kategori | Jumlah | Logistic Regression | Decision Tree | Random Forest |
|---|---|---|---|---|
| Poor | 37 | 0 | 1 | 1 |
| Average | 748 | 662 | 574 | 633 |
| Good | 420 | 234 | 192 | **241** |
| Very Good | 216 | **90** | 75 | 85 |
| Excellent | 60 | 6 | **15** | 13 |

### 5.3 Uji ulang dengan 5-fold cross validation

Data dibagi dan diuji 5 kali dengan pembagian berbeda, lalu hasilnya dirata-rata, supaya hasilnya tidak kebetulan.

| Algoritma | Akurasi rata-rata | F1 macro rata-rata | F1 macro dengan penanganan kelas tidak seimbang (*class weighting*) |
|---|---|---|---|
| Logistic Regression | 67,4% | 0,416 | 0,422 |
| Decision Tree | 59,8% | 0,404 | 0,405 |
| **Random Forest** | 66,6% | **0,446** | **0,483** |

### 5.4 Cara membaca hasil

1. **Random Forest jauh lebih baik daripada Decision Tree.** Keduanya sama-sama hampir hafal data training (99,7%). Tapi di data testing, Decision Tree turun ke 57,9%, sedangkan Random Forest bertahan di 65,7%. Random Forest menebak benar **116 restoran lebih banyak**. Ini bukti bahwa satu pohon mudah terlalu hafal data latih, dan gabungan banyak pohon mengatasinya.
2. **Akurasi Logistic Regression sedikit lebih tinggi, tapi F1-nya lebih rendah.** Logistic Regression lebih sering menebak kelas mayoritas (Average), sehingga akurasinya terbantu. Tapi kelas kecil hampir tidak tertebak: Excellent hanya benar 6 dari 60, dan Poor 0 dari 37. Karena data rating tidak seimbang, **F1 macro adalah ukuran yang lebih adil**, karena setiap kategori dihitung setara.
3. **Random Forest memiliki F1 macro tertinggi** di semua pengujian. Setelah ditambah penanganan kelas tidak seimbang, keunggulannya makin jelas (0,483, dibanding 0,422 untuk Logistic Regression dan 0,405 untuk Decision Tree).

### 5.5 Kesimpulan perbandingan

Random Forest dipilih karena memberikan **keseimbangan terbaik**: akurasinya hampir setara dengan Logistic Regression, jauh lebih baik daripada Decision Tree, dan paling mampu mengenali semua kategori rating, termasuk kategori yang datanya sedikit.

Kelemahan yang perlu dicatat: kategori **Poor** masih sangat sulit ditebak oleh ketiga algoritma, karena datanya hanya 186 restoran (2,5%).

*Hasil di bagian ini bisa diulang dengan menjalankan file uji_rf_vs_dt.py.*

---

## 6. Sumber

**Dataset**
- srisyra02. *Restaurant Market Analysis: Ratings, Prices* [Dataset]. Kaggle. https://www.kaggle.com/datasets/srisyra02/zomato-market-analysis

**Jurnal**
1. Farid, M., & Fitrianah, D. (2021). Rekomendasi Pemilihan Restoran Berdasarkan Rating Online Menggunakan Algoritma C4.5. *InComTech: Jurnal Telekomunikasi dan Komputer*, 11(1), 9–22. https://publikasi.mercubuana.ac.id/index.php/Incomtech/article/download/9791/4389
2. Jonathan, B., Sihotang, J. I., & Martin, S. (2019). Sentiment Analysis of Customer Reviews in Zomato Bangalore Restaurants Using Random Forest Classifier. *Abstract Proceedings International Scholars Conference*, 7(1), 1719–1728. https://jurnal.unai.edu/index.php/isc/article/view/1003
3. Vaidhy, A., Batham, D., Jain, R., & Manjhwar, A. K. (2025). Machine Learning-Driven Statistical Analysis of Indian Restaurants: Insights from the Zomato Dataset. *Facta Universitatis, Series: Electronics and Energetics*, 38(2), 355–374. https://casopisi.junis.ni.ac.rs/index.php/FUElectEnerg/article/view/13158
4. Riansyah, M., Bahri, S., & Fiqna, H. P. (2025). Perbandingan Akurasi Algoritma Decision Tree dan Random Forest dalam Diagnosa Penyakit Hepatitis. *Indonesian Journal of Science, Technology and Humanities*, 2(3), 261–267. https://jurnal.intekom.id/index.php/ijstech/article/view/1108/899
5. Hajar, S., & Nurdin, N. (2026). Perbandingan Klasifikasi Algoritma Decision Tree dan Random Forest pada Dataset Index Prestasi Semester Mahasiswa (IPS). *JUTECH: Journal Education and Technology*, 7(1), 90–104. https://jurnal.stkippersada.ac.id/jurnal/index.php/jutech/article/download/6625/pdf
6. Putra, R. S., Simbolon, H. F. S., Linhar P, A., & Izhari, F. (2025). Perbandingan Algoritma Decision Tree dan Random Forest dalam Klasifikasi Kepuasan Pengguna Sistem Informasi Akademik. *TECHSI: Jurnal Teknik Informatika*, 16(2), 96–104. https://ojs.unimal.ac.id/techsi/article/view/25799/10818
7. Anjani, A. F., Anggraeni, D., & Tirta, I. M. (2023). Implementasi Random Forest Menggunakan SMOTE untuk Analisis Sentimen Ulasan Aplikasi Sister for Students UNEJ. *Jurnal Nasional Teknologi dan Sistem Informasi (TEKNOSI)*, 9(2), 163–172. https://teknosi.fti.unand.ac.id/index.php/teknosi/article/download/2325/pdf
