# Laporan Proyek Machine Learning - Bizzati Hanif Raushan Fikri

## **Domain Proyek**
Sistem rekomendasi merupakan salah satu cabang kecerdasan buatan (artificial intelligence) yang populer digunakan untuk membantu pengguna menemukan produk atau konten yang relevan berdasarkan preferensi mereka. Dalam proyek ini, topik yang dipilih adalah **“Rekomendasi Anime”**, yaitu merancang sistem yang dapat merekomendasikan judul anime kepada pengguna berdasarkan data historis interaksi (rating) dan/atau konten metadata anime.

## Business Understanding

### Problem Statements
1. Pengguna seringkali kesulitan memilih anime yang sesuai di antara ribuan judul yang tersedia, sehingga mereka membutuhkan rekomendasi yang relevan agar tidak terlalu lama mencari pilihan.
2. Berbagai platform streaming anime ingin meningkatkan kepuasan dan retensi pengguna dengan menampilkan judul anime yang sesuai dengan preferensi masing-masing pengguna.

### Goals
1. Membuat sistem rekomendasi anime yang mampu menghasilkan **Top-10 rekomendasi** untuk setiap pengguna.  
2. Mengukur performa sistem menggunakan metrik **RMSE ≤ 1.00** (untuk kualitas prediksi rating) dan **Precision@10 ≥ 70%** (untuk kualitas Top-N rekomendasi).  
3. Mengimplementasikan dua pendekatan utama:
   - **Content-Based Filtering (CBF)**: Merekomendasikan anime berdasarkan kesamaan metadata (mis. genre, tipe, episodes).  
   - **Collaborative Filtering (CF)**: Merekomendasikan anime berdasarkan interaksi historis (rating) antar pengguna dan anime.

### Solution Statements
**Solution 1. Content-Based Filtering**
- **Ide Dasar**: Setiap anime diwakili oleh vektor fitur yang memuat metadata (genre, tipe, jumlah episode). Sistem merekomendasikan anime yang mirip dengan anime yang pernah disukai atau diminati oleh pengguna, berdasarkan kemiripan vektor fitur (mis. Cosine Similarity).

**Solution 2 Collaborative Filtering**  
- **Ide Dasar**: Sistem mempelajari pola rating antar pengguna dan anime untuk memprediksi rating yang belum pernah diberikan oleh pengguna tertentu. Dua sub-kategori:
- **Neural Collaborative Filtering (NCF)**: Memetakan setiap user/anime ke vektor laten (embedding), lalu memprediksi rating via dot product atau lapisan terhubung (_dense_) di jaringan.

## Data Understanding
Dataset yang digunakan diambil dari website database kaggle dengan tautan: [Kaggle Anime Recommendation](https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database?select=anime.csv)

Dataset yang digunakan berisikan informasi preferensi user dengan total 73.516 user terhadap 12.294 anime, data tersebut diambil dari website [myanimelist.net](https://myanimelist.net/) yang merupakan website katalog online yang berisi informasi tentang anime dan manga, yang memungkinkan pengguna untuk membuat daftar, menilai, dan membagikan pendapat mereka tentang anime dan manga.

### **Deskripsi Fitur**
Terdapat 2 file .csv pada dataset ini yaitu `anime.csv` & `rating.csv` dengan isi awal seperti di bawah;

**Anime.csv:**

- `anime_id` : ID unik dari tiap anime pada myanimelist.net. (numerik)
- `name` : nama dari seri anime. (objek)
- `genre` : list genre dari anime tersebut terpisah dengan koma. (objek)
- `type` : tipe anime; movie, seri TV, seri OVA, dll. (objek)
- `episodes` : Jumlah episode dari seri tersebut. (1 jika movie & "Unknown" jika ongoing). (objek)
- `rating` : rata-rata rating per 10 dari anime tersebut. (numerik)
- `members` : Jumlah member dari komunitas yang berada pada "group" anime tersebut. (numerik)

**Rating.csv**

- `user_id` : identifikasi unik user. (numerik)
- `anime_id` : ID unik dari anime yang dirating oleh user. (numerik)
- `rating` : rating dari 10 yang diberikan user (-1 jika user menonton namun tidak memberi rating). (numerik)

### **Kondisi Data**
Jumlah data awal yaitu 12.294 data entri pada `anime.csv` dan 7.813.736 data entri pada `rating.csv`.
- Anime.csv:
  - **Tipe data kurang tepat:** Jumlah episode (`episodes`) dalam bentuk objek sedangkan seharusnya numerik, namun setelah validasi ulang memang terdapat seri anime yang belum tamat dan diberi penjelasan `unknown`. Untuk itu data akan tetap ditransformasi namun data NULL setelahnya akan dibiarkan.
  - **Missing Values:** Terdapat sejumlah baris dengan Missing Values; 62 pada kolom `genre`, 25 pada kolom `type`, dan 230 pada kolom `rating`. Hal ini akan ditangani dengan penghapusan baris demi kelancaran dalam training model
  - **Data Duplikat:** Tidak ditemukan baris duplikat pada `Anime.csv`

- Rating.csv:
  - **Tipe data salah:** Tipe data dari tiap kolom sudah sesuai
  - **Missing Values:** Tidak terdapat missing values yang ditemukan pada `Rating.csv`.
  - **Data Duplikat:** Terdapat 1 baris duplikat yang akan ditangani untuk mengurangi bias pada persiapan data.

### Exploratory Data Analysis (EDA)
Untuk memahami lebih lanjut mengenai statistika dan distribusi data, dilakukan pengecekan pada masing-masing variabel dengan penemuan bahwa:
- Terdapat 73.515 jumlah pengguna unik
- Terdapat 12.294 jumlah judul anime unik
- Terdapat 6 tipe penayangan anime berbeda
- Terdapat 43 genre anime unik dalam dataset
- Terdapat 6.337.241 baris data di mana user memberikan rating terhadap anime yang ditonton. Sebaliknya, 1.476.496 baris di mana user tidak memberikan rating terhadap anime yang ditonton (`rating`= -1)

## Data Preparation
Maksud dari tahap ini adalah untuk membersihkan dan mempersiapkan dataset sebelum digunakan untuk modeling demi mengantisipasi error seperti invalid value dan sebagainya. Tahap-tahap yang dilakukan adalah:

1. **Menghapus data Missing Values:** Sebagaimana hasil pengecekan kondisi data, terdapat sejumlah baris dengan data NULL pada dataset `anime.csv` yang perlu dihapus.

2. **Menghapus data duplikat:** Sebagaimana hasil pengecekan kondisi awal data, terdapat 1 baris yang duplikat pada dataset `rating.csv` yang perlu dihapus.

3. **Koreksi tipe data:** Tipe data jumlah episode (`episodes`) pada `anime.csv` saat ini adalah objek, sementara isi dari kolom tersebut terbatas pada angka (numerik). Maka itu tipe data `episodes` akan diubah ke integer. Ini menghasilkan 187 data NULL pada kolom `episodes` karena data tersebut awalnya string "Unknown" alias masih dalam produksi, data NULL ini tidak ditindaklanjuti berhubung tidak relevan dengan pembangunan model nantinya.

4. **Hapus data kurang relevan:**
   - **data dengan skor `rating` -1 pada `rating.csv`:** Sumber dataset mengatakan bahwa nilai -1 pada data `rating` dari `rating.csv` menandakan user tidak melakukan rating terhadap anime,    alias data tersebut tidak memberikan sentimen pengguna & tidak cocok untuk membangun model

6. **Sampling data:** Jumlah data pada `rating.csv` terlalu banyak secara total (7.813.736 entri) sehingga demi mempersingkat waktu pemodelan dan beban komputasi, data akan disampling hingga 500.000 baris data saja

7. **Persiapan untuk content based filtering (CBF):**
   - konversi data ke vektor TF-IDF lalu menerapkan cosine similarity untuk menghitung kemiripan berdasarkan `genre`anime

9. **Persiapan untuk collaborative filtering (CF):**
    - Encode user_id dan anime_id menjadi indeks integer berurutan
    - **Split data ke data training & testing:** split data ke dua bagian yaitu 80% untuk pelatihan dan 20% untuk pengujian

## Modeling

### Model 1: Content Based Filtering (CBF)

#### Cara kerja

pertama kita vektorkan metadata anime (genre di‐one‐hot, tipe di‐one‐hot, dan jumlah episode dinormalisasi); kemudian membangun matriks fitur item `(n_anime × n_features)` dan menghitung Cosine Similarity antar baris (anime). Untuk merekomendasikan, misalnya, `recommend_by_title("Fullmetal Alchemist", df, cos_sim, top_n=10)` langsung mengurutkan skor similarity terhadap anime acuan dan menampilkan Top-10 judul teratas.  


#### Contoh output top-10 rekomendasi
1. **Contoh rekomendasi CBF berdasarkan input judul**
   
   Saat menjalankan:
   ```python
   recommend_by_title("Fullmetal Alchemist", clean_anime, cos_sim, top_n=10)
   ```
   Hasil output:
   ```
   Rekomendasi untuk 'Fullmetal Alchemist':
   ====================================================================================================
   No. Judul Anime                                       Similarity     Genre Match
   ----------------------------------------------------------------------------------------------------
   1   Fullmetal Alchemist: The Sacred Star of Milos     1.0000          ✓
   2   Fullmetal Alchemist: Brotherhood                  0.9735          ✓
   3   Fullmetal Alchemist: Brotherhood Specials         0.9302          ✓
   4   Tales of Vesperia: The First Strike               0.8515          ✓
   5   Fullmetal Alchemist: Reflections                  0.8511          ✓
   6   Tide-Line Blue                                    0.8108          ✓
   7   Fairy Tail (2014)                                 0.7975          ✓
   8   Fairy Tail                                        0.7975          ✓
   9   Fairy Tail Movie 1: Houou no Miko                 0.7975          ✓
   10  Fairy Tail x Rave                                 0.7975          ✓
   ====================================================================================================
   ```
2. **Contoh rekomendasi CBF berdasarkan input genre**

   Saat menjalankan:
   ```python
   print(recommend_by_genre('Action', clean_anime, cosine_sim))
   ```
   Hasil outut:
   ```
   Rekomendasi untuk genre: 'action':
   ====================================================================================================
   No. Judul Anime                                       Similarity     Genre Match
   ----------------------------------------------------------------------------------------------------
   1   Fullmetal Alchemist                               0.9735          ✓
   2   Fullmetal Alchemist: The Sacred Star of Milos     0.9735          ✓
   3   Fullmetal Alchemist: Brotherhood Specials         0.9555          ✓
   4   Tales of Vesperia: The First Strike               0.8747          ✓
   5   Tide-Line Blue                                    0.8329          ✓
   6   Fullmetal Alchemist: Reflections                  0.8112          ✓
   7   Magi: The Kingdom of Magic                        0.7848          ✓
   8   Magi: The Labyrinth of Magic                      0.7848          ✓
   9   Magi: Sinbad no Bouken (TV)                       0.7848          ✓
   10  Magi: Sinbad no Bouken                            0.7848          ✓
   ====================================================================================================
   ```
   
#### Kelebihan
- Mudah diimplementasikan, hanya memerlukan metadata anime.
- Interpretabilitas tinggi (bisa dijelaskan: “karena genre sama”).
- Tidak memerlukan data interaksi pengguna yang besar.

#### Kekurangan                                                                           
- **Cold-Start Item**: Jika ada anime baru tanpa metadata lengkap, sulit direkomendasikan.
- **Filter Bubble**: Rekomendasi cenderung “anak” dalam genre yang sama, kurang beragam.
- Kualitas rekomendasi sangat tergantung kelengkapan dan akurasi metadata. 

### Model 2: Collaborative Filtering (Neural CF)

#### Cara kerja
- Rating pengguna di-encode ke `user_idx`/`anime_idx` (rentang 0…n−1) dengan `LabelEncoder`, lalu dibagi menjadi train/test.  
- Model Neural CF dibangun dengan dua layer embedding (untuk user dan anime), di-flatten, digabung ke satu vektor, lalu dilanjutkan ke dua dense layer (128→64) dengan Dropout dan output linear untuk regresi rating.  
- Kompilasi model menggunakan `loss='mse'` dan metrik `rmse`.  
- Latih model dengan EarlyStopping (_monitor='val_loss', patience=5_) dan ReduceLROnPlateau (_monitor='val_loss', factor=0.2, patience=3_).

#### Contoh output top-10 rekomendasi
- Misal kita ingin membuat rekomendasi anime untuk dengan ID 49503:
   ```python
   recommendations_df = recommend_for_user(49503, model, clean_anime)
   ```
   Hasil outut:
   ```
   Rekomendasi untuk User 49503:
   ====================================================================================================
   No.  Judul Anime                                        Predicted  Global   Genre
   ----------------------------------------------------------------------------------------------------
   1    Ginga Eiyuu Densetsu                               9.3472     9.11     Drama, Military, Sci-Fi, Space
   2    Gintama°                                           9.2306     9.25     Action, Comedy, Historical, Parody, Samurai, Sci-Fi, Shounen
   3    Haikyuu!!: Karasuno Koukou VS Shiratorizawa Gak... 9.0693     9.15     Comedy, Drama, School, Shounen, Sports
   4    Fullmetal Alchemist: Brotherhood                   8.9858     9.26     Action, Adventure, Drama, Fantasy, Magic, Military, Shounen
   5    Gintama&#039;: Enchousen                           8.8982     9.11     Action, Comedy, Historical, Parody, Samurai, Sci-Fi, Shounen
   6    Qin Shiming Yue Zhi: Zhu Zi Bai Jia                8.8745     7.49     Action, Fantasy, Historical, Martial Arts
   7    Gintama Movie: Kanketsu-hen - Yorozuya yo Eien ... 8.8731     9.10     Action, Comedy, Historical, Parody, Samurai, Sci-Fi, Shounen
   8    Gintama                                            8.8293     9.04     Action, Comedy, Historical, Parody, Samurai, Sci-Fi, Shounen
   9    Haikyuu!! Second Season                            8.8183     8.93     Comedy, Drama, School, Shounen, Sports
   10   Hellsing Ultimate                                  8.8055     8.59     Action, Horror, Military, Seinen, Supernatural, Vampire
   ====================================================================================================
   ```

#### Kelebihan
- **Interaksi Non-Linier:** Model Neural CF dapat mempelajari pola interaksi non-linier antara preferensi pengguna dan karakteristik anime, sehingga seringkali menghasilkan prediksi rating yang lebih akurat daripada metode sederhana (misalnya dot-product murni).

- **Scalability:** Embedding user dan anime direpresentasikan sebagai vektor laten. Setelah di-train, inferensi prediksi untuk satu pasangan (user, anime) hanya memerlukan satu forward pass pada jaringan, sehingga dapat melayani permintaan real-time lebih cepat.

- **Recover Bias dan Regularisasi:** Dengan L2-regularization pada layer embedding dan Dropout di lapisan dense, model mampu mengurangi overfitting, khususnya saat jumlah parameter (bobot) menjadi besar.

#### Kekurangan
- **Kebutuhan Data Besar:** Neural CF memerlukan jumlah data rating yang relatif besar agar embedding “belajar” pola preferensi dengan baik. Jika data interaksi sedikit, embedding cenderung tidak representatif.

- **Komputasi Lebih Berat:** Proses training melibatkan banyak parameter (embedding untuk ribuan user/anime + lapisan dense), sehingga memerlukan GPU atau waktu training yang lebih lama dibanding metode memory-based kNN.

- **Cold-Start Problem:** Model ini tidak dapat memprediksi rating untuk user atau anime yang belum pernah muncul di data training (karena embedding-nya tidak ada). Untuk mengatasi, perlu digabung dengan metode CBF atau hybrid.

## Evaluation
Tahap ini adalah tahap penilaian efektifitas dan akurasi dari model-model. Kedua model menggunakan metrik evaluasi yang berbeda menyesuaikan fungsi dan output akhir/ tujuan model tersebut.
### Model 1: Content Based Filtering (CBF)
#### Metrik Evaluasi
**Precision at K=10** adalah metrik yang digunakan untuk menilai ketepatan sejumlah K rekomendasi yang diberikan. Pada kasus proyek, metrik bekerja dengan menilai persentase anime di dalam Top-10 rekomendasi yang benar-benar relevan (berdasar genre/metadata) jika dibandingkan dengan data ground truth, dalam hal ini dianggap “relevan” bila anime tersebut share minimal satu genre dengan anime acuan.

**Formula:**

$$
\text{Precision at K} = \frac{\text{Jumlah anime relevan dalam top-}K}{K}
$$

#### Hasil
Selama seluruh output top-k memiliki setidaknya satu fitur yang sama dengan target acuan, maka nilai precision at K akan selalu 100%. Ini adalah hasil yang telah dikonfirmasi juga pada tahap pemanggilan fungsi baik input nama anime atau genre di mana kolom genre menunjukkan tanda centang '✓' ketika terdapat setidaknya satu genre yang sama. Model yang dihasilkan sudah sangat akurat dalam pemilihan rekomendasi berdasarkan kemiripan fitur `genre`

### Model 2: Collaborative Filtering (CF)
#### Metrik Evaluasi
**Root Mean Squared Error (RMSE)** adalah metrik utama yang digunakan untuk mengukur rata-rata perbedaan nilai prediksi dengan nilai aktual. Semakin mendekati 0 semakin bagus

**Formula:**

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)^2, 
\quad
\text{RMSE} = \sqrt{\text{MSE}}
$$

di mana  
- $$\(\hat{y}_i\)$$ = rating yang diprediksi oleh model,  
- $$\(y_i\)$$ = rating asli,  
- $$\(n\)$$ = jumlah sampel pada data uji.

#### Hasil
Berikut cuplikan RMSE selama pelatihan model Neural CF pada data uji:

| Epoch | Training RMSE | Validation RMSE |
|:-----:|:-------------:|:---------------:|
| 1     | 4.4505        | 1.3451          |
| 2     | 1.4925        | 1.2964          |
| 3     | 1.4232        | 1.2865          |
| 4     | 1.4013        | 1.2856          |
| 5     | 1.3845        | 1.2855          |
| **6** | **1.3668**    | **1.2795**      |
| 7     | 1.3414        | 1.2864          |
| 8     | 1.3241        | 1.2987          |
| 9     | 1.3055        | 1.2918          |
| 10    | ...           | ...             |

Epoch berangsur hingga epoch 11 sebelum berhenti dengan early stopping dan melakukan callback ke epoch 6 sebagai bestmodel-nya, dari pelatihan model tersebut juga dapat dilihat bahwa hasil best model dipilih berdasarkan skor Validation RMSE yang paling rendah dengan skor 1,2794. Ini memang bukanlah yang terbaik dan goal untuk mencapai skor RMSE < 1.0 belum dapat terpenuhi sayangnya.

## Conclusion

Dengan hasil analisis sebelumnya dapat disimpulkan beberapa hal terutama untuk menjawab *business undestanding* di awal:
1. **Goal belum sepenuhnya terpenuhi:** skor RMSE untuk mengukur performa solusi model Neural Collaborative Filter masih belum optimal (RMSE < 1.0). Disamping itu target seperti Precision at K bernilai >75% sudah tercapai dengan skor yang sangat memadai
2. **Solusi Content Based Filtering:** pendekatan CBF cocok untuk digunakan ketika ingin merekomendasikan objek dengan fitur eksplisit seperti `genre`.
3. **Solusi Collaborative Filtering:** pendekatan CF lebih cocok untuk digunakan ketika ingin merekomendasikan objek secara lebih personal dari feedback user terhadap produk/jasa, tentu kelemahannya adalah model tidak bisa diterapkan tanpa adanya bentuk feedback awal dari user.
