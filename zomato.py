"""
MENGHITUNG AKURASI TRAINING DAN TESTING - RANDOM FOREST (DATASET ZOMATO)
========================================================================
Langkah:
  1. Menyiapkan data (X = ciri restoran, y = Rating text)
  2. Membagi data 80% training, 20% testing
  3. Melatih Random Forest
  4. Akurasi TRAINING  -> model menebak ulang data yang sudah dipelajarinya
  5. Akurasi TESTING   -> model menebak data yang belum pernah dilihatnya
  6. Membandingkan training vs testing (deteksi overfitting)
  7. Membandingkan 3 algoritma (Logistic Regression, Decision Tree, Random Forest)
  8. Menyimpan hasil tebakan per restoran (training & testing) ke Excel

Cara menjalankan:
  pip install pandas numpy scikit-learn openpyxl
  python hitung_akurasi_training_testing.py
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

FILE_CSV = "Zomato_Dataset.csv"   # ganti sesuai nama file Anda
KATEGORI = ["Poor", "Average", "Good", "Very Good", "Excellent"]


def garis(judul):
    print("\n" + "=" * 72)
    print(judul)
    print("=" * 72)


def hitung_akurasi(y_asli, y_tebakan, nama_data):
    """Cocokkan tebakan dengan rating asli, lalu hitung benar / total."""
    benar = int((np.asarray(y_asli) == np.asarray(y_tebakan)).sum())
    total = len(y_asli)
    print(f"Akurasi {nama_data:8s} = {benar} benar / {total} restoran = {benar / total:.1%}")
    return benar, total


def benar_per_kategori(y_asli, y_tebakan):
    """Tabel: berapa restoran per kategori dan berapa yang ditebak benar."""
    y_asli, y_tebakan = np.asarray(y_asli), np.asarray(y_tebakan)
    baris = []
    for k in KATEGORI:
        mask = y_asli == k
        n, b = int(mask.sum()), int((y_tebakan[mask] == k).sum())
        baris.append({"Kategori": k, "Jumlah": n, "Benar": b, "Persen benar": f"{b / n:.1%}"})
    return pd.DataFrame(baris)


# ----------------------------------------------------------------------
# 1. MENYIAPKAN DATA
# ----------------------------------------------------------------------
garis("1. MENYIAPKAN DATA")
df = pd.read_csv(FILE_CSV, encoding="utf-8-sig")
print(f"Data awal                 : {len(df)} restoran")
df = df[df["Rating text"] != "Not rated"].copy()       # tidak punya kunci jawaban
df["Cuisines"] = df["Cuisines"].fillna("")
print(f"Setelah buang 'Not rated' : {len(df)} restoran")

# X = ciri restoran (28 fitur). Aggregate rating & Rating color tidak dipakai (bocoran jawaban)
X = pd.DataFrame(index=df.index)
X["price_range"] = df["Price range"]
X["votes_log"] = np.log1p(df["Votes"])
X["table_booking"] = (df["Has Table booking"] == "Yes").astype(int)
X["online_delivery"] = (df["Has Online delivery"] == "Yes").astype(int)
X["india"] = (df["Country Code"] == 1).astype(int)
X["biaya_persentil"] = df.groupby("Currency")["Average Cost for two"].rank(pct=True)
X["jml_jenis_masakan"] = df["Cuisines"].str.count(",") + (df["Cuisines"] != "").astype(int)
X["jml_restoran_kota"] = df["City"].map(df["City"].value_counts())
for m in df["Cuisines"].str.split(",").explode().str.strip().value_counts().head(20).index:
    X["masakan_" + m] = df["Cuisines"].str.contains(m, regex=False).astype(int)

# y = kunci jawaban
y = df["Rating text"]
print(f"Jumlah fitur              : {X.shape[1]}")

# ----------------------------------------------------------------------
# 2. MEMBAGI DATA 80% TRAINING, 20% TESTING
# ----------------------------------------------------------------------
garis("2. MEMBAGI DATA")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)
print(f"Training : {len(X_train)} restoran (punya rating, dipakai untuk belajar)")
print(f"Testing  : {len(X_test)} restoran (punya rating, tapi disembunyikan dari model)")

# ----------------------------------------------------------------------
# 3. MELATIH RANDOM FOREST
# ----------------------------------------------------------------------
garis("3. MELATIH RANDOM FOREST (200 pohon)")
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)          # model belajar dari X_train DAN y_train
print("Training selesai.")

# ----------------------------------------------------------------------
# 4. AKURASI TRAINING
#    Model menebak ulang 5.922 restoran yang SUDAH dipelajarinya
# ----------------------------------------------------------------------
garis("4. AKURASI TRAINING (menebak ulang data yang sudah dipelajari)")
tebakan_train = rf.predict(X_train)
benar_train, total_train = hitung_akurasi(y_train, tebakan_train, "training")
print("\nBenar per kategori (training):")
print(benar_per_kategori(y_train, tebakan_train).to_string(index=False))

# ----------------------------------------------------------------------
# 5. AKURASI TESTING
#    Model menebak 1.481 restoran yang BELUM pernah dilihatnya
# ----------------------------------------------------------------------
garis("5. AKURASI TESTING (menebak data yang belum pernah dilihat)")
tebakan_test = rf.predict(X_test)
benar_test, total_test = hitung_akurasi(y_test, tebakan_test, "testing")
print("\nBenar per kategori (testing):")
print(benar_per_kategori(y_test, tebakan_test).to_string(index=False))

# ----------------------------------------------------------------------
# 6. TRAINING VS TESTING (deteksi overfitting)
# ----------------------------------------------------------------------
garis("6. TRAINING VS TESTING")
acc_train, acc_test = benar_train / total_train, benar_test / total_test
print(f"Akurasi training : {acc_train:.1%}")
print(f"Akurasi testing  : {acc_test:.1%}")
print(f"Selisih          : {(acc_train - acc_test) * 100:.1f} poin")
print("Selisih besar = model hafal data training tapi kurang mampu di data baru (overfitting).")
print("Yang dilaporkan sebagai kemampuan model adalah AKURASI TESTING.")

# ----------------------------------------------------------------------
# 7. PERBANDINGAN 3 ALGORITMA (kondisi sama persis)
# ----------------------------------------------------------------------
garis("7. PERBANDINGAN 3 ALGORITMA")
algoritma = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": rf,                      # sudah dilatih di langkah 3
}
hasil_algoritma = []
for nama, model in algoritma.items():
    if nama != "Random Forest":
        model.fit(X_train, y_train)
    p_train, p_test = model.predict(X_train), model.predict(X_test)
    hasil_algoritma.append({
        "Algoritma": nama,
        "Benar training": f"{(p_train == y_train.values).sum()} / {len(y_train)}",
        "Akurasi training": f"{accuracy_score(y_train, p_train):.1%}",
        "Benar testing": f"{(p_test == y_test.values).sum()} / {len(y_test)}",
        "Akurasi testing": f"{accuracy_score(y_test, p_test):.1%}",
        "Selisih (poin)": round((accuracy_score(y_train, p_train) - accuracy_score(y_test, p_test)) * 100, 1),
    })
print(pd.DataFrame(hasil_algoritma).to_string(index=False))

# ----------------------------------------------------------------------
# 8. SIMPAN HASIL PER RESTORAN
# ----------------------------------------------------------------------
garis("8. SIMPAN HASIL PER RESTORAN")


def tabel_hasil(X_bagian, y_bagian, tebakan, nama_data):
    t = pd.DataFrame({
        "Data": nama_data,
        "Restoran": df.loc[X_bagian.index, "Restaurant Name"].values,
        "Kota": df.loc[X_bagian.index, "City"].values,
        "Votes": df.loc[X_bagian.index, "Votes"].values,
        "Rating asli": y_bagian.values,
        "Tebakan": tebakan,
    })
    t["Hasil"] = np.where(t["Rating asli"] == t["Tebakan"], "Benar", "Salah")
    return t


hasil_train = tabel_hasil(X_train, y_train, tebakan_train, "Training")
hasil_test = tabel_hasil(X_test, y_test, tebakan_test, "Testing")
try:
    with pd.ExcelWriter("hasil_training_testing.xlsx") as w:
        hasil_train.to_excel(w, sheet_name="Training", index=False)
        hasil_test.to_excel(w, sheet_name="Testing", index=False)
        pd.DataFrame(hasil_algoritma).to_excel(w, sheet_name="Perbandingan", index=False)
    print("Disimpan ke: hasil_training_testing.xlsx (sheet Training, Testing, Perbandingan)")
except ImportError:
    pd.concat([hasil_train, hasil_test]).to_csv("hasil_training_testing.csv", index=False,
                                                sep=";", encoding="utf-8-sig")
    print("Disimpan ke: hasil_training_testing.csv")