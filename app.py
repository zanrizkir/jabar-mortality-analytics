from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum
from pyspark.sql.types import IntegerType


# 1. Inisialisasi Spark
spark = SparkSession.builder \
    .appName("MiniProject_KematianJabar") \
    .getOrCreate()

# 2. Load Data (Tahap Storage ke Processing)
df = spark.read.csv("data/dataset.csv", header=True, inferSchema=True)


# Hitung jumlah null di setiap kolom
print("=== Cek Missing Values ===")
df.select(*(sum(col(c).isNull().cast("int")).alias(c) for c in df.columns)).show()

# Hitung jumlah baris awal
total_awal = df.count()

# Hapus duplikat
df_clean = df.dropDuplicates()
total_akhir = df_clean.count()

print(f"Data awal: {total_awal}, Data setelah hapus duplikat: {total_akhir}")
print(f"Jumlah duplikat dibuang: {total_awal - total_akhir}")

df_final = df_clean.select(
    "tahun",
    "nama_kabupaten_kota",
    "jenis_kematian",
    "penyebab_kematian",
    "jumlah_kematian"
)

df_final = df_final.withColumn("jumlah_kematian", col("jumlah_kematian").cast(IntegerType()))

df_final.printSchema()

#Total kematian per penyebab
df_final.groupBy("penyebab_kematian").sum("jumlah_kematian").show()