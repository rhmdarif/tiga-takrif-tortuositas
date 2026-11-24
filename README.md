# Tiga Takrif Tortuositas pada Satu Batuan

**Nama:** Rahmad Arif
**NPM:** 251020250510
**Jalur:** simulasi
**Proyek:** Bab 9, Tiga Takrif Tortuositas pada Satu Batuan
**Percontoh:** Bentheimer, 256³ voxel, Δx = 2,25 µm (himpunan sebelas batupasir, Neumann dkk. 2021)

---

## Pertanyaan yang dijawab

Seberapa jauh tortuositas hidrolik, geometrik, dan listrik berbeda pada Bentheimer di ketiga
arah sumbu, dan berapa faktor pelebihan permeabilitas dugaan Kozeny–Carman bila τg dipakai
menggantikan τh?

## Cara menjalankan

Python 3.12. GPU NVIDIA bersifat pilihan. Tanpa GPU, LBM berjalan pada CPU dengan numba,
kira-kira lima kali lebih lambat.

```sh
pip install -r lingkungan.txt
pip install "numba-cuda[cu13]==0.30.4"       # pilihan, hanya bila ada GPU NVIDIA
python program/00_verifikasi.py              # wajib lulus; keluar dengan kode 1 bila tidak
python program/01_kelayakan_dan_ambang.py
python program/02_jalankan_simulasi.py --arah x
python program/02_jalankan_simulasi.py --arah y
python program/02_jalankan_simulasi.py --arah z
python program/03_tortuositas.py
python program/04_kozeny_carman.py
python program/05_sapuan_ambang.py siapkan
python program/05_sapuan_ambang.py lbm
python program/05_sapuan_ambang.py geometri  # boleh bersamaan dengan langkah sebelumnya
python program/05_sapuan_ambang.py gabung
python program/06_uji_rev.py
python program/07_gambar.py
python program/08_animasi.py irisan          # animasi paparan, 1 menit
python program/08_animasi.py berkembang      # animasi paparan; menjalankan ulang LBM arah y, 5-10 menit di GPU
python program/09_visual3d.py data           # data penampil 3D (permukaan pori, garis arus, lintasan BFS), detik
python program/09_visual3d.py gif            # kubus 3D berputar, 1 menit
```

Pembanding opsional yang dipakai pada laporan:
`python program/02_jalankan_simulasi.py --arah x --kolisi bgk` dan
`python program/06_uji_rev.py --cara acak`.

Pada Google Colab, buka `colab/peluncur_colab.ipynb`. Simulasi menulis titik simpan tiap
15 menit dan melanjutkan sendiri bila sesi terputus.

**Perkiraan waktu hitung seluruhnya:** sekitar 2 jam 30 menit pada GTX 1660 Super (6 GB)
dengan CPU 8 inti. LBM 256³ memakan 4–10 menit per arah pada GPU (sekitar 150 juta pembaruan
kisi per detik) dan 30–75 menit per arah pada CPU. Pemecah Laplace memakan 3–6 menit per arah.

**Ruang cakram:** medan kecepatan (`medan_u.npy`, 200 MB per arah) dan domain sapuan memerlukan
sekitar 3,7 GB. Berkas itu tidak disertakan dalam repositori yang dikumpulkan karena dapat
dihasilkan ulang dengan perintah di atas. Semua angka pada laporan tersimpan pada berkas CSV
dan JSON yang kecil di `hasil/`.

## Perjanjian yang dipakai di seluruh program

| Hal | Perjanjian |
|---|---|
| Sumbu | x, y, z = indeks larik ke-0, ke-1, ke-2 (urutan C). Pola anisotropi K sama dengan berkas acuan. |
| Pori penghantar | Kluster pori bersambungan muka (6 tetangga) yang menyentuh muka masuk dan keluar. Menghasilkan ulang φ tersambung acuan 0,262393 tepat. |
| τh hidrolik | Σ\|u\| / Σu_aliran atas voxel pori penghantar yang sama, dari medan LBM D3Q19 TRT (Λ = 3/16, τ = 1) |
| τg geometrik | Rerata (cacah voxel pada lintasan terpendek BFS 6 tetangga) / N atas voxel penghantar di muka keluar. Cacah voxel = cacah langkah + 1, sehingga tabung lurus memberi 1. |
| τe listrik | √(F·φc), dengan F dari persamaan Laplace (volume hingga, potensial 1 dan 0 setengah voxel di luar muka). Perjanjian akar, bukan Fφ. |
| τe empirik | φc^(−1/2), yaitu Archie dengan a = 1 dan m = 2 |
| Domain LBM | Penyangga fluida 4 voxel di muka masuk dan keluar, dinding 1 voxel pada sisi melintang (264 × 258 × 258), sama dengan penjalanan acuan |
| K | ν q / G, dengan q = rerata u_aliran atas volume batuan dan G = g + c_s²(ρ_masuk − ρ_keluar)/(N+1) |
| Konvergensi | Perubahan nisbi ⟨u⟩ per 100 langkah < 10⁻⁶ |
| S_v | Cacah muka voxel × 2/3, per volume seluruhnya |
| Kozeny–Carman | k = φc³ / (c₀ τ² S_v²) dengan c₀ = 2. φc dan S_v sama untuk ketiga takrif. |
| Sapuan ambang | Citra biner dihaluskan Gauss (σ = 1 voxel), lalu diambang pada T = Φ(−s/σ), sehingga antarmuka datar bergeser s voxel. s = −0,5 … +0,5. |

## Daftar angka pada laporan beserta asalnya

| Angka pada laporan | Nilai | Program | Berkas keluaran |
|---|---|---|---|
| Verifikasi celah: galat k terhadap h²/12 (h = 8, 16, 32) | 0,78 / 0,20 / 0,05 % | `00_verifikasi.py` | `hasil/verifikasi_celah.csv` |
| Verifikasi celah: galat profil titik demi titik | ≤ 1,2 × 10⁻⁹ | `00_verifikasi.py` | `hasil/verifikasi.json` |
| Verifikasi celah: τh | 1,000000 | `00_verifikasi.py` | `hasil/verifikasi_celah.csv` |
| Verifikasi tabung lurus: τg dan F·φ | 1,0000 dan 1,000000 | `00_verifikasi.py` | `hasil/verifikasi.json` |
| Perjanjian τg keliru (cacah langkah) pada tabung | 0,9844 | `00_verifikasi.py` | `hasil/verifikasi.json` |
| Rugi masuk–keluar pada celah dengan penyangga (panjang 96, 192, 384) | −4,0 / −2,0 / −0,9 % | `00_verifikasi.py` | `hasil/verifikasi.json` |
| Leher pengendali d_c (x, y, z) | 24,2 / 30,2 / 22,9 µm | `01_kelayakan_dan_ambang.py` | `hasil/kelayakan.json` |
| Tabel ambang (9 geseran) | φ 0,214 … 0,315 | `01_kelayakan_dan_ambang.py` | `hasil/tabel_ambang.csv` |
| K simulasi (x, y, z) | 2737 / 3205 / 2806 mD | `02_jalankan_simulasi.py` | `hasil/lbm/arah_*/ringkasan.json` |
| Nisbah K terhadap acuan | 0,92 / 0,92 / 0,95 | `02_jalankan_simulasi.py` | `hasil/lbm/arah_*/ringkasan.json` |
| K simulasi BGK, arah x | 2779 mD | `02_jalankan_simulasi.py --kolisi bgk` | `hasil/lbm/arah_x_bgk/ringkasan.json` |
| Sisa akhir, langkah, Ma, Re pori | lihat berkas | `02_jalankan_simulasi.py` | `hasil/lbm/arah_*/riwayat.csv`, `ringkasan.json` |
| τh (x, y, z) | 1,4395 / 1,3785 / 1,4460 | `03_tortuositas.py` | `hasil/tortuositas.csv` |
| τg (x, y, z) | 1,3196 / 1,3327 / 1,3342 | `03_tortuositas.py` | `hasil/tortuositas.csv` |
| τe = √(F φc) (x, y, z) | 1,7261 / 1,6677 / 1,7276 | `03_tortuositas.py` | `hasil/tortuositas.csv` |
| τh rerata x, y, z terhadap acuan tortuositas fluks (1,4194) | 1,4213 (+0,14 %) | `03_tortuositas.py` | `hasil/tortuositas.csv`, `data/acuan_parameter_batuan.csv` |
| φc^(−1/2) dan φ^(−1/2) | 1,9522 dan 1,9463 | `03_tortuositas.py` | `hasil/tortuositas.csv` |
| F dan m tercocok (x, y, z) | 11,35 / 10,60 / 11,37; 1,816 / 1,765 / 1,817 | `03_tortuositas.py` | `hasil/tortuositas.csv` |
| Nisbah terbesar/terkecil τh, τg, τe | 1,049 / 1,011 / 1,036 | `03_tortuositas.py` | `hasil/tortuositas.json` |
| (τg/τh)² (x, y, z) | 0,84 / 0,93 / 0,85 | `03_tortuositas.py` | `hasil/tortuositas.csv` |
| Fluks yang dibawa 10 % voxel tercepat | 42–47 % | `03_tortuositas.py` | `hasil/tortuositas.json` |
| Uji bola: cacah muka / luas eksak, sesudah 2/3, marching cubes | 1,495 / 0,997 / 1,010 | `04_kozeny_carman.py` | `hasil/luas_spesifik.json` |
| S_v terkoreksi | 2,23 × 10⁴ m⁻¹ | `04_kozeny_carman.py` | `hasil/luas_spesifik.json` |
| K Kozeny–Carman per takrif dan arah, nisbah terhadap simulasi | lihat berkas | `04_kozeny_carman.py` | `hasil/kozeny_carman.csv` |
| Pelebihan K_KC(τg) terhadap K_KC(τh) (x, y, z) | +19,0 / +7,0 / +17,5 % | `04_kozeny_carman.py` | `hasil/kozeny_carman.csv` |
| Rentang sapuan: semua besaran, lebar dalam % | lihat berkas | `05_sapuan_ambang.py gabung` | `hasil/rentang_ambang.csv` |
| Lebar rentang K simulasi; τh; τg; τe; (τg/τh)² | 60–65; 2–4; 2–3; 6,5–7,3; 1,6–3,0 % | `05_sapuan_ambang.py gabung` | `hasil/rentang_ambang.csv` |
| Elastisitas η = d ln K / d ln φc (K simulasi x/y/z) | 3,50 / 3,23 / 3,56 | `05_sapuan_ambang.py elastisitas` | `hasil/elastisitas_ambang.csv` |
| C_v φ pada 64 dan 128 voxel (petak tak bertumpang) | 32,9 % dan 9,1 % | `06_uji_rev.py` | `hasil/rev_ringkasan.csv` |
| C_v K pada 64 voxel (x, y, z), subvolume menembus | 203 / 131 / 150 % | `06_uji_rev.py` | `hasil/rev_ringkasan.csv` |
| C_v K pada 64 voxel (x, y, z), semua petak (tak menembus = 0) | 209 / 134 / 150 % | `06_uji_rev.py` | `hasil/rev_ringkasan.csv` |
| C_v K pada 128 voxel (x, y, z) | 34 / 57 / 33 % | `06_uji_rev.py` | `hasil/rev_ringkasan.csv` |
| Petak 64³ yang tidak menembus | 5 dari 192 penjalanan | `06_uji_rev.py` | `hasil/rev_subvolume.csv` |
| Rerata φ subvolume acak 128³ (bias pusat) | 0,324, padahal φ domain 0,264 | `06_uji_rev.py --cara acak` | `hasil/rev_ringkasan_acak.csv` |
| Seluruh gambar | | `07_gambar.py` | `gambar/*.png` |
| Animasi paparan: aliran berkembang (arah y) dan penyusuran irisan (arah x) | | `08_animasi.py` | `gambar/animasi_*.gif`, `hasil/animasi/berkembang_y.json` |
| Visual 3D: garis arus, lintasan terpendek BFS (τg minimum diperiksa ulang = 1,03125) | | `09_visual3d.py` | `gambar/animasi_3d_x.gif`, `hasil/visual3d/visual3d.json` (berkas `.bin` tidak dikemas) |

## Susunan folder

| Folder | Isi |
|---|---|
| `data/` | Citra Bentheimer (`.raw`), `metadata.json`, CSV acuan sebelas batupasir, riwayat konvergensi acuan |
| `program/` | `umum.py` (pustaka bersama) dan program bernomor 00–07, satu untuk tiap tahap |
| `hasil/` | Keluaran berangka (CSV dan JSON); medan kecepatan `.npy` dapat dihasilkan ulang |
| `gambar/` | Gambar laporan, seluruhnya dibuat oleh `program/07_gambar.py` |
| `colab/` | Notebook peluncur untuk Google Colab |

## Keadaan yang perlu diketahui pembaca

- **Simulasi yang belum konvergen:** tidak ada. Ketiga arah domain penuh dan ke-15 penjalanan
  sapuan konvergen di bawah 10⁻⁶.
- **Volume elementer tidak tercapai.** Pada 128 voxel, C_v φ = 9,1 % dan C_v K = 33–57 %, semuanya
  di atas patokan 5 %. Pada 256 voxel hanya ada satu domain, sehingga C_v tidak dapat diukur.
  Ekstrapolasi C_v φ ∝ L^(−3/2) memberi sekitar 3–4 % pada 256 (porositas mungkin representatif),
  tetapi C_v K yang turun dengan laju serupa masih di atas 10 %. Jadi permeabilitas 256³ belum
  dapat dinyatakan representatif. Ini adalah ekstrapolasi, bukan ukuran.
  Subvolume bersudut acak (`--cara acak`) tidak dipakai karena bias ke daerah pusat yang lebih
  berpori: rerata φ delapan subvolume 128³ acak adalah 0,324, padahal φ domain 0,264.
- **Simulasi pada uji REV:** seluruh 216 penjalanan konvergen. Lima petak 64³ tidak menembus pada
  arahnya, dicatat sebagai K = 0, dan dilaporkan pada lajur tersendiri (C_v K "semua").
- **Selisih K terhadap acuan:** K saya 5–9 % di bawah acuan pada ketiga arah. Pergantian TRT ke
  BGK menjelaskan sekitar 1,5 poin. Sisanya diduga berasal dari beda cara mendorong aliran (gaya
  badan dengan penyangga, sedangkan acuan memakai beda kerapatan), tetapi dugaan ini belum diuji.
- **Rugi masuk–keluar:** susunan penyangga menurunkan k celah sebesar sekitar 3,6/panjang (voxel).
  Susunan ini sama dengan penjalanan acuan.
- **Titik tengah sapuan bukan citra asli:** penghalusan Gauss menghapus pori terkecil, sehingga
  s = 0 memberi φ 0,2620 (bukan 0,2640) dan 304 komponen (bukan 616).
- **Metrik τg:** BFS 6 tetangga memakai jarak Manhattan, sehingga lintasan miring dihitung
  terlalu panjang. τg sebenarnya sedikit lebih kecil, dan selisihnya terhadap τh sedikit lebih
  besar daripada yang dilaporkan.
- **Nilai mutlak Kozeny–Carman** bergantung pada c₀ = 2. Nisbah antartakrif tidak bergantung
  pada c₀.

## Pemakaian alat bantu kecerdasan buatan

Model bahasa besar Claude (Anthropic) dipakai melalui Claude Code pada empat bagian pekerjaan:
menulis dan memeriksa program pada folder `program/`, menjalankan program dan simulasi pada komputer
penulis, menyusun draf laporan dan naskah paparan beserta perapian kalimatnya, serta membuat
salindia paparan dari naskah. Seluruh angka pada laporan ini berasal dari keluaran program pada
repositori yang tersimpan sebagai berkas CSV dan JSON di `hasil/`; model bahasa tidak dipakai
sebagai sumber angka. Penulis mencocokkan sendiri angka pada tabel laporan dengan berkas asalnya
(Lampiran D laporan), dan memutuskan atau menyetujui sendiri setiap keputusan pada rantai kerja,
yaitu pilihan percontoh, perjanjian ketiga takrif, rentang ambang, dan patokan konvergensi.

## Lingkungan

Versi pustaka tercatat pada `lingkungan.txt`.
