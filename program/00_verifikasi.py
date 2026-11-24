"""Tahap 0. Verifikasi ketiga pemecah pada geometri yang penyelesaiannya diketahui.

Patokan lulus (pedoman Bab 9):
  * celah berdinding sejajar: permeabilitas meleset <= 2 persen terhadap h^2/12,
    dan tau_h berada pada 0,99 sampai 1,01;
  * tabung lurus: tau_g tepat 1,000 (penangkap kekeliruan satu voxel);
  * tabung lurus: F*phi = 1, sehingga tau_e = sqrt(F*phi) = 1.
Ditambah dua uji pendukung: celah yang dijalankan melalui susunan domain batuan
(penyangga dan gradien efektif), serta kesamaan hasil backend CPU dan GPU.

Keluaran: hasil/verifikasi.json, hasil/verifikasi_celah.csv, hasil/riwayat_celah_h16.csv

    python program/00_verifikasi.py [--backend auto|cpu|gpu]
"""

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import umum as U  # noqa: E402


def celah_murni(h, backend, tau=1.0, g=1e-6):
    padat = U.celah_sejajar(h)
    r = U.jalankan_lbm(padat, tau=tau, g=g, tol=1e-10, periksa=50, maks=400000, backend=backend,
                       cetak=lambda *_: None, min_langkah=200)
    alir = ~padat
    k = r["nu"] * r["U"][0][alir].mean() / g          # k atas luas celah
    th = U.tortuositas_hidrolik(r["U"], alir)
    k_teori = h * h / 12.0
    # profil eksak pada pusat voxel: u(z) = g z (h - z) / (2 nu), dinding setengah voxel di luar
    z = np.arange(1, h + 1) - 0.5
    u_eksak = g * z * (h - z) / (2 * r["nu"])
    galat_titik = float(np.abs(r["U"][0][0, 0, 1:-1] - u_eksak).max() / u_eksak.max())
    # rerata parabola eksak atas pusat voxel = h^2/12 + 1/24 (kuadratur titik tengah), bukan h^2/12
    k_titik_tengah = h * h / 12.0 + 1.0 / 24.0
    return dict(h=h, k_kisi=k, k_teori=k_teori, galat_k=(k - k_teori) / k_teori, tau_h=th,
                galat_titik_maks=galat_titik, galat_k_thd_kuadratur=(k - k_titik_tengah) / k_titik_tengah,
                langkah=r["langkah"], konvergen=r["konvergen"], Ma=float(np.sqrt(3) * np.abs(r["U"]).max()),
                riwayat=r["riwayat"], backend=r["backend"])


def celah_jalur_penuh(h, backend, panjang=96, penyangga=4):
    """Celah yang diperlakukan seperti citra batuan: penyangga + gradien pendorong efektif."""
    batuan = np.zeros((panjang, 4, h + 2), bool)
    batuan[:, :, 0] = True; batuan[:, :, -1] = True
    padat, iris, pori = U.domain_aliran(batuan, "x", penyangga=penyangga, dinding=False)
    r = U.jalankan_lbm(padat, tol=1e-10, periksa=50, maks=400000, backend=backend, g=1e-6,
                       cetak=lambda *_: None, min_langkah=200)
    s = U.sifat_aliran(r, iris, pori, penyangga)
    k_teori = (h / (h + 2)) * h * h / 12.0              # permeabilitas Darcy atas seluruh muka
    th = U.tortuositas_hidrolik(r["U"][(slice(None),) + iris], pori)
    return dict(h=h, k_kisi=s["k_kisi"], k_teori=k_teori, galat_k=(s["k_kisi"] - k_teori) / k_teori,
                tau_h=th, langkah=r["langkah"], konvergen=r["konvergen"])


def uji_tabung():
    baris = []
    for a, nama in enumerate("xyz"):
        padat = U.tabung_lurus(64, 10, arah=a)
        pori = U.pori_penghantar(U.bingkai_aliran(padat, nama))
        tg = U.tortuositas_geometrik(pori)
        N = pori.shape[0]
        salah = (tg["langkah_min"]) / N                  # perjanjian keliru: langkah, bukan voxel
        ff = U.faktor_formasi(pori)
        phi = pori.mean()
        baris.append(dict(arah=nama, tau_g=tg["tau_g"], tau_g_min=tg["tau_g_min"], tau_g_perjanjian_keliru=salah,
                          phi=float(phi), F=ff["F"], F_phi=ff["F"] * phi, tau_e=float(np.sqrt(ff["F"] * phi))))
    return baris


def uji_cpu_gpu():
    if not U.gpu_tersedia():
        return dict(dijalankan=False, alasan="GPU tidak tersedia")
    rng = np.random.default_rng(3)
    from scipy import ndimage as ndi
    medan = ndi.gaussian_filter(rng.standard_normal((40, 40, 40)), 2.5, mode="wrap")
    padat = medan > np.quantile(medan, 0.70)               # porositas 0,70
    hasil = {}
    for be in ("cpu", "gpu"):
        r = U.jalankan_lbm(padat, tol=0, maks=2000, periksa=500, backend=be, cetak=lambda *_: None, min_langkah=0)
        hasil[be] = r["U"]
    beda = float(np.abs(hasil["cpu"] - hasil["gpu"]).max() / np.abs(hasil["cpu"]).max())
    return dict(dijalankan=True, selisih_nisbi_maks=beda, lulus=beda < 1e-10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="auto")
    arg = ap.parse_args()
    os.makedirs(U.HASIL, exist_ok=True)

    print("A. Celah berdinding sejajar, periodik murni (pemecah LBM)")
    print(f"   {'h':>4} {'k':>10} {'h^2/12':>10} {'galat %':>9} {'galat titik':>12} {'tau_h':>9} {'langkah':>8}")
    celah = []
    for h in (8, 16, 32):
        r = celah_murni(h, arg.backend)
        celah.append(r)
        print(f"   {h:4d} {r['k_kisi']:10.5f} {r['k_teori']:10.5f} {100 * r['galat_k']:9.4f} {r['galat_titik_maks']:12.1e} {r['tau_h']:9.6f} {r['langkah']:8d}")
    r16 = [r for r in celah if r["h"] == 16][0]
    with open(os.path.join(U.HASIL, "riwayat_celah_h16.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["langkah", "rerata_ux", "sisa"]); w.writerows(r16["riwayat"].tolist())

    print("\nB. Celah melalui susunan domain batuan (penyangga + gradien efektif)")
    print("   Galat yang menyusut sebanding 1/panjang menandakan rugi masuk dan keluar, bukan galat rumus.")
    jp = []
    for L in (96, 192, 384):
        r = celah_jalur_penuh(16, arg.backend, panjang=L); r["panjang"] = L; jp.append(r)
        print(f"   h=16 panjang {L:4d}  k {r['k_kisi']:.5f}  teori {r['k_teori']:.5f}  galat {100 * r['galat_k']:7.3f} %"
              f"  galat*panjang {r['galat_k'] * L:6.3f}  tau_h {r['tau_h']:.5f}")

    print("\nC dan D. Tabung lurus 64^3, sisi 10 voxel (BFS dan Laplace)")
    print(f"   {'arah':>4} {'tau_g':>8} {'tau_g_min':>9} {'keliru*':>8} {'F*phi':>9} {'tau_e':>8}")
    tabung = uji_tabung()
    for t in tabung:
        print(f"   {t['arah']:>4} {t['tau_g']:8.4f} {t['tau_g_min']:9.4f} {t['tau_g_perjanjian_keliru']:8.4f} {t['F_phi']:9.6f} {t['tau_e']:8.5f}")
    print("   * menghitung langkah, bukan voxel: memberi nilai di bawah 1 yang mustahil")

    print("\nE. Kesamaan backend CPU dan GPU (medan acak 40^3, 2000 langkah)")
    cg_ = uji_cpu_gpu()
    print("   " + (f"selisih nisbi maks {cg_['selisih_nisbi_maks']:.2e}" if cg_["dijalankan"] else cg_["alasan"]))

    patokan = {
        "celah_galat_k_maks_persen": max(abs(100 * r["galat_k"]) for r in celah),
        "celah_tau_h_selisih_maks": max(abs(r["tau_h"] - 1) for r in celah),
        "celah_galat_titik_maks": max(r["galat_titik_maks"] for r in celah),
        "tabung_tau_g_selisih_maks": max(abs(t["tau_g"] - 1) for t in tabung),
        "tabung_F_phi_selisih_maks": max(abs(t["F_phi"] - 1) for t in tabung),
    }
    lulus = dict(
        celah_k=patokan["celah_galat_k_maks_persen"] <= 2.0,
        celah_tau_h=patokan["celah_tau_h_selisih_maks"] <= 0.01,
        celah_profil_titik=patokan["celah_galat_titik_maks"] < 1e-8,
        tabung_tau_g=patokan["tabung_tau_g_selisih_maks"] < 1e-9,
        tabung_F_phi=patokan["tabung_F_phi_selisih_maks"] < 1e-6,
    )
    print("\nPATOKAN")
    for k, v in lulus.items():
        print(f"   [{'LULUS' if v else 'GAGAL'}] {k}")
    U.tulis_json(os.path.join(U.HASIL, "verifikasi.json"),
                 dict(celah=[{k: v for k, v in r.items() if k != "riwayat"} for r in celah], celah_jalur_penuh=jp,
                      tabung=tabung, cpu_gpu=cg_, patokan=patokan, lulus=lulus))
    with open(os.path.join(U.HASIL, "verifikasi_celah.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["h", "k_kisi", "k_teori", "galat_persen", "tau_h", "langkah", "backend"])
        for r in celah:
            w.writerow([r["h"], r["k_kisi"], r["k_teori"], 100 * r["galat_k"], r["tau_h"], r["langkah"], r["backend"]])
    sys.exit(0 if all(lulus.values()) else 1)


if __name__ == "__main__":
    main()
