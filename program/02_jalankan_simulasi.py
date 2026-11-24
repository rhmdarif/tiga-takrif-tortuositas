"""Tahap 2. Simulasi aliran LBM D3Q19 TRT pada citra batuan, satu arah per pemanggilan.

Susunan domain meniru penjalanan acuan pada data/sebelas-batupasir.csv: pori penghantar saja,
penyangga fluida 4 voxel pada muka masuk dan keluar, dinding satu voxel pada keempat sisi
melintang (domain 264 x 258 x 258). Aliran didorong gaya badan; permeabilitas dihitung dari
gradien pendorong efektif yang diukur (lihat umum.sifat_aliran).

Patokan berhenti: perubahan nisbi <u_x> per 100 langkah di bawah --tol (bawaan 1e-6, sepuluh
kali lebih ketat daripada patokan 1e-5 pada pedoman).

Keluaran pada --keluaran (bawaan hasil/lbm/arah_<x|y|z>/):
  ringkasan.json   permeabilitas, Ma, Re pori, langkah, waktu, keadaan konvergensi
  riwayat.csv      riwayat sisa konvergensi terhadap langkah
  medan_u.npy      medan kecepatan pada batuan, float32, bentuk (3, N, N, N), BINGKAI ALIRAN
                   (komponen ke-0 searah aliran)
  titik_simpan.npz dipakai untuk melanjutkan bila penjalanan terputus; dihapus bila selesai

Contoh:
  python program/02_jalankan_simulasi.py --arah x
  python program/02_jalankan_simulasi.py --arah y --keluaran /content/drive/MyDrive/tortuositas/lbm
  python program/02_jalankan_simulasi.py --arah x --uji-langkah 500     # ukur kecepatan saja
"""

import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import umum as U  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arah", required=True, choices=list("xyz"))
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--g", type=float, default=2.5e-6)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--maks", type=int, default=60000)
    ap.add_argument("--penyangga", type=int, default=4)
    ap.add_argument("--kolisi", default="trt", choices=["trt", "bgk"],
                    help="bgk hanya untuk pembanding terhadap penjalanan acuan")
    ap.add_argument("--backend", default="auto", choices=["auto", "cpu", "gpu"])
    ap.add_argument("--keluaran", default=os.path.join(U.HASIL, "lbm"))
    ap.add_argument("--simpan-tiap-menit", type=float, default=15.0)
    ap.add_argument("--uji-langkah", type=int, default=0, help="hanya mengukur MLUPS selama sekian langkah")
    ap.add_argument("--padat-npy", default=None,
                    help="larik padat .npy pengganti citra asli (dipakai sapuan ambang)")
    arg = ap.parse_args()

    meta = U.muat_metadata()
    padat_batuan = np.load(arg.padat_npy) if arg.padat_npy else U.baca_citra(meta)
    dx = meta["ukuran_voxel_m"]
    padat, iris, pori = U.domain_aliran(padat_batuan, arg.arah, penyangga=arg.penyangga, dinding=True)
    print(f"Arah {arg.arah}: domain {padat.shape}, porositas penghantar {pori.mean():.6f}")

    if arg.uji_langkah:
        p = U.PemecahLBM(padat, tau=arg.tau, g=arg.g, backend=arg.backend)
        p.langkah(100)                                   # pemanasan dan kompilasi
        t = time.time(); p.langkah(arg.uji_langkah); dt = time.time() - t
        mlups = p.n * arg.uji_langkah / dt / 1e6
        print(f"backend {p.backend}: {mlups:.1f} MLUPS, {dt / arg.uji_langkah * 1000:.1f} ms per langkah")
        print(f"perkiraan 30.000 langkah: {30000 * dt / arg.uji_langkah / 60:.1f} menit")
        return

    keluar = os.path.join(arg.keluaran, f"arah_{arg.arah}" + ("" if arg.kolisi == "trt" else f"_{arg.kolisi}"))
    os.makedirs(keluar, exist_ok=True)
    titik = os.path.join(keluar, "titik_simpan.npz")
    r = U.jalankan_lbm(padat, tau=arg.tau, g=arg.g, maks=arg.maks, tol=arg.tol, periksa=100,
                       backend=arg.backend, titik_simpan=titik, simpan_tiap_detik=60 * arg.simpan_tiap_menit,
                       kolisi=arg.kolisi)
    s = U.sifat_aliran(r, iris, pori, arg.penyangga)
    Ub = r["U"][(slice(None),) + iris]
    # bilangan Reynolds pori: u pori rerata * diameter hidraulik / nu, d_h = 4 phi / S (satuan voxel)
    muka = sum(np.sum(pori != np.roll(pori, 1, axis=a)) for a in range(3))
    d_h = 4.0 * pori.sum() / muka
    Re = s["u_pori_rerata_kisi"] * d_h / r["nu"]
    k_m2 = s["k_kisi"] * dx * dx
    ringkas = dict(arah=arg.arah, kolisi=arg.kolisi, backend=r["backend"], tau=arg.tau, tol=arg.tol,
                   domain=list(padat.shape), phi_penghantar=float(pori.mean()),
                   langkah=r["langkah"], konvergen=r["konvergen"], meledak=r["meledak"],
                   sisa_akhir=float(r["riwayat"][-1][2]), detik=r["detik"], MLUPS=r["MLUPS"],
                   K_kisi=s["k_kisi"], K_m2=k_m2, K_mD=k_m2 / U.DARCY_M2 * 1000,
                   Ma=s["Ma"], Re_pori=Re, d_h_voxel=d_h, **{k: v for k, v in s.items() if k not in ("k_kisi", "Ma")})
    U.tulis_json(os.path.join(keluar, "ringkasan.json"), ringkas)
    with open(os.path.join(keluar, "riwayat.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["langkah", "rerata_ux", "sisa"]); w.writerows(r["riwayat"].tolist())
    np.save(os.path.join(keluar, "medan_u.npy"), Ub.astype(np.float32))
    if r["konvergen"] and os.path.exists(titik):
        os.remove(titik)
    print(f"K = {ringkas['K_mD']:.1f} mD, langkah {r['langkah']}, konvergen {r['konvergen']}, "
          f"Ma {s['Ma']:.1e}, Re pori {Re:.1e}, {r['detik'] / 60:.1f} menit")


if __name__ == "__main__":
    main()
