"""Tahap 4. Akibat pilihan takrif tortuositas terhadap permeabilitas dugaan Kozeny dan Carman.

Bentuk yang dipakai (Pertemuan 5, dengan tetapan Kozeny c = c0 * tau^2):

    k_KC = phi^3 / (c0 * tau^2 * S_v^2)

dengan S_v luas antarmuka pori-padatan per satuan VOLUME SELURUHNYA (perjanjian perujukan ini
dinyatakan tegas karena P5 memakai luas per volume padatan, S_0 = S_v / (1 - phi); keduanya
setara). c0 = 2 adalah faktor bentuk tabung bundar (Carman), sehingga c = 5 ketika tau^2 = 2,5.
Nilai c0 tidak memengaruhi NISBAH antartakrif, yang menjadi pertanyaan proyek ini.

phi dan S_v dihitung pada ruang pori penghantar dan SAMA PERSIS untuk ketiga takrif; yang
berubah hanya tau. Akibatnya k_KC(tau_g) / k_KC(tau_h) = (tau_h / tau_g)^2, yaitu kebalikan
faktor (tau_g/tau_h)^2.

Koreksi bias tangga voxel (dua cara):
  (a) cacah muka voxel dikali 2/3 (permukaan berarah acak; cacah muka melebihkan 3/2 kali);
  (b) luas jaring marching cubes atas citra biner yang dihaluskan tipis.
Keduanya diuji pada bola berjari-jari 24 voxel yang luasnya diketahui. Takrif (a) dipakai
sebagai nilai utama, (b) sebagai pembanding.

Keluaran: hasil/luas_spesifik.json, hasil/kozeny_carman.csv
    python program/04_kozeny_carman.py
"""

import csv
import json
import os
import sys

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import umum as U  # noqa: E402

C0 = 2.0


def cacah_muka(pori):
    """Cacah muka voxel antara pori dan padatan, tanpa pembungkusan pada tepi larik."""
    return int(sum(np.count_nonzero(np.diff(pori.astype(np.int8), axis=a)) for a in range(3)))


def luas_marching_cubes(pori, sigma=0.7):
    from skimage import measure
    halus = ndi.gaussian_filter(np.pad(pori, 1, mode="edge").astype(np.float32), sigma)
    verts, faces, _, _ = measure.marching_cubes(halus, 0.5)
    return float(measure.mesh_surface_area(verts, faces))


def luas_spesifik(pori):
    """Mengembalikan S_v dalam 1/voxel: mentah, terkoreksi 2/3, dan marching cubes."""
    V = pori.size
    muka = cacah_muka(pori)
    return dict(S_v_mentah=muka / V, S_v_koreksi_2per3=(2 / 3) * muka / V,
                S_v_marching_cubes=luas_marching_cubes(pori) / V)


def uji_bola(r=24.0, n=64):
    z, y, x = np.indices((n, n, n)) - (n - 1) / 2
    bola = (x * x + y * y + z * z) <= r * r
    eksak = 4 * np.pi * r * r
    return dict(r=r, luas_eksak=eksak, nisbah_mentah=cacah_muka(bola) / eksak,
                nisbah_koreksi_2per3=(2 / 3) * cacah_muka(bola) / eksak,
                nisbah_marching_cubes=luas_marching_cubes(bola) / eksak)


def main():
    meta = U.muat_metadata()
    dx = meta["ukuran_voxel_m"]
    padat = U.baca_citra(meta)
    with open(os.path.join(U.HASIL, "tortuositas.json"), encoding="utf-8") as f:
        tort = {b["arah"]: b for b in json.load(f)["per_arah"]}
    sim = {}
    for a in "xyz":
        with open(os.path.join(U.HASIL, "lbm", f"arah_{a}", "ringkasan.json"), encoding="utf-8") as f:
            sim[a] = json.load(f)

    bola = uji_bola()
    print(f"Uji bola r=24: cacah muka / luas eksak {bola['nisbah_mentah']:.3f}, sesudah 2/3 "
          f"{bola['nisbah_koreksi_2per3']:.3f}, marching cubes {bola['nisbah_marching_cubes']:.3f}")

    # ruang pori penghantar sama pada ketiga arah untuk percontoh ini; dihitung sekali per arah agar umum
    luas = {}
    baris = []
    for a in "xyz":
        pori = U.pori_penghantar(U.bingkai_aliran(padat, a))
        phi = float(pori.mean())
        S = luas_spesifik(pori)
        luas[a] = dict(phi_penghantar=phi, **S, S_v_koreksi_2per3_per_m=S["S_v_koreksi_2per3"] / dx)
        t = tort[a]
        k_sim = sim[a]["K_mD"]
        for nama, tau in [("tau_h", t["tau_h"]), ("tau_g", t["tau_g"]), ("tau_e", t["tau_e"]),
                          ("tau_e_archie", t["tau_e_archie_phi_penghantar"])]:
            for cara in ("S_v_koreksi_2per3", "S_v_marching_cubes", "S_v_mentah"):
                k_vox = phi ** 3 / (C0 * tau ** 2 * S[cara] ** 2)
                k_mD = k_vox * dx * dx / U.DARCY_M2 * 1000
                baris.append(dict(arah=a, takrif=nama, tau=tau, luas=cara, phi=phi, S_v_per_voxel=S[cara],
                                  K_KC_mD=k_mD, K_sim_mD=k_sim, nisbah_KC_thd_sim=k_mD / k_sim))

    with open(os.path.join(U.HASIL, "kozeny_carman.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(baris[0])); w.writeheader(); w.writerows(baris)
    U.tulis_json(os.path.join(U.HASIL, "luas_spesifik.json"), dict(uji_bola=bola, per_arah=luas, c0=C0))

    utama = [b for b in baris if b["luas"] == "S_v_koreksi_2per3"]
    print(f"\nS_v (penghantar): mentah {luas['x']['S_v_mentah'] / dx:.3e}, terkoreksi 2/3 "
          f"{luas['x']['S_v_koreksi_2per3_per_m']:.3e}, marching cubes {luas['x']['S_v_marching_cubes'] / dx:.3e} 1/m")
    print(f"\n{'arah':>4} {'K_sim':>8} " + "".join(f"{n:>22s}" for n in ("K_KC(tau_h)", "K_KC(tau_g)", "K_KC(tau_e)",
                                                                     "K_KC(phi^-1/2)")))
    for a in "xyz":
        r = {b["takrif"]: b for b in utama if b["arah"] == a}
        print(f"{a:>4} {r['tau_h']['K_sim_mD']:8.0f} " + "".join(
            f"{r[n]['K_KC_mD']:11.0f} ({r[n]['nisbah_KC_thd_sim']:5.2f}x)   " for n in ("tau_h", "tau_g", "tau_e", "tau_e_archie")))
        print(f"     K_KC(tau_g)/K_KC(tau_h) = {r['tau_g']['K_KC_mD'] / r['tau_h']['K_KC_mD']:.3f}, "
              f"yaitu pelebihan {100 * (r['tau_g']['K_KC_mD'] / r['tau_h']['K_KC_mD'] - 1):+.1f} %")


if __name__ == "__main__":
    main()
