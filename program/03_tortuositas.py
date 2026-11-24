"""Tahap 3. Ketiga takrif tortuositas pada ketiga arah sumbu (sembilan angka).

Takrif dan perjanjian, semuanya atas himpunan voxel PORI PENGHANTAR arah yang bersangkutan:
  tau_h  hidrolik   = sum |u| / sum u_aliran, kedua jumlah atas voxel penghantar yang sama
                      (medan dari tahap 2, hasil/lbm/arah_*/medan_u.npy)
  tau_g  geometrik  = rerata atas voxel penghantar pada muka keluar dari
                      (cacah voxel pada lintasan terpendek BFS 6 tetangga) / N
  tau_e  listrik    = sqrt(F * phi_penghantar), F dari persamaan Laplace; perjanjian AKAR,
                      bukan F*phi
  tau_e,Archie      = phi^(-1/2), yaitu sqrt(F*phi) dengan F = phi^-2 (a = 1, m = 2),
                      dihitung pada phi penghantar dan pada phi total berdampingan
  m tercocok        = -ln F / ln phi_penghantar  (a = 1)

Keluaran:
  hasil/tortuositas.csv, hasil/tortuositas.json
  hasil/sebaran_laju_<arah>.csv  sebaran |u|/<|u|> pada voxel penghantar, dan bagian fluks
                                 yang dibawa voxel tercepat (pembobotan saluran utama)

    python program/03_tortuositas.py
"""

import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import umum as U  # noqa: E402


def sebaran_laju(Ub, pori):
    besar = np.sqrt((Ub ** 2).sum(0))[pori]
    ux = Ub[0][pori]
    nisbi = besar / besar.mean()
    tepi = np.concatenate([np.linspace(0, 1, 21)[:-1], np.geomspace(1, max(nisbi.max(), 1.01) * 1.0001, 41)])
    hist, _ = np.histogram(nisbi, bins=tepi)
    urut = np.sort(ux)[::-1]
    kum = np.cumsum(urut) / urut.sum()
    bagian = {f"fluks_oleh_{int(p * 100)}persen_tercepat": float(kum[int(p * urut.size) - 1]) for p in (0.01, 0.05, 0.10, 0.20)}
    return tepi, hist, bagian


def takrif_satu_arah(padat_batuan, arah, medan_dir=None, phi_total=None, cetak=print):
    pb = U.bingkai_aliran(padat_batuan, arah)
    pori = U.pori_penghantar(pb)
    phi_c = float(pori.mean())
    phi_t = float((~padat_batuan).mean()) if phi_total is None else phi_total
    hasil = dict(arah=arah, phi_total=phi_t, phi_penghantar=phi_c, n_komponen=U.cacah_komponen(~pb))

    if medan_dir is not None:
        Ub = np.load(os.path.join(medan_dir, f"arah_{arah}", "medan_u.npy")).astype(np.float64)
        hasil["tau_h"] = U.tortuositas_hidrolik(Ub, pori)
        hasil["_sebaran"] = sebaran_laju(Ub, pori)
        del Ub

    t = time.time()
    g = U.tortuositas_geometrik(pori)
    hasil.update(tau_g=g["tau_g"], tau_g_min=g["tau_g_min"], langkah_min=g["langkah_min"], n_keluar=g["n_keluar"])
    cetak(f"  {arah}: tau_g selesai ({time.time() - t:.0f} s)")

    t = time.time()
    f = U.faktor_formasi(pori)
    F = f["F"]
    hasil.update(F=F, Deff_rel=f["Deff_rel"], laplace_info=f["info"], laplace_sisa=f["sisa_akhir"],
                 laplace_selisih_fluks=f["selisih_fluks"],
                 tau_e=float(np.sqrt(F * phi_c)),
                 tau_e_salah_perjanjian_Fphi=float(F * phi_c),
                 m_cocok=float(-np.log(F) / np.log(phi_c)),
                 tau_e_archie_phi_penghantar=float(phi_c ** -0.5),
                 tau_e_archie_phi_total=float(phi_t ** -0.5))
    cetak(f"  {arah}: Laplace selesai ({time.time() - t:.0f} s, info {f['info']}, sisa {f['sisa_akhir']:.1e})")
    if "tau_h" in hasil:
        hasil["nisbah_g_h"] = hasil["tau_g"] / hasil["tau_h"]
        hasil["faktor_KC_g_thd_h"] = hasil["nisbah_g_h"] ** 2
    return hasil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--medan", default=os.path.join(U.HASIL, "lbm"), help="folder keluaran tahap 2")
    ap.add_argument("--keluaran", default=U.HASIL)
    arg = ap.parse_args()
    padat = U.baca_citra()

    baris = []
    for a in "xyz":
        h = takrif_satu_arah(padat, a, arg.medan)
        tepi, hist, bagian = h.pop("_sebaran")
        h.update(bagian)
        with open(os.path.join(arg.keluaran, f"sebaran_laju_{a}.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["tepi_bawah", "tepi_atas", "cacah_voxel"])
            for i in range(len(hist)):
                w.writerow([tepi[i], tepi[i + 1], int(hist[i])])
        baris.append(h)

    kolom = ["arah", "phi_total", "phi_penghantar", "tau_h", "tau_g", "tau_g_min", "langkah_min", "F", "m_cocok",
             "tau_e", "tau_e_archie_phi_penghantar", "tau_e_archie_phi_total", "tau_e_salah_perjanjian_Fphi",
             "nisbah_g_h", "faktor_KC_g_thd_h", "fluks_oleh_10persen_tercepat"]
    with open(os.path.join(arg.keluaran, "tortuositas.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=kolom, extrasaction="ignore"); w.writeheader(); w.writerows(baris)

    ringkas = {}
    for t in ("tau_h", "tau_g", "tau_e", "tau_e_archie_phi_penghantar"):
        v = [b[t] for b in baris]
        ringkas[t] = dict(terbesar=max(v), terkecil=min(v), nisbah_terbesar_terkecil=max(v) / min(v),
                          arah_terbesar="xyz"[int(np.argmax(v))])
    nis = [b["nisbah_g_h"] for b in baris]
    ringkas["nisbah_g_h_terbesar_pada_arah"] = "xyz"[int(np.argmax(nis))]
    U.tulis_json(os.path.join(arg.keluaran, "tortuositas.json"), dict(per_arah=baris, ringkasan=ringkas))

    print(f"\n{'':22s}" + "".join(f"{a:>10s}" for a in "xyz") + f"{'maks/min':>10s}")
    for t, nama in [("tau_h", "tau_h  hidrolik"), ("tau_g", "tau_g  geometrik"), ("tau_e", "tau_e  sqrt(F phi_c)"),
                    ("tau_e_archie_phi_penghantar", "phi_c^-1/2 (m = 2)")]:
        print(f"{nama:22s}" + "".join(f"{b[t]:10.4f}" for b in baris) + f"{ringkas[t]['nisbah_terbesar_terkecil']:10.4f}")
    for t, nama in [("F", "F"), ("m_cocok", "m tercocok"), ("nisbah_g_h", "tau_g / tau_h"),
                    ("faktor_KC_g_thd_h", "(tau_g/tau_h)^2"), ("tau_e_archie_phi_total", "phi_total^-1/2")]:
        print(f"{nama:22s}" + "".join(f"{b[t]:10.4f}" for b in baris))


if __name__ == "__main__":
    main()
