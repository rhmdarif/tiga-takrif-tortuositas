"""Tahap 1. Kelayakan citra dan tabel ambang segmentasi setara.

Kelayakan: ukuran voxel dibandingkan terhadap leher pengendali, yaitu diameter bola terbesar
yang masih dapat melintasi percontoh (2 r_c), pada ketiga arah.

Ambang: citra bundel sudah biner, sehingga penyapuan dikerjakan pada aras keabuan SEMU, yaitu
indikator pori yang dihaluskan Gauss sigma = 1 voxel (umum.citra_abu_semu). Ambang T dipilih
dari geseran antarmuka s yang dikehendaki: T = Phi(-s / sigma), sehingga antarmuka datar bergeser
tepat s voxel. Rentang yang dipakai pada sapuan adalah s = -0,5 ... +0,5 voxel, yaitu
ketidakpastian setengah voxel akibat volume parsial pada tiap antarmuka.

Keluaran: hasil/kelayakan.json, hasil/tabel_ambang.csv
    python program/01_kelayakan_dan_ambang.py
"""

import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import umum as U  # noqa: E402

GESERAN_TABEL = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]
GESERAN_SAPUAN = [-0.5, -0.25, 0.0, 0.25, 0.5]


def main():
    meta = U.muat_metadata()
    dx = meta["ukuran_voxel_m"]
    padat = U.baca_citra(meta)

    print("Kelayakan: leher pengendali (bola terbesar yang melintas)")
    leher = {}
    for a in "xyz":
        pori = U.pori_penghantar(U.bingkai_aliran(padat, a))
        rc = U.jari_jari_kritis(pori)
        leher[a] = dict(r_c_voxel=rc, d_c_voxel=2 * rc, d_c_um=2 * rc * dx * 1e6)
        print(f"  {a}: r_c = {rc:.2f} voxel, d_c = {2 * rc:.1f} voxel = {2 * rc * dx * 1e6:.1f} um")

    print("\nTabel ambang (aras keabuan semu, sigma = 1 voxel)")
    print(f"  {'geseran s':>9} {'ambang T':>9} {'phi':>9} {'phi_hantar':>11} {'komponen':>9} {'menembus':>9}")
    abu = U.citra_abu_semu(padat)
    baris = [dict(geseran_voxel="asli", ambang_T="", phi=float((~padat).mean()),
                  phi_penghantar=float(U.pori_penghantar(U.bingkai_aliran(padat, "x")).mean()),
                  komponen=U.cacah_komponen(~padat), menembus_xyz="ya")]
    for s in GESERAN_TABEL:
        p = U.geser_antarmuka(padat, s, abu=abu)
        tembus = [U.pori_penghantar(U.bingkai_aliran(p, a)) for a in "xyz"]
        baris.append(dict(geseran_voxel=s, ambang_T=U.ambang_untuk_geseran(s), phi=float((~p).mean()),
                          phi_penghantar=float(tembus[0].mean()), komponen=U.cacah_komponen(~p),
                          menembus_xyz="ya" if all(t.any() for t in tembus) else "tidak",
                          dipakai_sapuan=s in GESERAN_SAPUAN))
    for b in baris:
        s, T = b["geseran_voxel"], b["ambang_T"]
        teks_s = s if isinstance(s, str) else f"{s:+.2f}"
        teks_T = T if T == "" else f"{T:.4f}"
        print(f"  {teks_s:>9} {teks_T:>9} {b['phi']:9.4f} {b['phi_penghantar']:11.4f} {b['komponen']:9d}"
              f" {b['menembus_xyz']:>9}")

    with open(os.path.join(U.HASIL, "tabel_ambang.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["geseran_voxel", "ambang_T", "phi", "phi_penghantar", "komponen",
                                          "menembus_xyz", "dipakai_sapuan"])
        w.writeheader(); w.writerows(baris)
    U.tulis_json(os.path.join(U.HASIL, "kelayakan.json"),
                 dict(ukuran_voxel_um=dx * 1e6, leher=leher, sigma_penghalusan=1.0,
                      geseran_sapuan=GESERAN_SAPUAN,
                      phi_sapuan=[b["phi"] for b in baris if b.get("dipakai_sapuan")]))


if __name__ == "__main__":
    main()
