"""Tahap 6. Uji ukuran domain bertingkat (volume elementer representatif).

Ukuran 64 dan 128 voxel: seluruh petak yang TIDAK BERTUMPANG (64 petak 64^3, 8 oktan 128^3).
Sudut acak tidak dipakai sebagai bawaan: pada kubus 256^3, setiap subvolume 128^3 bersudut acak pasti
memuat voxel pusat, sehingga daerah pusat terwakili berlebih (pada Bentheimer rerata phi delapan
subvolume acak 0,324, padahal phi domain 0,264). Pilihan --cara acak tetap tersedia sebagai pembanding.
Ukuran 256: domain penuh dari tahap 2 (satu sampel, sehingga C_v tidak terdefinisi di sana).
Pada tiap subvolume: porositas total, porositas penghantar, dan permeabilitas LBM tiga arah dengan
susunan domain yang sama dengan tahap 2. Subvolume yang tidak menembus dicatat sebagai K = 0 dan
dihitung pada lajur tersendiri, tidak dibuang diam-diam.

Koefisien variasi C_v = simpangan baku (pembagi n-1) / rerata, dibandingkan terhadap 5 persen.
Untuk K dilaporkan dua bentuk: atas subvolume yang menembus saja, dan atas seluruh subvolume.

Keluaran: hasil/rev_subvolume.csv, hasil/rev_ringkasan.csv
    python program/06_uji_rev.py [--cacah 8] [--backend auto]
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import umum as U  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cacah", type=int, default=8, help="hanya untuk --cara acak")
    ap.add_argument("--cara", default="petak", choices=["petak", "acak"])
    ap.add_argument("--ukuran", type=int, nargs="+", default=[64, 128])
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--benih", type=int, default=2026)
    arg = ap.parse_args()
    meta = U.muat_metadata(); dx = meta["ukuran_voxel_m"]
    padat = U.baca_citra(meta)
    n = padat.shape[0]
    rng = np.random.default_rng(arg.benih)
    akhiran = "" if arg.cara == "petak" else "_acak"
    jalur_sub = os.path.join(U.HASIL, f"rev_subvolume{akhiran}.csv")
    selesai = set()
    baris = []
    if os.path.exists(jalur_sub):
        with open(jalur_sub, encoding="utf-8") as f:
            for b in csv.DictReader(f):
                baris.append(b); selesai.add((int(b["ukuran"]), int(b["nomor"]), b["arah"]))

    kolom = ["ukuran", "nomor", "sudut_x", "sudut_y", "sudut_z", "arah", "phi", "phi_penghantar", "menembus",
             "K_mD", "langkah", "konvergen"]
    for L in arg.ukuran:
        if arg.cara == "acak":
            sudut = rng.integers(0, n - L + 1, size=(arg.cacah, 3))
        else:
            t = np.arange(0, n - L + 1, L)
            sudut = np.array([(i, j, k) for i in t for j in t for k in t])
        for i, (x0, y0, z0) in enumerate(sudut):
            sub = padat[x0:x0 + L, y0:y0 + L, z0:z0 + L]
            for a in "xyz":
                if (L, i, a) in selesai:
                    continue
                d, iris, pori = U.domain_aliran(sub, a)
                b = dict(ukuran=L, nomor=i, sudut_x=x0, sudut_y=y0, sudut_z=z0, arah=a, phi=float((~sub).mean()),
                         phi_penghantar=float(pori.mean()), menembus=bool(pori.any()), K_mD=0.0, langkah=0, konvergen=True)
                if pori.any():
                    r = U.jalankan_lbm(d, backend=arg.backend, cetak=lambda *_: None)
                    s = U.sifat_aliran(r, iris, pori, 4)
                    b.update(K_mD=s["k_kisi"] * dx * dx / U.DARCY_M2 * 1000, langkah=r["langkah"], konvergen=r["konvergen"])
                print(f"L {L:3d} #{i} {a}: phi {b['phi']:.4f} phi_c {b['phi_penghantar']:.4f} K {b['K_mD']:8.1f} mD "
                      f"langkah {b['langkah']}", flush=True)
                baris.append(b)
                with open(jalur_sub, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=kolom); w.writeheader(); w.writerows(baris)

    # domain penuh dari tahap 2
    for a in "xyz":
        with open(os.path.join(U.HASIL, "lbm", f"arah_{a}", "ringkasan.json"), encoding="utf-8") as f:
            s = json.load(f)
        baris.append(dict(ukuran=n, nomor=0, sudut_x=0, sudut_y=0, sudut_z=0, arah=a, phi=float((~padat).mean()),
                          phi_penghantar=s["phi_penghantar"], menembus=True, K_mD=s["K_mD"], langkah=s["langkah"],
                          konvergen=s["konvergen"]))

    def cv(v):
        v = np.asarray(v, float)
        return float(v.std(ddof=1) / v.mean()) if v.size > 1 and v.mean() > 0 else float("nan")

    ringkas = []
    print(f"\n{'L':>4} {'arah':>4} {'n':>3} {'tembus':>6} {'phi':>8} {'Cv phi %':>9} {'K rerata':>9} {'Cv K %':>8} {'Cv K semua %':>12}")
    for L in sorted({int(b["ukuran"]) for b in baris}):
        for a in "xyz":
            g = [b for b in baris if int(b["ukuran"]) == L and b["arah"] == a]
            phi = [float(b["phi"]) for b in g]
            K = [float(b["K_mD"]) for b in g]
            Kt = [k for k, b in zip(K, g) if str(b["menembus"]) in ("True", "true", "1")]
            r = dict(ukuran=L, arah=a, cacah=len(g), menembus=len(Kt), phi_rerata=float(np.mean(phi)), Cv_phi=cv(phi),
                     K_rerata_tembus=float(np.mean(Kt)) if Kt else 0.0, Cv_K_tembus=cv(Kt), Cv_K_semua=cv(K),
                     Cv_phi_di_bawah_5persen=cv(phi) < 0.05, Cv_K_di_bawah_5persen=cv(Kt) < 0.05)
            ringkas.append(r)
            print(f"{L:4d} {a:>4} {len(g):3d} {len(Kt):6d} {r['phi_rerata']:8.4f} {100 * r['Cv_phi']:9.2f} "
                  f"{r['K_rerata_tembus']:9.0f} {100 * r['Cv_K_tembus']:8.1f} {100 * r['Cv_K_semua']:12.1f}")
    with open(os.path.join(U.HASIL, f"rev_ringkasan{akhiran}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ringkas[0])); w.writeheader(); w.writerows(ringkas)
    print("\nCatatan: pada 256 hanya ada satu domain, sehingga C_v tidak terdefinisi (nan). "
          "Apabila C_v K pada 128 masih di atas 5 persen, volume elementer belum dapat dinyatakan tercapai pada 256.")


if __name__ == "__main__":
    main()
