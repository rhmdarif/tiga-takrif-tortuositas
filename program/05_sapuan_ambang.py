"""Tahap 5. Sapuan ambang segmentasi: langkah 5 sampai 8 diulang pada lima geseran antarmuka.

Geseran s = -0,5; -0,25; 0; +0,25; +0,5 voxel (lihat 01_kelayakan_dan_ambang.py). Pada tiap
geseran: LBM tiga arah (tau_h dan K simulasi), BFS dan Laplace tiga arah (tau_g, tau_e), luas
spesifik, dan tiga permeabilitas dugaan Kozeny dan Carman.

Pekerjaannya dipecah menjadi empat sub-perintah agar GPU (lbm) dan CPU (geometri) dapat berjalan
bersamaan dan dapat dilanjutkan bila terputus; tiap sub-perintah melewati hasil yang sudah ada.

    python program/05_sapuan_ambang.py siapkan
    python program/05_sapuan_ambang.py lbm         # GPU, sekitar 1,5 jam
    python program/05_sapuan_ambang.py geometri    # CPU, sekitar 1,2 jam, boleh bersamaan dengan lbm
    python program/05_sapuan_ambang.py gabung

    python program/05_sapuan_ambang.py elastisitas # hanya membaca sapuan_ambang.csv

Keluaran: hasil/sapuan/s_<geseran>/..., hasil/sapuan_ambang.csv, hasil/rentang_ambang.csv,
hasil/elastisitas_ambang.csv

Rentang: untuk tiap besaran, nilai tengah = (maks + min) / 2 dan lebar = (maks - min) / tengah,
dalam persen. Ketidakpastian baku dengan anggapan sebaran seragam (Pertemuan 13) adalah
u = (maks - min) / (2 sqrt 3), juga dinyatakan dalam persen terhadap nilai tengah.

Elastisitas: eta = d ln K / d ln phi_penghantar, yaitu kemiringan garis kuadrat terkecil ln K
terhadap ln phi_c atas lima geseran, per arah. Artinya: galat relatif 1 persen pada phi_c menjadi
sekitar eta persen pada K. Dihitung untuk K simulasi dan ketiga K dugaan Kozeny dan Carman, juga
untuk tiap tau (kepekaan tau terhadap phi_c). Dilaporkan pula R^2 dan kemiringan dua titik ujung
(s = -0,5 dan +0,5) sebagai pembanding.
"""

import argparse
import csv
import importlib.util
import json
import os
import subprocess
import sys

import numpy as np

DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, DIR)
import umum as U  # noqa: E402

GESERAN = [-0.5, -0.25, 0.0, 0.25, 0.5]
AKAR_SAPUAN = os.path.join(U.HASIL, "sapuan")


def _modul(nama_berkas):
    spec = importlib.util.spec_from_file_location(nama_berkas[:-3], os.path.join(DIR, nama_berkas))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def folder(s):
    return os.path.join(AKAR_SAPUAN, f"s_{s:+.2f}")


def siapkan():
    padat = U.baca_citra()
    abu = U.citra_abu_semu(padat)
    for s in GESERAN:
        os.makedirs(folder(s), exist_ok=True)
        p = U.geser_antarmuka(padat, s, abu=abu)
        np.save(os.path.join(folder(s), "padat.npy"), p)
        print(f"s = {s:+.2f}: phi {(~p).mean():.4f}")


def lbm(backend):
    for s in GESERAN:
        for a in "xyz":
            if os.path.exists(os.path.join(folder(s), "lbm", f"arah_{a}", "ringkasan.json")):
                print(f"s = {s:+.2f} arah {a}: sudah ada, dilewati"); continue
            print(f"s = {s:+.2f} arah {a}: menjalankan LBM", flush=True)
            subprocess.run([sys.executable, "-W", "ignore", os.path.join(DIR, "02_jalankan_simulasi.py"),
                            "--arah", a, "--backend", backend, "--padat-npy", os.path.join(folder(s), "padat.npy"),
                            "--keluaran", os.path.join(folder(s), "lbm")], check=True)


def geometri():
    t3 = _modul("03_tortuositas.py")
    for s in GESERAN:
        jalur = os.path.join(folder(s), "geometri.json")
        if os.path.exists(jalur):
            print(f"s = {s:+.2f}: sudah ada, dilewati"); continue
        padat = np.load(os.path.join(folder(s), "padat.npy"))
        print(f"s = {s:+.2f}", flush=True)
        hasil = [t3.takrif_satu_arah(padat, a, None, cetak=lambda m: print(m, flush=True)) for a in "xyz"]
        U.tulis_json(jalur, hasil)


def gabung():
    t4 = _modul("04_kozeny_carman.py")
    dx = U.muat_metadata()["ukuran_voxel_m"]
    baris = []
    for s in GESERAN:
        padat = np.load(os.path.join(folder(s), "padat.npy"))
        with open(os.path.join(folder(s), "geometri.json"), encoding="utf-8") as f:
            geo = {g["arah"]: g for g in json.load(f)}
        for a in "xyz":
            with open(os.path.join(folder(s), "lbm", f"arah_{a}", "ringkasan.json"), encoding="utf-8") as f:
                sim = json.load(f)
            pori = U.pori_penghantar(U.bingkai_aliran(padat, a))
            Ub = np.load(os.path.join(folder(s), "lbm", f"arah_{a}", "medan_u.npy")).astype(np.float64)
            tau_h = U.tortuositas_hidrolik(Ub, pori); del Ub
            S = t4.luas_spesifik(pori)["S_v_koreksi_2per3"]
            phi = float(pori.mean())
            g = geo[a]
            b = dict(geseran=s, arah=a, phi=float((~padat).mean()), phi_penghantar=phi, S_v_per_m=S / dx,
                     K_sim_mD=sim["K_mD"], konvergen=sim["konvergen"], langkah=sim["langkah"], Ma=sim["Ma"],
                     tau_h=tau_h, tau_g=g["tau_g"], tau_e=g["tau_e"], F=g["F"], m_cocok=g["m_cocok"],
                     tau_e_archie=g["tau_e_archie_phi_penghantar"])
            for nama in ("tau_h", "tau_g", "tau_e"):
                k = phi ** 3 / (t4.C0 * b[nama] ** 2 * S ** 2) * dx * dx / U.DARCY_M2 * 1000
                b[f"K_KC_{nama}_mD"] = k
                b[f"nisbah_KC_{nama}_thd_sim"] = k / sim["K_mD"]
            b["faktor_g_thd_h_kuadrat"] = (b["tau_g"] / b["tau_h"]) ** 2
            baris.append(b)
            print(f"s {s:+.2f} {a}: phi_c {phi:.4f} K {sim['K_mD']:7.0f} tau_h {tau_h:.4f} tau_g {g['tau_g']:.4f} "
                  f"tau_e {g['tau_e']:.4f} (tau_g/tau_h)^2 {b['faktor_g_thd_h_kuadrat']:.3f}")
    with open(os.path.join(U.HASIL, "sapuan_ambang.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(baris[0])); w.writeheader(); w.writerows(baris)

    besaran = ["phi_penghantar", "K_sim_mD", "tau_h", "tau_g", "tau_e", "K_KC_tau_h_mD", "K_KC_tau_g_mD",
               "K_KC_tau_e_mD", "faktor_g_thd_h_kuadrat"]
    rentang = []
    print(f"\n{'besaran':>22} {'arah':>4} {'s=0':>10} {'min':>10} {'maks':>10} {'tengah':>10} {'lebar %':>8} {'u %':>6}")
    for nama in besaran:
        for a in "xyz":
            v = np.array([b[nama] for b in baris if b["arah"] == a])
            nol = [b[nama] for b in baris if b["arah"] == a and b["geseran"] == 0.0][0]
            tengah = (v.max() + v.min()) / 2
            r = dict(besaran=nama, arah=a, nilai_s0=nol, minimum=v.min(), maksimum=v.max(), tengah=tengah,
                     lebar_persen=100 * (v.max() - v.min()) / tengah,
                     u_seragam_persen=100 * (v.max() - v.min()) / (2 * np.sqrt(3)) / tengah)
            rentang.append(r)
            print(f"{nama:>22} {a:>4} {nol:10.4g} {v.min():10.4g} {v.max():10.4g} {tengah:10.4g} "
                  f"{r['lebar_persen']:8.1f} {r['u_seragam_persen']:6.1f}")
    with open(os.path.join(U.HASIL, "rentang_ambang.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rentang[0])); w.writeheader(); w.writerows(rentang)
    elastisitas()


def elastisitas():
    with open(os.path.join(U.HASIL, "sapuan_ambang.csv"), encoding="utf-8") as f:
        baris = list(csv.DictReader(f))
    besaran = ["K_sim_mD", "K_KC_tau_h_mD", "K_KC_tau_g_mD", "K_KC_tau_e_mD", "tau_h", "tau_g", "tau_e", "F"]
    keluar = []
    print(f"\nElastisitas terhadap phi_penghantar (d ln Y / d ln phi_c)")
    print(f"{'besaran':>16} {'arah':>4} {'eta':>7} {'R2':>7} {'eta ujung':>9}")
    for nama in besaran:
        for a in "xyz":
            g = sorted((b for b in baris if b["arah"] == a), key=lambda b: float(b["geseran"]))
            x = np.log([float(b["phi_penghantar"]) for b in g])
            y = np.log([float(b[nama]) for b in g])
            eta, c = np.polyfit(x, y, 1)
            sisa = y - (eta * x + c)
            r2 = 1 - (sisa @ sisa) / ((y - y.mean()) @ (y - y.mean()))
            ujung = (y[-1] - y[0]) / (x[-1] - x[0])
            keluar.append(dict(besaran=nama, arah=a, cacah=len(g), eta=float(eta), R2=float(r2), eta_ujung=float(ujung)))
            print(f"{nama:>16} {a:>4} {eta:7.3f} {r2:7.4f} {ujung:9.3f}")
    with open(os.path.join(U.HASIL, "elastisitas_ambang.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(keluar[0])); w.writeheader(); w.writerows(keluar)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("perintah", choices=["siapkan", "lbm", "geometri", "gabung", "elastisitas"])
    ap.add_argument("--backend", default="auto")
    arg = ap.parse_args()
    {"siapkan": siapkan, "lbm": lambda: lbm(arg.backend), "geometri": geometri, "gabung": gabung,
     "elastisitas": elastisitas}[arg.perintah]()


if __name__ == "__main__":
    main()
