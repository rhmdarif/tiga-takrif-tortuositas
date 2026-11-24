"""Tahap 7. Seluruh gambar laporan, dibuat dari berkas pada hasil/ (tanpa menghitung ulang).

    python program/07_gambar.py

Gambar yang memerlukan hasil sapuan atau uji REV dilewati dengan pesan bila berkasnya belum ada.
Warna: tiga slot pertama palet kategorikal (biru, oranye, hijau laut) dalam urutan tetap
tau_h, tau_g, tau_e; bentuk tanda juga berbeda sehingga pembeda tidak bergantung pada warna saja.
"""

import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import umum as U  # noqa: E402

WARNA = {"tau_h": "#2a78d6", "tau_g": "#eb6834", "tau_e": "#1baf7a"}
TANDA = {"tau_h": "o", "tau_g": "^", "tau_e": "s"}
NAMA = {"tau_h": r"$\tau_h$ hidrolik", "tau_g": r"$\tau_g$ geometrik", "tau_e": r"$\tau_e=\sqrt{F\phi_c}$ listrik"}
ARAH_WARNA = {"x": "#2a78d6", "y": "#eb6834", "z": "#1baf7a"}
TINTA = "#3d3d3a"; TINTA2 = "#73726c"; KISI = "#e4e3dc"

plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 200, "font.size": 9, "axes.edgecolor": TINTA2,
                     "axes.labelcolor": TINTA, "xtick.color": TINTA2, "ytick.color": TINTA2, "text.color": TINTA,
                     "axes.grid": True, "grid.color": KISI, "grid.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 2})


def baca_csv(nama):
    jalur = os.path.join(U.HASIL, nama)
    if not os.path.exists(jalur):
        return None
    with open(jalur, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def simpan(fig, nama):
    os.makedirs(U.GAMBAR, exist_ok=True)
    fig.savefig(os.path.join(U.GAMBAR, nama), bbox_inches="tight")
    plt.close(fig)
    print("  gambar/" + nama)


def gambar_kc():
    kc = baca_csv("kozeny_carman.csv")
    sapuan = baca_csv("sapuan_ambang.csv")
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    lo, hi = 1.5e3, 2e4
    xs = np.array([lo, hi])
    ax.plot(xs, xs, color=TINTA2, lw=1); ax.plot(xs, 2 * xs, color=TINTA2, lw=1, ls="--")
    ax.plot(xs, xs / 2, color=TINTA2, lw=1, ls="--")
    ax.text(hi * 0.93, hi * 0.93 * 0.80, "1:1", ha="right", color=TINTA2, fontsize=8)
    ax.text(hi * 0.45, hi * 0.97, "2x", ha="right", color=TINTA2, fontsize=8)
    ax.text(hi * 0.93, hi * 0.93 / 2 * 0.80, "1/2x", ha="right", color=TINTA2, fontsize=8)
    for t in ("tau_h", "tau_g", "tau_e"):
        rows = [r for r in kc if r["takrif"] == t and r["luas"] == "S_v_koreksi_2per3"]
        x = [float(r["K_sim_mD"]) for r in rows]; y = [float(r["K_KC_mD"]) for r in rows]
        if sapuan:
            for a in "xyz":
                rs = sorted([r for r in sapuan if r["arah"] == a], key=lambda r: float(r["geseran"]))
                ax.plot([float(r["K_sim_mD"]) for r in rs], [float(r[f"K_KC_{t}_mD"]) for r in rs], color=WARNA[t],
                        lw=1, alpha=0.55, marker=TANDA[t], ms=3, zorder=2)
        ax.scatter(x, y, s=46, marker=TANDA[t], color=WARNA[t], edgecolor="white", linewidth=1.5, zorder=3,
                   label=NAMA[t])
        for r, xi, yi in zip(rows, x, y):
            geser = {"x": (-9, 3), "y": (6, -3), "z": (2, -10)}[r["arah"]]
            ax.annotate(r["arah"], (xi, yi), xytext=geser, textcoords="offset points", fontsize=7, color=TINTA2)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    titik = [2000, 3000, 5000, 10000, 20000]
    for sb in (ax.xaxis, ax.yaxis):
        sb.set_major_locator(matplotlib.ticker.FixedLocator(titik))
        sb.set_major_formatter(matplotlib.ticker.FixedFormatter([f"{t:,}".replace(",", ".") for t in titik]))
        sb.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("K simulasi LBM (mD)"); ax.set_ylabel("K dugaan Kozeny–Carman (mD)")
    ax.set_title("Bentheimer, tiga arah, $c_0=2$, $S_v$ terkoreksi 2/3", fontsize=9, loc="left")
    ax.legend(loc="lower right", fontsize=8)
    if sapuan:
        ax.text(0.03, 0.97, "titik besar: citra asli\ngaris tipis: sapuan ambang s = -0,5 ... +0,5 voxel",
                transform=ax.transAxes, va="top", fontsize=7, color=TINTA2)
    simpan(fig, "kc_terhadap_simulasi.png")


def gambar_riwayat():
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    for a in "xyz":
        r = np.loadtxt(os.path.join(U.HASIL, "lbm", f"arah_{a}", "riwayat.csv"), delimiter=",", skiprows=1)
        ax.semilogy(r[1:, 0], r[1:, 2], color=ARAH_WARNA[a], label=f"Bentheimer arah {a}")
    c = np.loadtxt(os.path.join(U.HASIL, "riwayat_celah_h16.csv"), delimiter=",", skiprows=1)
    ax.semilogy(c[1:, 0], c[1:, 2], color=TINTA2, lw=1.2, ls="--", label="celah sejajar h = 16")
    ax.axhline(1e-5, color=TINTA, lw=0.8, ls=":"); ax.text(16000, 1.4e-5, "patokan pedoman 1e-5", fontsize=7, color=TINTA, ha="right")
    ax.axhline(1e-6, color=TINTA2, lw=0.8, ls=":"); ax.text(16000, 4.5e-7, "patokan dipakai 1e-6", fontsize=7, color=TINTA2, ha="right", va="top")
    ax.set_xlabel("langkah waktu"); ax.set_ylabel(r"sisa: perubahan nisbi $\langle u\rangle$ per 100 langkah")
    ax.legend(fontsize=7, loc="upper right")
    simpan(fig, "riwayat_konvergensi.png")


def gambar_sebaran():
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for a in "xyz":
        r = np.loadtxt(os.path.join(U.HASIL, f"sebaran_laju_{a}.csv"), delimiter=",", skiprows=1)
        tengah = 0.5 * (r[:, 0] + r[:, 1]); lebar = r[:, 1] - r[:, 0]
        rapat = r[:, 2] / r[:, 2].sum() / lebar
        ax.loglog(tengah[rapat > 0], rapat[rapat > 0], color=ARAH_WARNA[a], label=f"arah {a}")
    ax.set_xlabel(r"laju nisbi $|u|/\langle|u|\rangle$ pada voxel penghantar"); ax.set_ylabel("rapat peluang")
    ax.legend(fontsize=8)
    simpan(fig, "sebaran_laju.png")


def gambar_sapuan():
    sap = baca_csv("sapuan_ambang.csv")
    if not sap:
        print("  (lewati gambar sapuan: hasil/sapuan_ambang.csv belum ada)"); return
    fig, axs = plt.subplots(1, 3, figsize=(8.6, 2.9), sharey=True)
    for ax, a in zip(axs, "xyz"):
        rows = sorted([r for r in sap if r["arah"] == a], key=lambda r: float(r["phi_penghantar"]))
        phi = [float(r["phi_penghantar"]) for r in rows]
        for t in ("tau_h", "tau_g", "tau_e"):
            ax.plot(phi, [float(r[t]) for r in rows], color=WARNA[t], marker=TANDA[t], ms=6,
                    markeredgecolor="white", label=NAMA[t])
        ax.set_title(f"arah {a}", fontsize=9, loc="left"); ax.set_xlabel(r"$\phi_c$ (penghantar)")
    axs[0].set_ylabel(r"$\tau$"); axs[0].legend(fontsize=7)
    simpan(fig, "tortuositas_terhadap_ambang.png")


def gambar_irisan():
    padat = U.baca_citra()
    Ub = np.load(os.path.join(U.HASIL, "lbm", "arah_x", "medan_u.npy"))
    k = padat.shape[2] // 2
    besar = np.sqrt((Ub[:, :, :, k].astype(float) ** 2).sum(0))
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.5))
    axs[0].imshow(~padat[:, :, k].T, origin="lower", cmap="gray"); axs[0].set_title("irisan z = 128 (putih = pori)", fontsize=9, loc="left")
    peta = plt.get_cmap("Blues").copy(); peta.set_bad("#9a9992")          # padatan abu-abu
    im = axs[1].imshow(np.ma.masked_where(padat[:, :, k], besar).T, origin="lower", cmap=peta,
                       vmin=0, vmax=np.quantile(besar[besar > 0], 0.995))
    axs[1].set_title(r"$|u|$, aliran arah x ke kanan (abu-abu = padatan)", fontsize=9, loc="left")
    for ax in axs:
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    fig.colorbar(im, ax=axs[1], shrink=0.8, label="laju (satuan kisi)")
    simpan(fig, "irisan_dan_medan.png")


def gambar_rev():
    rev = baca_csv("rev_ringkasan.csv")
    if not rev:
        print("  (lewati gambar REV: hasil/rev_ringkasan.csv belum ada)"); return
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    rows = [r for r in rev if r["Cv_phi"] != "nan"]
    L = sorted({int(r["ukuran"]) for r in rows})
    cphi = [100 * float([r for r in rows if int(r["ukuran"]) == l][0]["Cv_phi"]) for l in L]
    ax.loglog(L, cphi, color=TINTA, marker="D", ms=6, label=r"$C_v$ porositas")
    for a in "xyz":
        ck = [100 * float([r for r in rows if int(r["ukuran"]) == l and r["arah"] == a][0]["Cv_K_tembus"]) for l in L]
        ax.loglog(L, ck, color=ARAH_WARNA[a], marker="o", ms=6, label=f"$C_v$ K arah {a}")
    Lx = np.array([64, 256])
    ax.loglog(Lx, cphi[0] * (Lx / 64.0) ** -1.5, color=TINTA2, lw=1, ls="--", label=r"$\propto L^{-3/2}$ dari 64")
    ax.axhline(5, color=TINTA, lw=0.8, ls=":"); ax.text(250, 5.6, "patokan 5 %", fontsize=7, color=TINTA, ha="right")
    ax.set_xticks([64, 128, 256]); ax.set_xticklabels(["64", "128", "256"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("sisi subvolume (voxel)"); ax.set_ylabel(r"$C_v$ (%)")
    ax.set_title("petak tak bertumpang: 64 petak 64³, 8 oktan 128³", fontsize=8, loc="left")
    ax.set_ylim(3, 400); ax.set_xlim(55, 300)
    ax.legend(fontsize=7, loc="upper right")
    simpan(fig, "uji_rev.png")


def main():
    gambar_kc(); gambar_riwayat(); gambar_sebaran(); gambar_sapuan(); gambar_irisan(); gambar_rev()


if __name__ == "__main__":
    main()
