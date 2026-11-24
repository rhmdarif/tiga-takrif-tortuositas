"""Tahap 8. Animasi untuk paparan: aliran yang berkembang dan penyusuran irisan medan kecepatan.

    python program/08_animasi.py berkembang [--arah y]      # menjalankan ulang LBM, 5-10 menit di GPU
    python program/08_animasi.py berkembang --hanya-gambar  # menggambar ulang dari rekaman yang ada
    python program/08_animasi.py irisan [--arah x]          # dari hasil/lbm/arah_*/medan_u.npy, 1-2 menit

berkembang: LBM D3Q19 TRT pada domain penuh, dengan pemecah, susunan domain, dan patokan berhenti
yang sama persis dengan tahap 2 (perubahan nisbi <u> per 100 langkah < 1e-6). Satu irisan |u| pada
tengah sumbu z direkam pada langkah berjarak logaritmik (5, 6, 8, ...), karena aliran pada skala
pori terbentuk dalam ratusan langkah pertama dan sisanya penyempurnaan medan tekanan. Sebagai
pemeriksa, permeabilitas penjalanan ini dibandingkan dengan ringkasan tahap 2.

irisan: irisan z = 0 ... 255 (tiap dua irisan) dari medan kecepatan tahap 2 yang sudah konvergen,
dengan tata letak dan warna yang sama dengan gambar/irisan_dan_medan.png.

Keluaran: gambar/animasi_berkembang_<arah>.gif, gambar/animasi_irisan_<arah>.gif,
hasil/animasi/berkembang_<arah>.json dan .csv (ringkasan dan riwayat), serta rekaman irisan
hasil/animasi/berkembang_<arah>.npz (tidak ikut dikemas; dapat dihasilkan ulang).
"""
import argparse
import csv
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import umum as U  # noqa: E402

TINTA = "#3d3d3a"; TINTA2 = "#73726c"; KISI = "#e4e3dc"; PADATAN = "#9a9992"
WARNA_ARAH = {"x": "#2a78d6", "y": "#eb6834", "z": "#1baf7a"}
plt.rcParams.update({"font.size": 11, "axes.edgecolor": TINTA2, "axes.labelcolor": TINTA,
                     "xtick.color": TINTA2, "ytick.color": TINTA2, "text.color": TINTA,
                     "axes.grid": True, "grid.color": KISI, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2})
FOLDER = os.path.join(U.HASIL, "animasi")


def peta_biru():
    peta = plt.get_cmap("Blues").copy()
    peta.set_bad(PADATAN)                     # padatan dan pori tak menghantar abu-abu
    return peta


def ke_gambar(fig):
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    return Image.fromarray(rgba[..., :3].copy())


def simpan_gif(bingkai, durasi, nama):
    os.makedirs(U.GAMBAR, exist_ok=True)
    jalur = os.path.join(U.GAMBAR, nama)
    # satu palet bersama untuk seluruh bingkai agar warna tidak berkedip antarbingkai
    contoh = Image.new("RGB", (bingkai[0].width, bingkai[0].height * 3))
    for i, j in enumerate((0, len(bingkai) // 2, len(bingkai) - 1)):
        contoh.paste(bingkai[j], (0, i * bingkai[0].height))
    palet = contoh.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    p = [b.quantize(palette=palet, dither=Image.Dither.NONE) for b in bingkai]
    p[0].save(jalur, save_all=True, append_images=p[1:], duration=durasi, loop=0, optimize=True)
    print(f"  gambar/{nama}: {len(p)} bingkai, {os.path.getsize(jalur) / 1e6:.1f} MB")


# --------------------------------------------------------------------------- berkembang

def jalankan_berkembang(arah, backend, tol=1e-6, maks=60000, penyangga=4):
    meta = U.muat_metadata()
    padat, iris, pori = U.domain_aliran(U.baca_citra(meta), arah, penyangga=penyangga, dinding=True)
    kz = iris[2].start + (iris[2].stop - iris[2].start) // 2          # tengah sumbu ke-2 batuan
    p = U.PemecahLBM(padat, backend=backend)
    idx = -np.ones(padat.shape, np.int64)
    idx[p.alir] = np.arange(p.n)
    simpul = idx[iris[0], iris[1], kz]                                 # (N, lintang), -1 = padatan
    ada = simpul >= 0
    print(f"Arah {arah}: domain {padat.shape}, simpul fluida {p.n:,}, backend {p.backend}, irisan ke-{kz}")

    def irisan_laju():
        u, _ = p.momen()
        out = np.full(simpul.shape, np.nan, np.float32)
        out[ada] = np.sqrt((u[:, simpul[ada]] ** 2).sum(0))
        return out, float(u[0].mean())

    # Bingkai berjarak logaritmik: aliran pada skala pori terbentuk dalam ratusan langkah pertama,
    # sisanya penyempurnaan medan tekanan. Pemeriksaan konvergensi tetap tiap 100 langkah.
    jadwal = sorted(set(int(round(v)) for v in np.geomspace(5, maks, 110)))
    bingkai, langkah_bingkai, ux_bingkai, riwayat = [], [], [], []
    k = 0; ji = 0; cek = 100; ux_lama = 0.0; konvergen = False; t0 = time.time()
    while k < maks:
        sasaran = min(cek, jadwal[ji]) if ji < len(jadwal) else cek
        p.langkah(sasaran - k); k = sasaran
        rekam = ji < len(jadwal) and k == jadwal[ji]
        if not (rekam or k == cek):
            continue
        laju, ux = irisan_laju()
        if rekam:
            bingkai.append(laju); langkah_bingkai.append(k); ux_bingkai.append(ux); ji += 1
        if k == cek:                                                    # patokan sama dengan tahap 2
            cek += 100
            ubah = abs(ux - ux_lama) / max(abs(ux), 1e-300)
            riwayat.append((k, ux, ubah)); ux_lama = ux
            if k % 2000 == 0:
                print(f"  langkah {k:6d}  <u> {ux:.6e}  sisa {ubah:.2e}  {time.time() - t0:6.0f} s")
            if k >= 1000 and ubah < tol:
                konvergen = True
                break
    if langkah_bingkai[-1] != k:
        laju, ux = irisan_laju(); bingkai.append(laju); langkah_bingkai.append(k); ux_bingkai.append(ux)
    detik = time.time() - t0

    # pemeriksa: permeabilitas penjalanan ini terhadap ringkasan tahap 2
    Ufull, R = p.medan()
    s = U.sifat_aliran(dict(U=Ufull, R=R, nu=p.nu, g=p.g), iris, pori, penyangga)
    k_mD = s["k_kisi"] * meta["ukuran_voxel_m"] ** 2 / U.DARCY_M2 * 1000
    acuan = os.path.join(U.HASIL, "lbm", f"arah_{arah}", "ringkasan.json")
    k_tahap2 = json.load(open(acuan))["K_mD"] if os.path.exists(acuan) else None

    os.makedirs(FOLDER, exist_ok=True)
    np.savez_compressed(os.path.join(FOLDER, f"berkembang_{arah}.npz"), bingkai=np.array(bingkai),
                        langkah=np.array(langkah_bingkai), ux=np.array(ux_bingkai), riwayat=np.array(riwayat))
    with open(os.path.join(FOLDER, f"berkembang_{arah}.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["langkah", "rerata_u_aliran", "sisa"]); w.writerows(riwayat)
    ringkas = dict(arah=arah, backend=p.backend, langkah=k, konvergen=konvergen, detik=detik,
                   irisan_sumbu2=int(kz - iris[2].start), cacah_bingkai=len(bingkai),
                   K_mD=k_mD, K_mD_tahap2=k_tahap2,
                   selisih_nisbi_thd_tahap2=(k_mD / k_tahap2 - 1) if k_tahap2 else None)
    U.tulis_json(os.path.join(FOLDER, f"berkembang_{arah}.json"), ringkas)
    print(f"  selesai: langkah {k}, konvergen {konvergen}, {detik / 60:.1f} menit")
    print(f"  K = {k_mD:.1f} mD (tahap 2: {k_tahap2:.1f} mD)" if k_tahap2 else f"  K = {k_mD:.1f} mD")


def gambar_berkembang(arah):
    d = np.load(os.path.join(FOLDER, f"berkembang_{arah}.npz"))
    bingkai, langkah, ux_b, riwayat = d["bingkai"], d["langkah"], d["ux"], d["riwayat"]
    ringkas = json.load(open(os.path.join(FOLDER, f"berkembang_{arah}.json")))
    bingkai = bingkai * 1e4                                             # tampil dalam satuan 1e-4
    akhir = bingkai[-1]
    vmax = float(np.nanquantile(akhir[akhir > 0], 0.995))
    ux_akhir = ux_b[-1]
    kmax = langkah[-1]
    peta = peta_biru()
    gambar = []
    for laju, k in zip(bingkai, langkah):
        fig = plt.figure(figsize=(9.6, 6.0), dpi=100)
        gs = fig.add_gridspec(2, 2, width_ratios=[1.55, 1], height_ratios=[1, 1],
                              left=0.03, right=0.97, top=0.86, bottom=0.10, wspace=0.30, hspace=0.55)
        ax = fig.add_subplot(gs[:, 0])
        im = ax.imshow(np.ma.masked_invalid(laju).T, origin="lower", cmap=peta, vmin=0, vmax=vmax,
                       interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        ax.set_xlabel(f"arah aliran {arah} →", fontsize=11)
        cb = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
        cb.set_label("laju |u| (×10⁻⁴ satuan kisi)", fontsize=10); cb.ax.tick_params(labelsize=8)

        sel = langkah <= k
        a1 = fig.add_subplot(gs[0, 1])
        a1.semilogx(langkah, ux_b / ux_akhir, color=KISI, lw=2)
        a1.semilogx(langkah[sel], ux_b[sel] / ux_akhir, color=WARNA_ARAH[arah], lw=2)
        a1.plot([k], [ux_b[sel][-1] / ux_akhir], "o", color=WARNA_ARAH[arah], ms=6)
        a1.set_ylim(0, 1.15); a1.set_xlim(langkah[0], kmax)
        a1.set_title("kecepatan rerata / nilai akhir", fontsize=10, loc="left")
        a1.tick_params(labelsize=9)
        a2 = fig.add_subplot(gs[1, 1])
        a2.loglog(riwayat[1:, 0], riwayat[1:, 2], color=KISI, lw=2)
        s2 = riwayat[1:, 0] <= k
        if s2.any():
            a2.loglog(riwayat[1:, 0][s2], riwayat[1:, 2][s2], color=WARNA_ARAH[arah], lw=2)
        a2.axhline(1e-6, color=TINTA2, lw=1, ls="--")
        a2.text(kmax, 1.6e-6, "patokan 10⁻⁶", ha="right", va="bottom", fontsize=9, color=TINTA2)
        a2.set_xlim(langkah[0], kmax); a2.set_ylim(5e-7, 1)
        a2.set_title("sisa konvergensi (tiap 100 langkah)", fontsize=10, loc="left")
        a2.set_xlabel("langkah waktu (skala log)", fontsize=10)
        a2.tick_params(labelsize=9)

        fig.text(0.03, 0.95, f"Simulasi LBM D3Q19 pada Bentheimer 256³, aliran arah {arah}",
                 fontsize=14, fontweight="bold", va="top")
        fig.text(0.03, 0.905, f"langkah {k:,}".replace(",", " ") + f" dari {int(riwayat[-1, 0]):,}".replace(",", " ")
                 + f" · irisan z = {ringkas['irisan_sumbu2']} · abu-abu = padatan",
                 fontsize=11, color=TINTA2, va="top")
        gambar.append(ke_gambar(fig)); plt.close(fig)
    durasi = [110] * len(gambar)
    durasi[-1] = 2500                                                   # bingkai akhir ditahan
    simpan_gif(gambar, durasi, f"animasi_berkembang_{arah}.gif")


# --------------------------------------------------------------------------- irisan

def gambar_irisan(arah, tiap=2):
    padat = U.baca_citra()
    Ub = np.load(os.path.join(U.HASIL, "lbm", f"arah_{arah}", "medan_u.npy"))   # bingkai aliran
    besar = np.sqrt((Ub.astype(np.float32) ** 2).sum(0)) * 1e4         # tampil dalam satuan 1e-4
    del Ub
    pb = U.bingkai_aliran(padat, arah)
    penghantar = U.pori_penghantar(pb)
    vmax = float(np.quantile(besar[penghantar][::50], 0.995))
    peta = peta_biru()
    dx_um = U.muat_metadata()["ukuran_voxel_m"] * 1e6
    gambar = []
    for k in range(0, pb.shape[2], tiap):
        fig, axs = plt.subplots(1, 2, figsize=(9.0, 4.15), dpi=100)
        fig.subplots_adjust(left=0.01, right=0.88, top=0.82, bottom=0.03, wspace=0.05)
        axs[0].imshow(~pb[:, :, k].T, origin="lower", cmap="gray", interpolation="nearest")
        axs[0].set_title("citra: putih = pori", fontsize=11, loc="left")
        im = axs[1].imshow(np.ma.masked_where(~penghantar[:, :, k], besar[:, :, k]).T, origin="lower",
                           cmap=peta, vmin=0, vmax=vmax, interpolation="nearest")
        axs[1].set_title(f"|u|, aliran arah {arah} ke kanan", fontsize=11, loc="left")
        for ax in axs:
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        cax = fig.add_axes([0.895, 0.10, 0.015, 0.62])
        cb = fig.colorbar(im, cax=cax); cb.ax.tick_params(labelsize=8)
        cb.set_label("laju (×10⁻⁴ satuan kisi)", fontsize=9)
        fig.text(0.01, 0.97, f"Bentheimer 256³: irisan z = {k} ({k * dx_um:.0f} µm dari {pb.shape[2] * dx_um:.0f} µm)",
                 fontsize=13, fontweight="bold", va="top")
        gambar.append(ke_gambar(fig)); plt.close(fig)
    simpan_gif(gambar, 90, f"animasi_irisan_{arah}.gif")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("perintah", choices=["berkembang", "irisan"])
    ap.add_argument("--arah", default=None, choices=list("xyz"),
                    help="bawaan: y untuk berkembang (paling cepat konvergen), x untuk irisan (sama dengan Gambar 1)")
    ap.add_argument("--backend", default="auto", choices=["auto", "cpu", "gpu"])
    ap.add_argument("--hanya-gambar", action="store_true", help="berkembang: pakai rekaman yang sudah ada")
    arg = ap.parse_args()
    if arg.perintah == "berkembang":
        arah = arg.arah or "y"
        if not arg.hanya_gambar:
            jalankan_berkembang(arah, arg.backend)
        gambar_berkembang(arah)
    else:
        gambar_irisan(arg.arah or "x")


if __name__ == "__main__":
    main()
