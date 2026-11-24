"""Tahap 9. Visual 3D untuk paparan: permukaan pori, garis arus dari medan LBM, dan lintasan terpendek BFS.

    python program/09_visual3d.py data   # data penampil web -> hasil/visual3d/ (sekitar 1 menit)
    python program/09_visual3d.py gif    # kubus berputar -> gambar/animasi_3d_<arah>.gif (beberapa menit)

Semua dihitung dari hasil yang sudah ada (tanpa simulasi ulang):
  * permukaan pori: marching cubes atas pori PENGHANTAR yang dikasarkan (rerata blok 3x3x3 untuk web,
    4x4x4 untuk GIF), dengan tepi kubus tertutup sehingga tampak sebagai cetakan ruang pori;
  * garis arus: ditelusuri pada medan kecepatan tahap 2 (hasil/lbm/arah_*/medan_u.npy) dengan resolusi
    penuh, Runge-Kutta orde dua, langkah 0,5 voxel, interpolasi trilinear. Titik awal disebar pada muka
    masuk dengan peluang sebanding fluks u_aliran, sehingga kerapatan garis mengikuti fluks;
  * lintasan terpendek BFS 6 tetangga: direkonstruksi dari peta jarak umum.tortuositas_geometrik, dan
    panjangnya diperiksa terhadap tau_g_min pada hasil/tortuositas.json.
Koordinat dalam satuan voxel pada bingkai aliran (sumbu ke-0 = arah aliran); 1 voxel = 2,25 um.
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection  # noqa: E402
from PIL import Image  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402
from skimage import measure  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import umum as U  # noqa: E402

FOLDER = os.path.join(U.HASIL, "visual3d")
FAKTOR_WEB = 3      # rerata blok 3x3x3 untuk penampil web (sekitar 300 ribu segitiga)


def muat(arah):
    padat = U.baca_citra()
    pb = U.bingkai_aliran(padat, arah)
    pori = U.pori_penghantar(pb)
    Ub = np.load(os.path.join(U.HASIL, "lbm", f"arah_{arah}", "medan_u.npy"))
    return pori, Ub


def permukaan(pori, faktor):
    """Marching cubes atas pori penghantar yang dikasarkan; titik dalam koordinat voxel asli."""
    n = pori.shape[0] // faktor
    blok = pori[:n * faktor, :n * faktor, :n * faktor].reshape(n, faktor, n, faktor, n, faktor)
    kasar = blok.mean(axis=(1, 3, 5)).astype(np.float32)
    kasar = np.pad(kasar, 1, constant_values=0.0)                     # tutup tepi kubus
    v, f, _, _ = measure.marching_cubes(kasar, level=0.5)
    v = (v - 1.0) * faktor + (faktor - 1) / 2.0
    return v.astype(np.float32), f.astype(np.uint32)


def garis_arus(pori, Ub, cacah=360, ds=0.5, maks_langkah=4000, benih=2026):
    """Garis arus dari muka masuk; mengembalikan daftar larik (titik, 5): x, y, z, laju, waktu."""
    N = pori.shape[0]
    rng = np.random.default_rng(benih)
    ux0 = np.where(pori[0], np.clip(Ub[0, 0], 0, None), 0.0).astype(np.float64)
    calon = np.flatnonzero(ux0)
    pilih = rng.choice(calon, size=min(cacah, calon.size), replace=False, p=ux0.ravel()[calon] / ux0.ravel()[calon].sum())
    yz = np.column_stack(np.unravel_index(pilih, ux0.shape)).astype(np.float64)
    yz += rng.uniform(-0.35, 0.35, yz.shape)
    p = np.column_stack([np.zeros(len(yz)), yz])                       # (m, 3)
    V = [Ub[a].astype(np.float32) for a in range(3)]
    batas = np.array(pori.shape, float) - 1.0

    def kecepatan(q):
        q = np.clip(q, 0.0, batas)
        return np.column_stack([ndi.map_coordinates(V[a], q.T, order=1, mode="nearest") for a in range(3)])

    jejak = [[] for _ in range(len(p))]
    aktif = np.arange(len(p))
    laju_ref = float(np.median(np.sqrt((Ub[:, pori[:]] ** 2).sum(0))[::97]))
    waktu = np.zeros(len(p))
    for _ in range(maks_langkah):
        if aktif.size == 0:
            break
        q = p[aktif]
        u1 = kecepatan(q); s1 = np.linalg.norm(u1, axis=1)
        for j, i in enumerate(aktif):
            jejak[i].append((q[j, 0], q[j, 1], q[j, 2], s1[j], waktu[i]))
        mati = s1 < 1e-12
        arah1 = u1 / np.maximum(s1, 1e-30)[:, None]
        u2 = kecepatan(q + 0.5 * ds * arah1); s2 = np.linalg.norm(u2, axis=1)
        arah2 = u2 / np.maximum(s2, 1e-30)[:, None]
        baru = q + ds * arah2
        waktu[aktif] += ds / np.maximum(0.5 * (s1 + s2), 0.02 * laju_ref)
        p[aktif] = baru
        keluar = (baru[:, 0] >= N - 1) | (baru[:, 0] < -0.5) | np.any(baru[:, 1:] < -0.5, axis=1) \
            | np.any(baru[:, 1:] > batas[1:] + 0.5, axis=1)
        aktif = aktif[~(keluar | mati)]
    return [np.array(t, np.float32) for t in jejak if len(t) > 10], laju_ref


def jalur_terpendek(pori):
    g = U.tortuositas_geometrik(pori)
    jarak = g["jarak"]
    N = pori.shape[0]
    akhir = np.where(pori[-1], jarak[-1], -1)
    akhir = np.where(akhir >= 0, akhir, np.iinfo(np.int32).max)
    y, z = np.unravel_index(np.argmin(akhir), akhir.shape)
    titik = [(N - 1, y, z)]
    x = N - 1
    while jarak[x, y, z] > 0:
        d = jarak[x, y, z]
        for dx, dy, dz in ((-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1), (1, 0, 0)):
            xx, yy, zz = x + dx, y + dy, z + dz
            if 0 <= xx < N and 0 <= yy < pori.shape[1] and 0 <= zz < pori.shape[2] and jarak[xx, yy, zz] == d - 1:
                x, y, z = xx, yy, zz
                break
        titik.append((x, y, z))
    jalur = np.array(titik[::-1], np.float32)
    return jalur, g


# --------------------------------------------------------------------------- data web

def buat_data(arah):
    pori, Ub = muat(arah)
    os.makedirs(FOLDER, exist_ok=True)
    v, f = permukaan(pori, FAKTOR_WEB)
    v.tofile(os.path.join(FOLDER, "pori_posisi.bin")); f.tofile(os.path.join(FOLDER, "pori_indeks.bin"))
    print(f"  permukaan pori: {len(v):,} titik, {len(f):,} segitiga")

    garis, laju_ref = garis_arus(pori, Ub)
    garis = [g[::2] if len(g) > 40 else g for g in garis]              # cukup tiap 1 voxel untuk web
    awal = np.cumsum([0] + [len(g) for g in garis]).astype(np.uint32)
    np.concatenate(garis).astype(np.float32).tofile(os.path.join(FOLDER, "garis_titik.bin"))
    awal.tofile(os.path.join(FOLDER, "garis_awal.bin"))
    semua_laju = np.concatenate([g[:, 3] for g in garis])
    print(f"  garis arus: {len(garis)} garis, {int(awal[-1]):,} titik")

    jalur, g = jalur_terpendek(pori)
    jalur.tofile(os.path.join(FOLDER, "jalur_bfs.bin"))
    N = pori.shape[0]
    tau_min = len(jalur) / N                                           # cacah voxel / N
    acuan = json.load(open(os.path.join(U.HASIL, "tortuositas.json")))
    acuan_min = [r for r in acuan["per_arah"] if r["arah"] == arah][0]["tau_g_min"]
    print(f"  lintasan terpendek BFS: {len(jalur)} voxel, tau_g_min {tau_min:.5f} (tortuositas.json: {acuan_min:.5f})")

    # panjang garis arus / N, sebagai pembanding tau_g dan tau_h (bukan takrif yang dilaporkan)
    panjang = np.array([np.linalg.norm(np.diff(t[:, :3], axis=0), axis=1).sum() for t in garis])
    sampai = np.array([t[-1, 0] >= N - 2 for t in garis])
    meta = dict(arah=arah, ukuran=list(pori.shape), ukuran_voxel_um=U.muat_metadata()["ukuran_voxel_m"] * 1e6,
                faktor_kasar_permukaan=FAKTOR_WEB, cacah_titik_permukaan=int(len(v)), cacah_segitiga=int(len(f)),
                cacah_garis=len(garis), cacah_titik_garis=int(awal[-1]),
                laju_kuantil_995=float(np.quantile(semua_laju, 0.995)), laju_median_pori=laju_ref,
                waktu_maks=float(max(t[-1, 4] for t in garis)),
                cacah_voxel_jalur_bfs=int(len(jalur)), tau_g_min_jalur=tau_min, tau_g_min_tortuositas_json=acuan_min,
                garis_sampai_keluar=int(sampai.sum()),
                rerata_panjang_garis_per_N=float(panjang[sampai].mean() / N) if sampai.any() else None,
                berkas={"pori_posisi.bin": "float32 (titik, 3)", "pori_indeks.bin": "uint32 (segitiga, 3)",
                        "garis_titik.bin": "float32 (titik, 5): x, y, z, laju, waktu",
                        "garis_awal.bin": "uint32 (garis + 1): indeks titik awal tiap garis",
                        "jalur_bfs.bin": "float32 (voxel, 3)"})
    U.tulis_json(os.path.join(FOLDER, "visual3d.json"), meta)
    print(f"  garis yang sampai muka keluar: {int(sampai.sum())} dari {len(garis)}; "
          f"rerata panjang/N {meta['rerata_panjang_garis_per_N']:.3f}")


# --------------------------------------------------------------------------- GIF berputar

def buat_gif(arah, bingkai=60):
    pori, Ub = muat(arah)
    v, f = permukaan(pori, 4)
    garis, _ = garis_arus(pori, Ub, cacah=160)
    jalur, _ = jalur_terpendek(pori)
    del Ub
    N = pori.shape[0]
    semua = np.concatenate([g[:, 3] for g in garis])
    vmax = float(np.quantile(semua, 0.99))
    peta = matplotlib.colors.LinearSegmentedColormap.from_list(
        "laju", ["#9cc3e6", "#2a78d6", "#0b2f66"])
    seg, warna = [], []
    for g in garis:
        g = g[::2]
        s = np.stack([g[:-1, :3], g[1:, :3]], axis=1)
        seg.append(s); warna.append(np.clip(0.5 * (g[:-1, 3] + g[1:, 3]) / vmax, 0, 1))
    seg = np.concatenate(seg); warna = peta(np.concatenate(warna))
    print(f"  GIF: {len(f):,} segitiga permukaan, {len(garis)} garis arus, {len(seg):,} ruas")

    gambar = []
    for i in range(bingkai):
        fig = plt.figure(figsize=(6.4, 5.4), dpi=100)
        ax = fig.add_axes([0, 0, 1, 0.9], projection="3d")
        ax.computed_zorder = False                                     # lintasan BFS selalu di atas
        ax.add_collection3d(Poly3DCollection(v[f], facecolor="#c9b89a", edgecolor="none", alpha=0.10, zorder=1))
        ax.add_collection3d(Line3DCollection(seg, colors=warna, linewidths=0.9, zorder=2))
        ax.plot(jalur[:, 0], jalur[:, 1], jalur[:, 2], color="#eb6834", lw=3.2, zorder=10)
        for a in (0, N):                                               # rusuk kubus
            for b in (0, N):
                ax.plot([0, N], [a, a], [b, b], color="#b5b3aa", lw=0.6)
                ax.plot([a, a], [0, N], [b, b], color="#b5b3aa", lw=0.6)
                ax.plot([a, a], [b, b], [0, N], color="#b5b3aa", lw=0.6)
        ax.set_xlim(0, N); ax.set_ylim(0, N); ax.set_zlim(0, N)
        ax.set_box_aspect((1, 1, 1)); ax.set_axis_off()
        ax.view_init(elev=22, azim=-60 + 360 * i / bingkai)
        fig.text(0.03, 0.965, f"Bentheimer 256³ (0,576 mm), aliran arah {arah}", fontsize=13,
                 fontweight="bold", va="top", color="#3d3d3a")
        fig.text(0.03, 0.915, "biru: garis arus LBM (gelap = cepat) · jingga: lintasan terpendek BFS",
                 fontsize=10, va="top", color="#73726c")
        fig.canvas.draw()
        gambar.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()))
        plt.close(fig)
    contoh = gambar[0]
    palet = contoh.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    p = [g.quantize(palette=palet, dither=Image.Dither.NONE) for g in gambar]
    os.makedirs(U.GAMBAR, exist_ok=True)
    jalur_gif = os.path.join(U.GAMBAR, f"animasi_3d_{arah}.gif")
    p[0].save(jalur_gif, save_all=True, append_images=p[1:], duration=90, loop=0, optimize=True)
    print(f"  gambar/animasi_3d_{arah}.gif: {len(p)} bingkai, {os.path.getsize(jalur_gif) / 1e6:.1f} MB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("perintah", choices=["data", "gif"])
    ap.add_argument("--arah", default="x", choices=list("xyz"))
    ap.add_argument("--bingkai", type=int, default=60)
    arg = ap.parse_args()
    if arg.perintah == "data":
        buat_data(arg.arah)
    else:
        buat_gif(arg.arah, arg.bingkai)


if __name__ == "__main__":
    main()
