"""Pustaka bersama proyek Tiga Takrif Tortuositas pada Satu Batuan.

Seluruh program pada folder ini memakai fungsi dari berkas ini, sehingga takrif yang sama
dipakai di setiap tahap. Perjanjian yang berlaku di seluruh berkas:

* Larik `padat` bernilai True pada voxel padatan; `pori` adalah kebalikannya.
* Sumbu x, y, z adalah indeks larik ke-0, ke-1, ke-2 (lihat data/metadata.json).
* Setiap hitungan berarah dikerjakan pada "bingkai aliran": larik diputar dengan
  np.moveaxis sehingga arah aliran selalu menjadi sumbu ke-0.
* Ruang pori penghantar = kluster pori bersambungan muka (6 tetangga) yang menyentuh
  muka masuk dan muka keluar, tanpa pembungkusan periodik pada arah melintang. Takrif ini
  menghasilkan ulang porositas tersambung acuan pada data/sebelas-batupasir.csv tepat
  sampai enam desimal.
* Ketiga tortuositas dihitung pada himpunan voxel penghantar yang sama.
"""

import json
import os
import time

import numpy as np
from scipy import ndimage as ndi
from scipy import sparse
from scipy.sparse.linalg import cg

try:
    import numba as nb
except ImportError as e:  # pragma: no cover
    raise SystemExit("numba belum terpasang. Jalankan: pip install -r lingkungan.txt") from e

AKAR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(AKAR, "data")
HASIL = os.path.join(AKAR, "hasil")
GAMBAR = os.path.join(AKAR, "gambar")

DARCY_M2 = 9.869233e-13          # 1 darcy dalam m^2, sama dengan yang dipakai berkas acuan
CS2 = 1.0 / 3.0                  # kuadrat laju bunyi kisi
SUMBU = "xyz"


# --------------------------------------------------------------------------- data

def muat_metadata(jalur=None):
    with open(jalur or os.path.join(DATA, "metadata.json"), encoding="utf-8") as f:
        return json.load(f)


def baca_citra(meta=None):
    """Mengembalikan larik padat (True = padatan) berbentuk (nx, ny, nz)."""
    meta = meta or muat_metadata()
    jalur = os.path.join(DATA, meta["berkas"])
    tipe = np.dtype(meta["tipe_data"]).newbyteorder("<" if meta["urutan_byte"] == "little" else ">")
    v = np.fromfile(jalur, dtype=tipe)
    n = int(np.prod(meta["ukuran"]))
    if v.size != n:
        raise ValueError(f"ukuran berkas {v.size} tidak sama dengan {n}; periksa metadata")
    citra = v.reshape(meta["ukuran"], order=meta["urutan_indeks"])
    nilai_pori = int([k for k, arti in meta["arti_nilai"].items() if arti == "pori"][0])
    return citra != nilai_pori


def bingkai_aliran(larik, arah):
    """Memutar larik sehingga arah aliran ('x', 'y', 'z' atau 0, 1, 2) menjadi sumbu ke-0."""
    a = SUMBU.index(arah) if isinstance(arah, str) else int(arah)
    return np.ascontiguousarray(np.moveaxis(larik, a, 0))


def pori_penghantar(padat_bingkai):
    """Kluster pori bersambungan muka yang menyentuh muka x=0 dan x=N-1 pada bingkai aliran."""
    lab, _ = ndi.label(~padat_bingkai)                      # struktur bawaan: 6 tetangga
    masuk = set(np.unique(lab[0])) - {0}
    keluar = set(np.unique(lab[-1])) - {0}
    tembus = sorted(masuk & keluar)
    return np.isin(lab, tembus) if tembus else np.zeros_like(padat_bingkai)


def cacah_komponen(pori):
    """Cacah komponen pori bersambungan muka."""
    return int(ndi.label(pori)[1])


# --------------------------------------------------------------------------- kisi D3Q19

def kisi_d3q19():
    c = [[0, 0, 0]]
    for a in range(3):
        for s in (1, -1):
            v = [0, 0, 0]; v[a] = s; c.append(v)
    for a in range(3):
        for b in range(a + 1, 3):
            for s1 in (1, -1):
                for s2 in (1, -1):
                    v = [0, 0, 0]; v[a] = s1; v[b] = s2; c.append(v)
    c = np.array(c, dtype=np.int64)
    w = np.array([1 / 3] + [1 / 18] * 6 + [1 / 36] * 12)
    opp = np.array([np.where((c == -ci).all(1))[0][0] for ci in c], dtype=np.int64)
    pi = np.array([i for i in range(19) if opp[i] >= i], dtype=np.int64)   # satu wakil tiap pasangan
    return c, w, opp, pi


# --------------------------------------------------------------------------- inti LBM
# Skema sama dengan pemecah TRT Pertemuan 11: keseimbangan orde dua, gaya Guo yang dipisah
# menjadi bagian genap dan ganjil, bounce-back setengah jalan, periodik pada tepi larik.
# Hanya simpul fluida yang disimpan. Keadaan yang disimpan adalah populasi pasca-tumbukan;
# satu langkah = tarik (streaming) lalu tumbuk, ditulis ke larik kedua.

@nb.njit(parallel=True, cache=True)
def _langkah_cpu(fa, fb, nbr, c, w, opp, pi, F, wp, wm):
    Q, n = fa.shape
    for j in nb.prange(n):
        for i in range(Q):
            s = nbr[i, j]
            fb[i, j] = fa[i, s] if s >= 0 else fa[opp[i], j]
        r = 0.0; m0 = 0.0; m1 = 0.0; m2 = 0.0
        for i in range(Q):
            v = fb[i, j]
            r += v; m0 += c[i, 0] * v; m1 += c[i, 1] * v; m2 += c[i, 2] * v
        u0 = (m0 + 0.5 * F[0]) / r; u1 = (m1 + 0.5 * F[1]) / r; u2 = (m2 + 0.5 * F[2]) / r
        uu = u0 * u0 + u1 * u1 + u2 * u2
        uF = u0 * F[0] + u1 * F[1] + u2 * F[2]
        for p in range(pi.shape[0]):
            i = pi[p]; o = opp[i]
            cu = c[i, 0] * u0 + c[i, 1] * u1 + c[i, 2] * u2
            cF = c[i, 0] * F[0] + c[i, 1] * F[1] + c[i, 2] * F[2]
            ei = w[i] * r * (1 + 3 * cu + 4.5 * cu * cu - 1.5 * uu)
            eo = w[i] * r * (1 - 3 * cu + 4.5 * cu * cu - 1.5 * uu)
            fi = fb[i, j]; fo = fb[o, j]
            fp = 0.5 * (fi + fo); fm = 0.5 * (fi - fo)
            ep = 0.5 * (ei + eo); em = 0.5 * (ei - eo)
            Sg = w[i] * (-3 * uF + 9 * cu * cF)
            Sa = w[i] * 3 * cF
            gp = -wp * (fp - ep) + (1 - 0.5 * wp) * Sg
            gm = -wm * (fm - em) + (1 - 0.5 * wm) * Sa
            if i == o:
                fb[i, j] = fi + gp
            else:
                fb[i, j] = fi + gp + gm
                fb[o, j] = fo + gp - gm


@nb.njit(parallel=True, cache=True)
def _momen_cpu(fa, nbr, c, opp, F, u, rho):
    Q, n = fa.shape
    for j in nb.prange(n):
        r = 0.0; m0 = 0.0; m1 = 0.0; m2 = 0.0
        for i in range(Q):
            s = nbr[i, j]
            v = fa[i, s] if s >= 0 else fa[opp[i], j]
            r += v; m0 += c[i, 0] * v; m1 += c[i, 1] * v; m2 += c[i, 2] * v
        rho[j] = r
        u[0, j] = (m0 + 0.5 * F[0]) / r
        u[1, j] = (m1 + 0.5 * F[1]) / r
        u[2, j] = (m2 + 0.5 * F[2]) / r


def _kernel_gpu():
    """Menyusun kernel CUDA bila GPU tersedia. Dipanggil malas agar CPU tetap jalan tanpa CUDA."""
    from numba import cuda

    @cuda.jit
    def langkah(fa, fb, nbr, c, w, opp, pi, F, wp, wm):
        j = cuda.grid(1)
        Q = fa.shape[0]; n = fa.shape[1]
        if j >= n:
            return
        for i in range(Q):
            s = nbr[i, j]
            if s >= 0:
                fb[i, j] = fa[i, s]
            else:
                fb[i, j] = fa[opp[i], j]
        r = 0.0; m0 = 0.0; m1 = 0.0; m2 = 0.0
        for i in range(Q):
            v = fb[i, j]
            r += v; m0 += c[i, 0] * v; m1 += c[i, 1] * v; m2 += c[i, 2] * v
        u0 = (m0 + 0.5 * F[0]) / r; u1 = (m1 + 0.5 * F[1]) / r; u2 = (m2 + 0.5 * F[2]) / r
        uu = u0 * u0 + u1 * u1 + u2 * u2
        uF = u0 * F[0] + u1 * F[1] + u2 * F[2]
        for p in range(pi.shape[0]):
            i = pi[p]; o = opp[i]
            cu = c[i, 0] * u0 + c[i, 1] * u1 + c[i, 2] * u2
            cF = c[i, 0] * F[0] + c[i, 1] * F[1] + c[i, 2] * F[2]
            ei = w[i] * r * (1 + 3 * cu + 4.5 * cu * cu - 1.5 * uu)
            eo = w[i] * r * (1 - 3 * cu + 4.5 * cu * cu - 1.5 * uu)
            fi = fb[i, j]; fo = fb[o, j]
            fp = 0.5 * (fi + fo); fm = 0.5 * (fi - fo)
            ep = 0.5 * (ei + eo); em = 0.5 * (ei - eo)
            Sg = w[i] * (-3 * uF + 9 * cu * cF)
            Sa = w[i] * 3 * cF
            gp = -wp * (fp - ep) + (1 - 0.5 * wp) * Sg
            gm = -wm * (fm - em) + (1 - 0.5 * wm) * Sa
            if i == o:
                fb[i, j] = fi + gp
            else:
                fb[i, j] = fi + gp + gm
                fb[o, j] = fo + gp - gm

    @cuda.jit
    def momen(fa, nbr, c, opp, F, u, rho):
        j = cuda.grid(1)
        Q = fa.shape[0]; n = fa.shape[1]
        if j >= n:
            return
        r = 0.0; m0 = 0.0; m1 = 0.0; m2 = 0.0
        for i in range(Q):
            s = nbr[i, j]
            if s >= 0:
                v = fa[i, s]
            else:
                v = fa[opp[i], j]
            r += v; m0 += c[i, 0] * v; m1 += c[i, 1] * v; m2 += c[i, 2] * v
        rho[j] = r
        u[0, j] = (m0 + 0.5 * F[0]) / r
        u[1, j] = (m1 + 0.5 * F[1]) / r
        u[2, j] = (m2 + 0.5 * F[2]) / r

    return cuda, langkah, momen


def gpu_tersedia():
    try:
        from numba import cuda
        return cuda.is_available()
    except Exception:
        return False


class PemecahLBM:
    """Pemecah D3Q19 TRT atas larik padat 3D yang periodik pada seluruh tepi larik.

    Gaya badan g dikenakan searah sumbu ke-0. Dinding dan penyangga diatur oleh pemanggil
    melalui isi larik `padat` (lihat domain_aliran).
    """

    def __init__(self, padat, tau=1.0, g=2.5e-6, lam=3 / 16, backend="auto", kolisi="trt"):
        if padat.ndim != 3:
            raise ValueError("pemecah ini hanya untuk 3D")
        self.bentuk = padat.shape
        self.alir = ~padat
        self.n = int(self.alir.sum())
        self.tau = tau; self.g = g
        self.nu = (tau - 0.5) / 3.0
        self.wp = 1.0 / tau
        # BGK adalah kasus khusus TRT dengan kedua laju relaksasi sama
        self.wm = self.wp if kolisi == "bgk" else 1.0 / (0.5 + lam / (tau - 0.5))
        self.kolisi = kolisi
        c, w, opp, pi = kisi_d3q19()
        self.c = c.astype(np.float64); self.w = w; self.opp = opp; self.pi = pi
        self.F = np.array([g, 0.0, 0.0])
        idx = -np.ones(self.bentuk, np.int64)
        idx[self.alir] = np.arange(self.n)
        nbr = np.empty((19, self.n), np.int32)
        for i in range(19):
            # indeks simpul pada x - c_i (sumber tarikan); -1 berarti padatan
            nbr[i] = np.roll(idx, shift=tuple(c[i]), axis=(0, 1, 2))[self.alir]
        del idx
        self.backend = ("gpu" if gpu_tersedia() else "cpu") if backend == "auto" else backend
        f0 = np.repeat(w[:, None], self.n, 1).astype(np.float64)
        if self.backend == "gpu":
            self._cuda, self._k_langkah, self._k_momen = _kernel_gpu()
            cu = self._cuda
            self.d_fa = cu.to_device(f0); self.d_fb = cu.device_array_like(self.d_fa)
            self.d_nbr = cu.to_device(nbr)
            self.d_c = cu.to_device(self.c); self.d_w = cu.to_device(self.w)
            self.d_opp = cu.to_device(self.opp); self.d_pi = cu.to_device(self.pi)
            self.d_F = cu.to_device(self.F)
            self.d_u = cu.device_array((3, self.n)); self.d_rho = cu.device_array(self.n)
            self.blok = 128
            self.grid = (self.n + self.blok - 1) // self.blok
        else:
            self.fa = f0; self.fb = np.empty_like(f0); self.nbr = nbr
            self.u = np.zeros((3, self.n)); self.rho = np.ones(self.n)

    # -- langkah dan momen
    def langkah(self, k=1):
        if self.backend == "gpu":
            for _ in range(k):
                self._k_langkah[self.grid, self.blok](self.d_fa, self.d_fb, self.d_nbr, self.d_c, self.d_w,
                                                      self.d_opp, self.d_pi, self.d_F, self.wp, self.wm)
                self.d_fa, self.d_fb = self.d_fb, self.d_fa
            self._cuda.synchronize()
        else:
            for _ in range(k):
                _langkah_cpu(self.fa, self.fb, self.nbr, self.c, self.w, self.opp, self.pi, self.F, self.wp, self.wm)
                self.fa, self.fb = self.fb, self.fa

    def momen(self):
        """Mengembalikan (u, rho) pada simpul fluida, berbentuk (3, n) dan (n,)."""
        if self.backend == "gpu":
            self._k_momen[self.grid, self.blok](self.d_fa, self.d_nbr, self.d_c, self.d_opp, self.d_F,
                                                self.d_u, self.d_rho)
            return self.d_u.copy_to_host(), self.d_rho.copy_to_host()
        _momen_cpu(self.fa, self.nbr, self.c, self.opp, self.F, self.u, self.rho)
        return self.u.copy(), self.rho.copy()

    def medan(self):
        """Medan kecepatan dan kerapatan pada seluruh larik; padatan bernilai u = 0, rho = 1."""
        u, rho = self.momen()
        U = np.zeros((3,) + self.bentuk); R = np.ones(self.bentuk)
        for a in range(3):
            U[a][self.alir] = u[a]
        R[self.alir] = rho
        return U, R

    # -- titik simpan
    def populasi(self):
        return self.d_fa.copy_to_host() if self.backend == "gpu" else self.fa.copy()

    def pasang_populasi(self, f):
        if f.shape != (19, self.n):
            raise ValueError("titik simpan tidak cocok dengan domain ini")
        if self.backend == "gpu":
            self.d_fa = self._cuda.to_device(np.ascontiguousarray(f))
        else:
            self.fa[...] = f


def _simpan_atomik(jalur, **isi):
    sementara = jalur + ".sementara.npz"
    np.savez(sementara, **isi)
    os.replace(sementara, jalur)


def jalankan_lbm(padat, tau=1.0, g=2.5e-6, maks=60000, tol=1e-6, periksa=100, backend="auto",
                 titik_simpan=None, simpan_tiap_detik=900.0, cetak=print, min_langkah=1000, kolisi="trt"):
    """Menjalankan pemecah sampai perubahan nisbi <u_x> per `periksa` langkah di bawah `tol`.

    Riwayat sisa konvergensi dicatat sebagai (langkah, rerata u_x atas simpul fluida, perubahan nisbi).
    Bila `titik_simpan` diberikan, populasi disimpan berkala ke berkas itu dan penjalanan
    dilanjutkan dari sana bila berkasnya sudah ada (berguna pada Colab yang dapat terputus).
    """
    p = PemecahLBM(padat, tau=tau, g=g, backend=backend, kolisi=kolisi)
    riwayat = []; k = 0; waktu_lalu = 0.0
    if titik_simpan and os.path.exists(titik_simpan):
        d = np.load(titik_simpan)
        p.pasang_populasi(d["f"]); k = int(d["langkah"]); riwayat = [tuple(r) for r in d["riwayat"]]
        waktu_lalu = float(d["detik"])
        cetak(f"  melanjutkan dari titik simpan, langkah {k}")
    cetak(f"  simpul fluida {p.n:,}, backend {p.backend}, kolisi {kolisi}, tau {tau}, g {g:.2e}")
    ux_lama = riwayat[-1][1] if riwayat else 0.0
    t0 = time.time(); t_simpan = t0; konvergen = False; meledak = False
    k_awal = k  # langkah pada awal sesi ini; MLUPS hanya atas langkah sesi ini
    while k < maks:
        p.langkah(periksa); k += periksa
        u, _ = p.momen()
        ux = float(u[0].mean())
        ubah = abs(ux - ux_lama) / max(abs(ux), 1e-300)
        riwayat.append((k, ux, ubah)); ux_lama = ux
        if not np.isfinite(ux):
            meledak = True; cetak("  MELEDAK: nilai bukan bilangan"); break
        if k % (periksa * 20) == 0:
            dt = time.time() - t0
            cetak(f"  langkah {k:6d}  <u_x> {ux:.6e}  sisa {ubah:.2e}  {p.n * (k - k_awal) / max(dt, 1e-9) / 1e6:6.1f} MLUPS")
        if k >= min_langkah and ubah < tol:
            konvergen = True; break
        if titik_simpan and time.time() - t_simpan > simpan_tiap_detik:
            _simpan_atomik(titik_simpan, f=p.populasi(), langkah=k, riwayat=np.array(riwayat),
                           detik=waktu_lalu + time.time() - t0)
            t_simpan = time.time()
            cetak(f"  titik simpan ditulis pada langkah {k}")
    detik = waktu_lalu + time.time() - t0
    langkah_sesi = k - k_awal
    U, R = p.medan()
    if titik_simpan:
        _simpan_atomik(titik_simpan, f=p.populasi(), langkah=k, riwayat=np.array(riwayat), detik=detik)
    return dict(U=U, R=R, nu=p.nu, g=g, tau=tau, langkah=k, konvergen=konvergen, meledak=meledak,
                riwayat=np.array(riwayat), detik=detik, n_fluida=p.n, backend=p.backend,
                MLUPS=p.n * langkah_sesi / max(time.time() - t0, 1e-9) / 1e6)


# --------------------------------------------------------------------------- domain aliran

def domain_aliran(padat_batuan, arah, penyangga=4, dinding=True):
    """Menyusun domain LBM dari citra batuan, meniru susunan penjalanan acuan (264 x 258 x 258).

    1. Larik diputar ke bingkai aliran, lalu hanya pori penghantar yang dipertahankan.
    2. Lapisan fluida setebal `penyangga` ditambahkan pada muka masuk dan muka keluar, sehingga
       pembungkusan periodik searah aliran melewati penyangga, bukan menyambung muka batuan
       yang tidak serasi.
    3. Bila `dinding`, satu lapis padatan menutup keempat sisi melintang (tanpa pembungkusan).
    Mengembalikan (padat_domain, iris_batuan, pori_penghantar_bingkai).
    """
    pb = bingkai_aliran(padat_batuan, arah)
    pori = pori_penghantar(pb)
    padat = ~pori
    padat = np.pad(padat, ((penyangga, penyangga), (0, 0), (0, 0)), constant_values=False)
    d = 1 if dinding else 0
    if dinding:
        padat = np.pad(padat, ((0, 0), (1, 1), (1, 1)), constant_values=True)
    N = pb.shape[0]
    iris = (slice(penyangga, penyangga + N), slice(d, d + pb.shape[1]), slice(d, d + pb.shape[2]))
    return padat, iris, pori


def sifat_aliran(hasil, iris, pori, penyangga):
    """Permeabilitas dan besaran pemeriksa dari medan hasil jalankan_lbm.

    Gradien pendorong efektif = g + cs^2 (rho_masuk - rho_keluar) / (N + 1), dengan rho diambil
    pada lapisan penyangga tepat sebelum dan sesudah batuan. Kecepatan Darcy q = rerata u_x atas
    seluruh volume batuan (padatan bernilai nol), yaitu fluks dibagi luas muka batuan.
    """
    U, R, nu, g = hasil["U"], hasil["R"], hasil["nu"], hasil["g"]
    Ub = U[(slice(None),) + iris]
    N = Ub.shape[1]
    q = float(Ub[0].mean())
    lintang = iris[1:]
    sisi_masuk = R[(penyangga - 1,) + lintang]; sisi_keluar = R[(penyangga + N,) + lintang]
    # lapisan penyangga seluruhnya fluida, sehingga rerata atas seluruh irisan sah
    rho_in = float(sisi_masuk.mean()); rho_out = float(sisi_keluar.mean())
    G = g + CS2 * (rho_in - rho_out) / (N + 1)
    k_kisi = nu * q / G
    fluks_irisan = Ub[0].sum(axis=(1, 2))
    besar = np.sqrt((Ub ** 2).sum(0))
    u_pori = float(besar[pori].mean())
    umaks = float(np.sqrt((U ** 2).sum(0)).max())
    return dict(k_kisi=k_kisi, q_kisi=q, G_kisi=G, g_kisi=g, rho_masuk=rho_in, rho_keluar=rho_out,
                variasi_fluks_irisan=float(fluks_irisan.std() / abs(fluks_irisan.mean())),
                u_pori_rerata_kisi=u_pori, umaks_kisi=umaks, Ma=umaks * np.sqrt(3.0))


def tortuositas_hidrolik(U_batuan, pori):
    """tau_h = sum|u| / sum u_x, kedua jumlah atas himpunan voxel `pori` yang sama (bingkai aliran)."""
    besar = np.sqrt((U_batuan ** 2).sum(0))
    return float(besar[pori].sum() / U_batuan[0][pori].sum())


# --------------------------------------------------------------------------- tortuositas geometrik

@nb.njit(cache=True)
def _bfs_jarak(pori):
    nx, ny, nz = pori.shape
    jarak = np.full((nx, ny, nz), -1, np.int32)
    antrian = np.empty(nx * ny * nz, np.int64)
    kepala = 0; ekor = 0
    for y in range(ny):
        for z in range(nz):
            if pori[0, y, z]:
                jarak[0, y, z] = 0
                antrian[ekor] = y * nz + z; ekor += 1
    while kepala < ekor:
        p = antrian[kepala]; kepala += 1
        x = p // (ny * nz); r = p % (ny * nz); y = r // nz; z = r % nz
        d = jarak[x, y, z] + 1
        for t in range(6):
            xx = x; yy = y; zz = z
            if t == 0: xx += 1
            elif t == 1: xx -= 1
            elif t == 2: yy += 1
            elif t == 3: yy -= 1
            elif t == 4: zz += 1
            else: zz -= 1
            if xx < 0 or xx >= nx or yy < 0 or yy >= ny or zz < 0 or zz >= nz:
                continue
            if pori[xx, yy, zz] and jarak[xx, yy, zz] < 0:
                jarak[xx, yy, zz] = d
                antrian[ekor] = (xx * ny + yy) * nz + zz; ekor += 1
    return jarak


def tortuositas_geometrik(pori):
    """BFS 6 tetangga dari seluruh voxel pori pada muka masuk (bingkai aliran).

    Panjang lintasan dihitung dalam cacah VOXEL yang dilalui, yaitu cacah langkah + 1, sehingga
    tabung lurus sepanjang N voxel memberi tepat N/N = 1. Mengembalikan:
      tau_g      : rerata (panjang terpendek / N) atas seluruh voxel penghantar pada muka keluar
      tau_g_min  : lintasan terpendek mutlak / N
      langkah_min: cacah langkah (sisi graf) pada lintasan terpendek mutlak
    """
    N = pori.shape[0]
    jarak = _bfs_jarak(np.ascontiguousarray(pori))
    d = jarak[-1][pori[-1]]
    d = d[d >= 0]
    if d.size == 0:
        return dict(tau_g=np.nan, tau_g_min=np.nan, langkah_min=-1, n_keluar=0)
    return dict(tau_g=float((d + 1).mean() / N), tau_g_min=float((d.min() + 1) / N),
                langkah_min=int(d.min()), n_keluar=int(d.size), jarak=jarak)


# --------------------------------------------------------------------------- faktor formasi

def faktor_formasi(pori, rtol=1e-9, maxiter=50000):
    """Persamaan Laplace pada ruang pori penghantar (bingkai aliran), volume hingga.

    Konduktansi 1 antara dua voxel pori bersisian, padatan dan sisi melintang bersifat isolator.
    Potensial 1 pada muka masuk dan 0 pada muka keluar, keduanya terletak setengah voxel di luar
    voxel pertama dan terakhir (konduktansi 2), sehingga panjang hantarnya tepat N.
    D_eff/D_0 = fluks * N / A, dengan A luas muka batuan, dan F = D_0/D_eff. Tabung lurus memberi
    tepat D_eff/D_0 = phi, sehingga F*phi = 1.
    """
    N = pori.shape[0]; A = pori.shape[1] * pori.shape[2]
    n = int(pori.sum())
    idx = -np.ones(pori.shape, np.int64); idx[pori] = np.arange(n)
    baris, kolom = [], []
    for a in range(3):
        s1 = [slice(None)] * 3; s2 = [slice(None)] * 3
        s1[a] = slice(0, -1); s2[a] = slice(1, None)
        i = idx[tuple(s1)]; j = idx[tuple(s2)]
        m = (i >= 0) & (j >= 0)
        baris.append(i[m]); kolom.append(j[m])
    i = np.concatenate(baris); j = np.concatenate(kolom)
    W = sparse.coo_matrix((np.ones(i.size), (i, j)), shape=(n, n)).tocsr()
    W = (W + W.T).tocsr()
    derajat = np.asarray(W.sum(1)).ravel()
    masuk = idx[0][idx[0] >= 0]; keluar = idx[-1][idx[-1] >= 0]
    derajat[masuk] += 2.0; derajat[keluar] += 2.0
    L = (sparse.diags(derajat) - W).tocsr()
    b = np.zeros(n); b[masuk] += 2.0
    M = sparse.diags(1.0 / derajat)
    c0 = np.repeat(1.0 - (np.arange(N) + 0.5) / N, pori.shape[1] * pori.shape[2]).reshape(pori.shape)[pori]
    c, info = cg(L, b, x0=c0, rtol=rtol, maxiter=maxiter, M=M)
    fluks_masuk = float((2.0 * (1.0 - c[masuk])).sum())
    fluks_keluar = float((2.0 * c[keluar]).sum())
    Deff = fluks_masuk * N / A
    return dict(Deff_rel=Deff, F=1.0 / Deff, info=int(info),
                selisih_fluks=abs(fluks_masuk - fluks_keluar) / fluks_masuk,
                sisa_akhir=float(np.linalg.norm(L @ c - b) / np.linalg.norm(b)))


# --------------------------------------------------------------------------- sapuan ambang

def citra_abu_semu(padat, sigma=1.0):
    """Aras keabuan semu: indikator pori yang dihaluskan Gauss (1 = pori murni, 0 = padatan murni).

    Citra pada bundel sudah biner, sehingga ambang tidak dapat disapu langsung. Penghalusan ini
    meniru efek volume parsial pemindaian; mengambangnya pada T selain 0,5 menggeser antarmuka.
    """
    return ndi.gaussian_filter((~padat).astype(np.float32), sigma, mode="reflect")


def ambang_untuk_geseran(s, sigma=1.0):
    """Ambang T yang menggeser antarmuka datar sejauh s voxel (s > 0: pori membesar)."""
    from scipy.stats import norm
    return float(norm.cdf(-s / sigma))


def geser_antarmuka(padat, s, sigma=1.0, abu=None):
    """Larik padat baru dengan antarmuka pori-padatan digeser s voxel (penyapuan ambang setara)."""
    abu = citra_abu_semu(padat, sigma) if abu is None else abu
    return ~(abu >= ambang_untuk_geseran(s, sigma))


def jari_jari_kritis(pori_bingkai):
    """Jari-jari bola terbesar (voxel) yang masih dapat melintas dari muka masuk ke muka keluar.

    Pusat bola berjari-jari r dapat ditempatkan pada voxel yang jaraknya ke padatan >= r; bola itu
    melintas bila himpunan pusat tersebut menembus. Diameter 2 r_c adalah taksiran leher pengendali.
    """
    jarak = ndi.distance_transform_edt(pori_bingkai)
    rendah, tinggi = 0.5, float(jarak.max())
    for _ in range(30):
        r = 0.5 * (rendah + tinggi)
        if pori_penghantar(~(jarak >= r)).any():
            rendah = r
        else:
            tinggi = r
    return rendah


# --------------------------------------------------------------------------- bentuk uji

def celah_sejajar(h, panjang=16, lebar=4):
    """Celah berdinding sejajar: h voxel fluida di antara dua lapis padatan (sumbu ke-2)."""
    padat = np.zeros((panjang, lebar, h + 2), bool)
    padat[:, :, 0] = True; padat[:, :, -1] = True
    return padat


def tabung_lurus(n=64, sisi=10, arah=0):
    """Tabung bersisi persegi selurus sumbu `arah`, di tengah kubus n^3 padatan."""
    padat = np.ones((n, n, n), bool)
    a, b = n // 2 - sisi // 2, n // 2 - sisi // 2 + sisi
    s = [slice(a, b)] * 3; s[arah] = slice(None)
    padat[tuple(s)] = False
    return padat


def tulis_json(jalur, isi):
    os.makedirs(os.path.dirname(jalur), exist_ok=True)

    def bersih(o):
        if isinstance(o, dict):
            return {k: bersih(v) for k, v in o.items() if not isinstance(v, np.ndarray) or v.size < 64}
        if isinstance(o, (list, tuple)):
            return [bersih(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o
    with open(jalur, "w", encoding="utf-8") as f:
        json.dump(bersih(isi), f, indent=1, ensure_ascii=False)
