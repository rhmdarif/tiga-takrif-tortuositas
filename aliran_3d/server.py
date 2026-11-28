"""Server lokal kecil untuk membuka index.html lewat http://localhost.

Browser memblokir fetch() dari file:// , jadi halaman harus disajikan lewat HTTP.
Skrip ini menyajikan folder tempat ia berada, memilih port yang kosong,
lalu membuka browser. Tutup jendela ini (atau tekan Ctrl+C) untuk berhenti.
"""
import functools
import http.server
import os
import sys
import threading
import webbrowser

AKAR = os.path.dirname(os.path.abspath(__file__))
PORT_PILIHAN = 8765


class Penangan(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass  # jangan penuhi terminal dengan log setiap permintaan

    def end_headers(self):
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()


def buat_server():
    penangan = functools.partial(Penangan, directory=AKAR)
    for port in (PORT_PILIHAN, 0):  # 0 = biarkan sistem memilih port kosong
        try:
            return http.server.ThreadingHTTPServer(("127.0.0.1", port), penangan)
        except OSError:
            continue
    raise SystemExit("Tidak ada port yang bisa dipakai.")


def main():
    if sys.version_info < (3, 7):
        raise SystemExit("Butuh Python 3.7 atau lebih baru.")
    if not os.path.isfile(os.path.join(AKAR, "index.html")):
        raise SystemExit("index.html tidak ditemukan di " + AKAR +
                         "\nLetakkan peluncur ini di folder yang sama dengan index.html dan folder data/.")
    if not os.path.isdir(os.path.join(AKAR, "data")):
        print("Peringatan: folder data/ tidak ditemukan; data 3D tidak akan termuat.")

    server = buat_server()
    alamat = "http://localhost:%d/" % server.server_address[1]
    print("Aliran 3D Bentheimer berjalan di " + alamat)
    print("Biarkan jendela ini terbuka selama aplikasi dipakai. Tutup jendela atau tekan Ctrl+C untuk berhenti.")
    threading.Timer(0.6, lambda: webbrowser.open(alamat)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
