#!/usr/bin/env bash
# Peluncur Aliran 3D Bentheimer untuk Linux.
#   ./jalankan-linux.sh            menjalankan aplikasi
#   ./jalankan-linux.sh --pasang   membuat ikon di menu aplikasi dan Desktop
set -e
DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
cd "$DIR"

if [ "$1" = "--pasang" ]; then
    ISI="[Desktop Entry]
Type=Application
Name=Aliran 3D Bentheimer
Comment=Visualisasi aliran 3D Bentheimer
Exec=\"$DIR/jalankan-linux.sh\"
Path=$DIR
Icon=applications-science
Terminal=true
Categories=Science;Education;"
    mkdir -p "$HOME/.local/share/applications"
    printf '%s\n' "$ISI" > "$HOME/.local/share/applications/aliran-3d-bentheimer.desktop"
    chmod +x "$HOME/.local/share/applications/aliran-3d-bentheimer.desktop"
    DESK="$(xdg-user-dir DESKTOP 2>/dev/null || echo "$HOME/Desktop")"
    if [ -d "$DESK" ]; then
        printf '%s\n' "$ISI" > "$DESK/aliran-3d-bentheimer.desktop"
        chmod +x "$DESK/aliran-3d-bentheimer.desktop"
        gio set "$DESK/aliran-3d-bentheimer.desktop" metadata::trusted true 2>/dev/null || true
    fi
    echo "Ikon dibuat di menu aplikasi${DESK:+ dan $DESK}."
    exit 0
fi

PY="$(command -v python3 || command -v python || true)"
if [ -z "$PY" ]; then
    echo "Python 3 tidak ditemukan. Pasang dulu, misalnya: sudo apt install python3"
    read -r -p "Tekan Enter untuk keluar " _
    exit 1
fi
exec "$PY" "$DIR/server.py"
