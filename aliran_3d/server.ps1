# Server lokal cadangan untuk Windows tanpa Python (memakai .NET HttpListener).
# Menyajikan folder ini lewat http://localhost lalu membuka browser.
# Tutup jendela ini untuk berhenti.

$ErrorActionPreference = 'Stop'
$akar = [IO.Path]::GetFullPath((Split-Path -Parent $MyInvocation.MyCommand.Path))
if (-not $akar.EndsWith('\')) { $akar = $akar + '\' }

if (-not (Test-Path (Join-Path $akar 'index.html'))) {
    Write-Host "index.html tidak ditemukan di $akar"
    Write-Host "Letakkan peluncur ini di folder yang sama dengan index.html dan folder data."
    Read-Host "Tekan Enter untuk keluar"
    exit 1
}

$pendengar = $null
foreach ($p in 8765..8800) {
    $l = New-Object System.Net.HttpListener
    $l.Prefixes.Add("http://localhost:$p/")
    try { $l.Start(); $pendengar = $l; $port = $p; break } catch { $l.Close() }
}
if (-not $pendengar) {
    Write-Host "Tidak ada port kosong antara 8765 dan 8800."
    Read-Host "Tekan Enter untuk keluar"
    exit 1
}

$mime = @{
    '.html' = 'text/html; charset=utf-8'; '.htm' = 'text/html; charset=utf-8'
    '.js'   = 'text/javascript; charset=utf-8'; '.css' = 'text/css; charset=utf-8'
    '.json' = 'application/json; charset=utf-8'; '.txt' = 'text/plain; charset=utf-8'
    '.svg'  = 'image/svg+xml'; '.png' = 'image/png'; '.jpg' = 'image/jpeg'; '.ico' = 'image/x-icon'
}

$alamat = "http://localhost:$port/"
Write-Host "Aliran 3D Bentheimer berjalan di $alamat"
Write-Host "Biarkan jendela ini terbuka selama aplikasi dipakai. Tutup jendela untuk berhenti."
Start-Process $alamat

try {
    while ($pendengar.IsListening) {
        $ctx = $pendengar.GetContext()
        $res = $ctx.Response
        try {
            $rel = [Uri]::UnescapeDataString($ctx.Request.Url.AbsolutePath).TrimStart('/')
            if ($rel -eq '') { $rel = 'index.html' }
            $penuh = [IO.Path]::GetFullPath((Join-Path $akar ($rel -replace '/', '\')))
            if ($penuh.StartsWith($akar, [StringComparison]::OrdinalIgnoreCase) -and (Test-Path -LiteralPath $penuh -PathType Leaf)) {
                $bait = [IO.File]::ReadAllBytes($penuh)
                $ext = [IO.Path]::GetExtension($penuh).ToLower()
                if ($mime.ContainsKey($ext)) { $res.ContentType = $mime[$ext] } else { $res.ContentType = 'application/octet-stream' }
                $res.AddHeader('Cache-Control', 'no-cache')
                $res.ContentLength64 = $bait.Length
                $res.OutputStream.Write($bait, 0, $bait.Length)
            } else {
                $res.StatusCode = 404
            }
        } catch {
            try { $res.StatusCode = 500 } catch {}
        } finally {
            $res.Close()
        }
    }
} finally {
    $pendengar.Stop()
}
