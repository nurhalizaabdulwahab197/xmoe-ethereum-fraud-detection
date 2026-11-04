# Langkah pertama pada PC WSL — tiada training

Pakej ini ialah calon release local. Ia belum sedia untuk push. Folder audit asal tidak dipindahkan atau dipadam.

## 1. Buka terminal pada folder pakej yang diekstrak

Gunakan Python environment audit yang sudah ada:

```bash
source "$HOME/blte-reproduction/.venv-author/bin/activate"
```

## 2. Semakan pakej dan pengiraan jadual sahaja

```bash
python scripts/check_package.py
python scripts/summarize_results.py --reference
```

## 3. Semak bukti penuh 120 run pada audit asal — baca sahaja

```bash
python scripts/verify_audit.py \
  --audit-root "$HOME/blte-reproduction/imported-audit-20261103-124745-627266/xmoe-blte-audit-20261102-234319"
```

Semakan membaca semua fail output yang hash-nya direkodkan, termasuk salinan graph/checkpoint dalam folder run; ia boleh mengambil masa. Ia mencetak kemajuan setiap run. Tiada optimizer step, training atau fail baharu. Jika gagal, simpan output ralat; jangan ubah bukti untuk memaksakan PASS.

## 4. Selepas semakan lulus: salin bukti ke folder baharu

Langkah ini menulis fail baharu sahaja. Jangan jalankan sebelum keputusan langkah 3 disemak. `results/reference` mesti belum wujud.

```bash
python scripts/verify_audit.py \
  --audit-root "$HOME/blte-reproduction/imported-audit-20261103-124745-627266/xmoe-blte-audit-20261102-234319" \
  --copy-to results/reference --include-checkpoints
```

Semakan penuh diulang sebelum menyalin. Prediction, metrik, sejarah yang tersedia dan checkpoint disalin dengan hash asal. Metadata penerbitan baharu menggunakan rujukan relatif dan menyimpan hash metadata asal. Dataset tidak disalin. Jika proses salinan terganggu, folder tidak dipadam dan tidak ditimpa semasa ulang.

## 5. Semakan GPU tanpa latihan eksperimen

```bash
python scripts/check_environment.py
python scripts/check_models.py
```

Jangan jalankan command `runner.py --run` sekarang. Wrapper latihan baharu belum diuji pada GPU ini. Tiada commit/push automatik dalam pakej.
