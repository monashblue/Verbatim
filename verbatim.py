"""
Transkrip Audio Verbatim

Alat transkripsi audio berbasis faster-whisper (pengenalan ucapan) dan
sherpa-onnx (pembeda pembicara). Dapat dijalankan sebagai aplikasi jendela
(tkinter) atau lewat baris perintah, dan dapat dikompilasi ke .exe dengan
PyInstaller.

Keluaran:
  1. NAMA_transkrip.txt : transkrip verbatim dengan penanda waktu,
     label Pembicara A, B, dan seterusnya, serta penanda [tidak terdengar].
  2. NAMA_tidak_terdengar.txt : daftar bagian yang ditandai [tidak terdengar]
     beserta tebakan mesin, supaya bisa dicek manual terhadap rekaman.

Catatan: mesin pengenal ucapan tidak menjamin hasil verbatim sempurna.
Transkrip tetap perlu diperiksa sambil mendengarkan rekaman.
"""

import argparse
import difflib
import math
import multiprocessing
import queue
import string
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path

TIDAK_TERDENGAR = "[tidak terdengar]"
BAHASA_TANPA_SPASI = {"zh", "ja", "th", "lo", "my", "km"}
SAMPLE_RATE = 16000

PROMPT_VERBATIM = {
    "id": (
        "Transkrip verbatim lengkap dengan kata pengisi dan pengulangan. "
        "Eh, em, hmm, jadi, jadi, jadi gini, gini, ya, ya, kan, kan, "
        "itu, itu maksudnya, anu."
    ),
    "en": (
        "Verbatim transcript including filler words and repetitions. "
        "Uh, um, hmm, so, so, I mean, like, you know, you know."
    ),
}
PREFIKS_ISTILAH = {"id": "Istilah dan nama:", "en": "Terms and names:"}

DAFTAR_BAHASA = [
    "id", "en", "auto", "ms", "jw", "su", "ar", "zh", "ja", "ko",
    "nl", "de", "fr", "es",
]
DAFTAR_MODEL = ["tiny", "base", "small", "medium", "large-v3"]


def folder_aplikasi():
    """Folder tempat skrip atau berkas .exe berada."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


MODEL_SEGMENTASI_DEFAULT = folder_aplikasi() / "models" / "diarisasi" / "segmentasi.onnx"
MODEL_EMBEDDING_DEFAULT = folder_aplikasi() / "models" / "diarisasi" / "embedding.onnx"


@dataclass
class Kata:
    mulai: float
    selesai: float
    teks: str
    pembicara: int = -1
    tidak_jelas: bool = False
    tebakan: str = ""
    keyakinan: float = 0.0


@dataclass
class Pengaturan:
    audio: str
    folder_keluaran: str = ""
    bahasa: str = "id"
    ukuran_model: str = "small"
    pisahkan_pembicara: bool = True
    jumlah_pembicara: int = 0
    model_segmentasi: str = str(MODEL_SEGMENTASI_DEFAULT)
    model_embedding: str = str(MODEL_EMBEDDING_DEFAULT)
    istilah: list = field(default_factory=list)
    ambang_kata: float = 0.30
    interval_penanda: int = 30
    koreksi_istilah: bool = True
    tandai_bagian_kosong: bool = True
    durasi_bagian_kosong: float = 1.5
    jumlah_thread: int = 4


def format_waktu(detik):
    total = int(max(detik, 0))
    jam, sisa = divmod(total, 3600)
    menit, dtk = divmod(sisa, 60)
    return f"{jam:02d}:{menit:02d}:{dtk:02d}"


def tentukan_model(ukuran):
    """Pakai model lokal jika ada di models/whisper/UKURAN, selain itu unduh."""
    lokal = folder_aplikasi() / "models" / "whisper" / ukuran
    if (lokal / "model.bin").exists():
        return str(lokal)
    return ukuran


def pisah_tanda(teks):
    """Pisahkan tanda baca di awal dan akhir kata."""
    awal = 0
    akhir = len(teks)
    while awal < akhir and not teks[awal].isalnum():
        awal += 1
    while akhir > awal and not teks[akhir - 1].isalnum():
        akhir -= 1
    return teks[:awal], teks[awal:akhir], teks[akhir:]


def koreksi_istilah(teks, daftar):
    """Samakan ejaan kata dengan daftar istilah bila sangat mirip."""
    depan, inti, belakang = pisah_tanda(teks)
    if len(inti) < 4 or not daftar:
        return teks
    kecil = inti.lower()
    peta = {x.lower(): x for x in daftar}
    if kecil in peta:
        return depan + peta[kecil] + belakang
    cocok = difflib.get_close_matches(kecil, list(peta), n=1, cutoff=0.85)
    if cocok:
        return depan + peta[cocok[0]] + belakang
    return teks


def transkripsi(audio, p, log):
    from faster_whisper import WhisperModel

    log(f"Memuat model Whisper ({p.ukuran_model}). Unduhan pertama memerlukan internet.")
    model = WhisperModel(
        tentukan_model(p.ukuran_model),
        device="cpu",
        compute_type="int8",
        cpu_threads=p.jumlah_thread,
        download_root=str(folder_aplikasi() / "models" / "whisper_cache"),
    )

    bagian_prompt = [PROMPT_VERBATIM.get(p.bahasa, "")]
    if p.istilah:
        prefiks = PREFIKS_ISTILAH.get(p.bahasa, "Terms:")
        bagian_prompt.append(prefiks + " " + ", ".join(p.istilah[:40]) + ".")
    prompt = " ".join(x for x in bagian_prompt if x)

    segmen, info = model.transcribe(
        audio,
        language=None if p.bahasa == "auto" else p.bahasa,
        task="transcribe",
        beam_size=5,
        word_timestamps=True,
        initial_prompt=prompt or None,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        condition_on_previous_text=True,
    )
    log(f"Bahasa yang dipakai: {info.language}. Transkripsi berjalan.")

    total = max(info.duration, 1.0)
    terakhir = -5
    hasil = []
    for s in segmen:
        persen = int(min(s.end / total, 1.0) / 0.01)
        if persen >= terakhir + 5:
            terakhir = persen
            log(f"Transkripsi {persen}%")

        segmen_ragu = s.no_speech_prob > 0.6 and s.avg_logprob < -1.0
        if segmen_ragu:
            hasil.append(Kata(
                s.start, s.end, TIDAK_TERDENGAR, tidak_jelas=True,
                tebakan=s.text.strip(), keyakinan=math.exp(s.avg_logprob),
            ))
            continue

        for w in s.words or []:
            teks = w.word.strip()
            if not teks:
                continue
            if w.probability < p.ambang_kata:
                hasil.append(Kata(
                    w.start, w.end, TIDAK_TERDENGAR, tidak_jelas=True,
                    tebakan=teks, keyakinan=w.probability,
                ))
            else:
                hasil.append(Kata(w.start, w.end, teks, keyakinan=w.probability))
    return hasil, info.language


def pisah_pembicara(audio, p, log):
    """Kembalikan daftar (mulai, selesai, id_pembicara) terurut menurut waktu mulai."""
    import sherpa_onnx

    for jalur, nama in (
        (p.model_segmentasi, "model segmentasi"),
        (p.model_embedding, "model embedding"),
    ):
        if not Path(jalur).is_file():
            raise FileNotFoundError(
                f"File {nama} tidak ditemukan: {jalur}. "
                "Letakkan modelnya di folder models/diarisasi atau matikan pilihan pisahkan pembicara."
            )

    konfigurasi = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=p.model_segmentasi
            ),
            num_threads=p.jumlah_thread,
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=p.model_embedding,
            num_threads=p.jumlah_thread,
        ),
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=p.jumlah_pembicara if p.jumlah_pembicara > 0 else -1,
            threshold=0.5,
        ),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )
    if not konfigurasi.validate():
        raise RuntimeError("Konfigurasi pembeda pembicara tidak valid. Periksa file model.")

    pembeda = sherpa_onnx.OfflineSpeakerDiarization(konfigurasi)
    if pembeda.sample_rate != SAMPLE_RATE:
        raise RuntimeError(
            f"Model pembeda pembicara memakai sample rate {pembeda.sample_rate}, bukan {SAMPLE_RATE}."
        )

    log("Membedakan pembicara.")
    terakhir = {"nilai": -10}

    def lapor(selesai, total):
        persen = int(selesai / max(total, 1) / 0.01)
        if persen >= terakhir["nilai"] + 10:
            terakhir["nilai"] = persen
            log(f"Pembeda pembicara {persen}%")
        return 0

    hasil = pembeda.process(audio, callback=lapor).sort_by_start_time()
    return [(r.start, r.end, r.speaker) for r in hasil]


def tetapkan_pembicara(kata_list, segmen_bicara):
    """Beri tiap kata pembicara dengan irisan waktu terbesar."""
    sebelumnya = segmen_bicara[0][2] if segmen_bicara else 0
    for k in kata_list:
        terbaik = None
        tumpang = 0.0
        for mulai, selesai, id_ in segmen_bicara:
            if mulai > k.selesai:
                break
            irisan = min(selesai, k.selesai) - max(mulai, k.mulai)
            if irisan > tumpang:
                tumpang = irisan
                terbaik = id_
        if terbaik is None:
            jarak_min = 1.5
            for mulai, selesai, id_ in segmen_bicara:
                jarak = max(mulai - k.selesai, k.mulai - selesai, 0.0)
                if jarak < jarak_min:
                    jarak_min = jarak
                    terbaik = id_
        if terbaik is None:
            terbaik = sebelumnya
        k.pembicara = terbaik
        sebelumnya = terbaik


def sisipkan_bagian_kosong(kata_list, segmen_bicara, durasi_min):
    """Tandai bagian yang dikenali sebagai ucapan oleh pembeda pembicara
    tetapi tidak menghasilkan kata dari pengenal ucapan."""
    tambahan = []
    for mulai, selesai, id_ in segmen_bicara:
        panjang = selesai - mulai
        if panjang < durasi_min:
            continue
        tertutup = 0.0
        for k in kata_list:
            if k.selesai < mulai:
                continue
            if k.mulai > selesai:
                break
            tertutup += max(0.0, min(selesai, k.selesai) - max(mulai, k.mulai))
        if tertutup / panjang < 0.15:
            tambahan.append(Kata(
                mulai, selesai, TIDAK_TERDENGAR, pembicara=id_, tidak_jelas=True,
                tebakan="", keyakinan=0.0,
            ))
    kata_list.extend(tambahan)
    kata_list.sort(key=lambda k: k.mulai)


def peta_label(kata_list):
    urutan = []
    for k in kata_list:
        if k.pembicara not in urutan:
            urutan.append(k.pembicara)
    huruf = string.ascii_uppercase
    label = {}
    for i, id_ in enumerate(urutan):
        label[id_] = f"Pembicara {huruf[i]}" if i < len(huruf) else f"Pembicara {i + 1}"
    return label


def susun_transkrip(kata_list, label, interval, bahasa):
    gabung = "" if bahasa in BAHASA_TANPA_SPASI else " "
    giliran = []
    sebelumnya_tak_jelas = False
    for k in kata_list:
        if not giliran or giliran[-1]["pembicara"] != k.pembicara:
            giliran.append({
                "pembicara": k.pembicara,
                "mulai": k.mulai,
                "penanda": k.mulai + interval,
                "bagian": [],
            })
            sebelumnya_tak_jelas = False
        g = giliran[-1]
        if k.tidak_jelas:
            if sebelumnya_tak_jelas:
                continue
            sebelumnya_tak_jelas = True
        else:
            sebelumnya_tak_jelas = False
        if interval > 0 and k.mulai >= g["penanda"]:
            g["bagian"].append(f"[{format_waktu(k.mulai)}]")
            while g["penanda"] <= k.mulai:
                g["penanda"] += interval
        g["bagian"].append(k.teks)

    baris = []
    for g in giliran:
        kepala = f"[{format_waktu(g['mulai'])}] {label[g['pembicara']]}: "
        baris.append(kepala + gabung.join(g["bagian"]))
    return "\n\n".join(baris)


def susun_catatan_ragu(kata_list):
    baris = []
    for k in kata_list:
        if not k.tidak_jelas:
            continue
        if k.tebakan:
            ket = f'tebakan mesin: "{k.tebakan}" (keyakinan {k.keyakinan:.2f})'
        else:
            ket = "tanpa tebakan (terdeteksi sebagai ucapan oleh pembeda pembicara)"
        baris.append(f"[{format_waktu(k.mulai)}] {ket}")
    if not baris:
        return "Tidak ada bagian yang ditandai [tidak terdengar]."
    return "\n".join(baris)


def jalankan(p, log):
    """Jalankan seluruh proses. Kembalikan jalur berkas transkrip."""
    jalur_audio = Path(p.audio)
    if not jalur_audio.is_file():
        raise FileNotFoundError(f"Berkas audio tidak ditemukan: {p.audio}")

    from faster_whisper import decode_audio

    log("Membaca audio.")
    audio = decode_audio(str(jalur_audio), sampling_rate=SAMPLE_RATE)
    durasi = len(audio) / SAMPLE_RATE

    kata_list, bahasa_dipakai = transkripsi(audio, p, log)
    if not kata_list:
        raise RuntimeError("Tidak ada ucapan yang terdeteksi pada audio.")

    if p.istilah and p.koreksi_istilah:
        tunggal = [x for x in p.istilah if " " not in x]
        for k in kata_list:
            if not k.tidak_jelas:
                k.teks = koreksi_istilah(k.teks, tunggal)

    if p.pisahkan_pembicara:
        segmen_bicara = pisah_pembicara(audio, p, log)
        if not segmen_bicara:
            raise RuntimeError("Pembeda pembicara tidak menemukan segmen ucapan.")
        tetapkan_pembicara(kata_list, segmen_bicara)
        if p.tandai_bagian_kosong:
            sisipkan_bagian_kosong(kata_list, segmen_bicara, p.durasi_bagian_kosong)
        label = peta_label(kata_list)
        status_pembicara = f"aktif ({len(label)} pembicara terdeteksi)"
    else:
        for k in kata_list:
            k.pembicara = 0
        label = {0: "Pembicara (tidak dibedakan)"}
        status_pembicara = "tidak aktif"

    isi = susun_transkrip(kata_list, label, p.interval_penanda, bahasa_dipakai)
    garis = "-".ljust(70, "-")
    kepala = "\n".join([
        "TRANSKRIP VERBATIM",
        f"Berkas audio : {jalur_audio.name}",
        f"Durasi       : {format_waktu(durasi)}",
        f"Bahasa       : {bahasa_dipakai}",
        f"Model        : Whisper {p.ukuran_model}",
        f"Pembicara    : {status_pembicara}",
        "Penanda      : [jam:menit:detik] di awal giliran bicara dan tiap "
        f"{p.interval_penanda} detik pada giliran panjang",
        "Bagian yang tidak jelas ditulis [tidak terdengar].",
        garis,
        "",
    ])

    folder = Path(p.folder_keluaran) if p.folder_keluaran else jalur_audio.parent
    folder.mkdir(parents=True, exist_ok=True)
    berkas_transkrip = folder / f"{jalur_audio.stem}_transkrip.txt"
    berkas_ragu = folder / f"{jalur_audio.stem}_tidak_terdengar.txt"
    berkas_transkrip.write_text(kepala + "\n" + isi + "\n", encoding="utf-8")
    berkas_ragu.write_text(susun_catatan_ragu(kata_list) + "\n", encoding="utf-8")
    log(f"Transkrip disimpan: {berkas_transkrip}")
    log(f"Catatan bagian ragu: {berkas_ragu}")
    return str(berkas_transkrip)


def baca_istilah(nilai):
    if not nilai:
        return []
    jalur = Path(nilai)
    if jalur.is_file():
        isi = jalur.read_text(encoding="utf-8").splitlines()
    else:
        isi = nilai.split(",")
    return [x.strip() for x in isi if x.strip()]


def buka_gui():
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    akar = tk.Tk()
    akar.title("Transkrip Audio Verbatim")
    akar.geometry("800x720")
    antrean = queue.Queue()

    audio_var = tk.StringVar()
    keluar_var = tk.StringVar()
    bahasa_var = tk.StringVar(value="id")
    model_var = tk.StringVar(value="small")
    pisah_var = tk.BooleanVar(value=True)
    jumlah_var = tk.IntVar(value=0)
    seg_var = tk.StringVar(value=str(MODEL_SEGMENTASI_DEFAULT))
    emb_var = tk.StringVar(value=str(MODEL_EMBEDDING_DEFAULT))

    bingkai = ttk.Frame(akar, padding=12)
    bingkai.pack(fill="both", expand=True)
    bingkai.columnconfigure(1, weight=1)

    def baris_berkas(nomor, teks, variabel, aksi):
        ttk.Label(bingkai, text=teks).grid(row=nomor, column=0, sticky="w", pady=3)
        ttk.Entry(bingkai, textvariable=variabel).grid(row=nomor, column=1, sticky="ew", padx=6)
        ttk.Button(bingkai, text="Pilih", command=aksi).grid(row=nomor, column=2)

    def pilih_audio():
        jalur = filedialog.askopenfilename(title="Pilih berkas audio")
        if jalur:
            audio_var.set(jalur)

    def pilih_keluar():
        jalur = filedialog.askdirectory(title="Pilih folder keluaran")
        if jalur:
            keluar_var.set(jalur)

    def pilih_seg():
        jalur = filedialog.askopenfilename(title="Pilih model segmentasi (.onnx)")
        if jalur:
            seg_var.set(jalur)

    def pilih_emb():
        jalur = filedialog.askopenfilename(title="Pilih model embedding (.onnx)")
        if jalur:
            emb_var.set(jalur)

    baris_berkas(0, "Berkas audio", audio_var, pilih_audio)
    baris_berkas(1, "Folder keluaran", keluar_var, pilih_keluar)

    ttk.Label(bingkai, text="Bahasa").grid(row=2, column=0, sticky="w", pady=3)
    ttk.Combobox(bingkai, textvariable=bahasa_var, values=DAFTAR_BAHASA, width=10).grid(
        row=2, column=1, sticky="w", padx=6
    )
    ttk.Label(bingkai, text="Ukuran model").grid(row=3, column=0, sticky="w", pady=3)
    ttk.Combobox(bingkai, textvariable=model_var, values=DAFTAR_MODEL, width=10).grid(
        row=3, column=1, sticky="w", padx=6
    )

    ttk.Checkbutton(bingkai, text="Bedakan pembicara (A, B, dan seterusnya)", variable=pisah_var).grid(
        row=4, column=0, columnspan=2, sticky="w", pady=3
    )
    ttk.Label(bingkai, text="Jumlah pembicara (0 = otomatis)").grid(row=5, column=0, sticky="w", pady=3)
    ttk.Spinbox(bingkai, from_=0, to=20, textvariable=jumlah_var, width=6).grid(
        row=5, column=1, sticky="w", padx=6
    )
    baris_berkas(6, "Model segmentasi", seg_var, pilih_seg)
    baris_berkas(7, "Model embedding", emb_var, pilih_emb)

    ttk.Label(bingkai, text="Istilah dan nama (satu per baris)").grid(
        row=8, column=0, columnspan=3, sticky="w", pady=(8, 2)
    )
    kotak_istilah = tk.Text(bingkai, height=5)
    kotak_istilah.grid(row=9, column=0, columnspan=3, sticky="ew")

    tombol = ttk.Button(bingkai, text="Mulai transkripsi")
    tombol.grid(row=10, column=0, columnspan=3, pady=10)
    bar = ttk.Progressbar(bingkai, mode="indeterminate")
    bar.grid(row=11, column=0, columnspan=3, sticky="ew")

    kotak_log = tk.Text(bingkai, height=12, state="disabled")
    kotak_log.grid(row=12, column=0, columnspan=3, sticky="nsew", pady=(8, 0))
    bingkai.rowconfigure(12, weight=1)

    def tambah_log(pesan):
        kotak_log.config(state="normal")
        kotak_log.insert("end", pesan + "\n")
        kotak_log.see("end")
        kotak_log.config(state="disabled")

    def kerja(pengaturan):
        try:
            hasil = jalankan(pengaturan, lambda s: antrean.put(("log", s)))
            antrean.put(("selesai", hasil))
        except Exception as galat:
            antrean.put(("galat", str(galat)))

    def mulai():
        if not audio_var.get().strip():
            messagebox.showwarning("Transkrip Audio", "Pilih berkas audio terlebih dahulu.")
            return
        istilah = [x.strip() for x in kotak_istilah.get("1.0", "end").splitlines() if x.strip()]
        try:
            jumlah = int(jumlah_var.get())
        except (tk.TclError, ValueError):
            jumlah = 0
        pengaturan = Pengaturan(
            audio=audio_var.get().strip(),
            folder_keluaran=keluar_var.get().strip(),
            bahasa=bahasa_var.get().strip() or "id",
            ukuran_model=model_var.get().strip() or "small",
            pisahkan_pembicara=bool(pisah_var.get()),
            jumlah_pembicara=jumlah,
            model_segmentasi=seg_var.get().strip(),
            model_embedding=emb_var.get().strip(),
            istilah=istilah,
        )
        tombol.config(state="disabled")
        bar.start(12)
        tambah_log("Memulai proses.")
        threading.Thread(target=kerja, args=(pengaturan,), daemon=True).start()

    tombol.config(command=mulai)

    def periksa_antrean():
        try:
            while True:
                jenis, isi = antrean.get_nowait()
                if jenis == "log":
                    tambah_log(isi)
                elif jenis == "selesai":
                    bar.stop()
                    tombol.config(state="normal")
                    messagebox.showinfo("Transkrip Audio", f"Selesai.\n{isi}")
                elif jenis == "galat":
                    bar.stop()
                    tombol.config(state="normal")
                    tambah_log(f"Galat: {isi}")
                    messagebox.showerror("Transkrip Audio", isi)
        except queue.Empty:
            pass
        akar.after(150, periksa_antrean)

    akar.after(150, periksa_antrean)
    akar.mainloop()


def cetak(pesan):
    if sys.stdout is not None:
        print(pesan, flush=True)


def main():
    pengurai = argparse.ArgumentParser(
        description="Transkrip audio verbatim dengan pembeda pembicara. "
        "Tanpa argumen, aplikasi jendela akan dibuka."
    )
    pengurai.add_argument("audio", nargs="?", help="Jalur berkas audio")
    pengurai.add_argument("--bahasa", default="id", help="Kode bahasa, misalnya id, en, atau auto")
    pengurai.add_argument("--model", default="small", choices=DAFTAR_MODEL)
    pengurai.add_argument("--keluar", default="", help="Folder keluaran")
    pengurai.add_argument("--pembicara", type=int, default=0, help="Jumlah pembicara, 0 = otomatis")
    pengurai.add_argument("--tanpa-pembicara", action="store_true", help="Matikan pembeda pembicara")
    pengurai.add_argument("--istilah", default="", help="Berkas teks (satu istilah per baris) atau daftar dipisah koma")
    pengurai.add_argument("--model-segmentasi", default=str(MODEL_SEGMENTASI_DEFAULT))
    pengurai.add_argument("--model-embedding", default=str(MODEL_EMBEDDING_DEFAULT))
    pengurai.add_argument("--ambang", type=float, default=0.30, help="Keyakinan minimum kata, di bawahnya ditandai [tidak terdengar]")
    pengurai.add_argument("--interval", type=int, default=30, help="Detik antar penanda waktu pada giliran panjang, 0 = mati")
    pengurai.add_argument("--tanpa-koreksi-istilah", action="store_true")
    pengurai.add_argument("--thread", type=int, default=4)
    args = pengurai.parse_args()

    if not args.audio:
        buka_gui()
        return

    pengaturan = Pengaturan(
        audio=args.audio,
        folder_keluaran=args.keluar,
        bahasa=args.bahasa,
        ukuran_model=args.model,
        pisahkan_pembicara=not args.tanpa_pembicara,
        jumlah_pembicara=args.pembicara,
        model_segmentasi=args.model_segmentasi,
        model_embedding=args.model_embedding,
        istilah=baca_istilah(args.istilah),
        ambang_kata=args.ambang,
        interval_penanda=args.interval,
        koreksi_istilah=not args.tanpa_koreksi_istilah,
        jumlah_thread=args.thread,
    )
    try:
        jalankan(pengaturan, cetak)
    except Exception as galat:
        cetak(f"Galat: {galat}")
        sys.exit(1)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
