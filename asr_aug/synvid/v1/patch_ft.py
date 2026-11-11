"""Make usr2_ft_v1.py from the v0 copy of usr2_ft.py (snapshot 849ef5f): extra-jsonl rows with "audio_only": true are
trained through the shared encoder's AUDIO input (frontend_a -> encoder -> ctc_a / decoder.out_layer_a). Video rows unchanged."""
import sys
src = open(sys.argv[1]).read()
R = [
 ("def forward_losses(model, v, ids, eos, dev, ls=0.1, a=None, wav=None):\n    from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask\n    m = stream_mask(a, v.shape[0], dev) if a is not None else None\n    if wav is None:",
  "def forward_losses(model, v, ids, eos, dev, ls=0.1, a=None, wav=None):\n    from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask\n    m = stream_mask(a, v.shape[0], dev) if (a is not None and v is not None) else None\n    if v is None:  # V1: audio-only row (TTS of new sentences) through the shared encoder's audio input\n        feat = model.encoder(xs_a=wav.view(1, -1, 1).to(dev))\n        ctc_head, out_layer = model.ctc_a, model.decoder.out_layer_a\n    elif wav is None:"),
 ("        if ids:\n            tr.append((remap(r[\"video_path\"]), ids, remap(r[\"audio_path\"]) if a.modality == \"av\" else None))",
  "        if ids and r.get(\"audio_only\"):\n            tr.append((None, ids, r[\"audio_path\"]))\n        elif ids:\n            tr.append((remap(r[\"video_path\"]), ids, remap(r[\"audio_path\"]) if a.modality == \"av\" else None))"),
 ("            v = load_tr(p)\n            with torch.autocast(\"cuda\", dtype=torch.bfloat16):\n                ctc, ce, _ = forward_losses(model, v, ids, eos, dev, ls=a.label_smoothing, a=a, wav=load_wav(ap, v.shape[0]) if ap else None)",
  "            v = load_tr(p) if p is not None else None\n            w_ = (load_wav(ap, v.shape[0]) if ap else None) if v is not None else load_wav_full(ap)\n            with torch.autocast(\"cuda\", dtype=torch.bfloat16):\n                ctc, ce, _ = forward_losses(model, v, ids, eos, dev, ls=a.label_smoothing, a=a, wav=w_)"),
 ("def forward_losses(",
  "def load_wav_full(path):  # V1: whole 16 kHz clip, padded to a multiple of 640 samples (one 25 fps frame)\n    import torchaudio\n    wav, sr = torchaudio.load(str(path)); wav = wav.mean(0)\n    assert sr == 16000, (path, sr)\n    return F.pad(wav, (0, (-wav.numel()) % 640))\n\n\ndef forward_losses("),
]
for old, new in R:
    assert src.count(old) == 1, old[:80]
    src = src.replace(old, new)
open(sys.argv[2], "w").write(src)
print("patched ok")
