# Quilt-LLaVA Patch Descriptions on wopr

This repo is set up at:

```bash
/home/dm236/Development/quilt-llava
```

Run a patch description with:

```bash
/home/dm236/Development/quilt-llava/scripts/describe_patch /path/to/patch.png
```

You can also pass a URL:

```bash
/home/dm236/Development/quilt-llava/scripts/describe_patch https://example.org/patch.jpg
```

The default prompt asks for neutral histopathology morphology and avoids making a definitive diagnosis unless the image is sufficient. To override it:

```bash
/home/dm236/Development/quilt-llava/scripts/describe_patch /path/to/patch.png \
  --prompt "Describe the tissue architecture and inflammatory infiltrate."
```

Useful flags:

```bash
--max-new-tokens 256
--temperature 0
--verbose
```

Notes:

- The environment is in `.venv` and uses Python 3.10.19.
- The model is `wisdomik/Quilt-Llava-v1.5-7b` from Hugging Face.
- PyTorch sees the RTX 3090 via CUDA, although `nvidia-smi` currently reports an NVML driver/library mismatch.
- Treat outputs as research/draft descriptions, not clinical diagnoses.
