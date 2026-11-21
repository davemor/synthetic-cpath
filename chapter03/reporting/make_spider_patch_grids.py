#!/usr/bin/env python3
"""Select SPIDER patches from split Hugging Face tar files and make a two-column grid."""

from __future__ import annotations

import argparse
import glob
import io
import json
import re
import tarfile
from pathlib import Path

import requests
from huggingface_hub import HfApi, get_token, hf_hub_url
from PIL import Image, ImageDraw, ImageFont, ImageOps


DATASETS = {
    "SPIDER-colorectal": [
        "Adenocarcinoma high grade", "Adenocarcinoma low grade", "Adenoma high grade",
        "Adenoma low grade", "Fat", "Hyperplastic polyp", "Inflammation", "Mucus",
        "Muscle", "Necrosis", "Sessile serrated lesion", "Stroma healthy", "Vessels",
    ],
    "SPIDER-breast": [
        "Adenosis", "Benign phyllodes tumor", "Ductal carcinoma in situ (high-grade)",
        "Ductal carcinoma in situ (low-grade)", "Fat", "Fibroadenoma", "Fibrocystic changes",
        "Fibrosis", "Invasive non-special type carcinoma", "Lipogranuloma",
        "Lobular invasive carcinoma", "Malignant phyllodes tumor", "Necrosis", "Normal ducts",
        "Normal lobules", "Sclerosing adenosis", "Typical ductal hyperplasia", "Vessels",
    ],
    "SPIDER-skin": [
        "Actinic keratosis", "Apocrine glands", "Basal cell carcinoma", "Carcinoma in situ",
        "Collagen", "Epidermis", "Fat", "Follicle", "Inflammation", "Invasive melanoma",
        "Kaposi’s sarcoma", "Keratin", "Melanoma in situ", "Mercel cell carcinoma", "Muscle",
        "Necrosis", "Nerves", "Nevus", "Sebaceous gland", "Seborrheic keratosis",
        "Solar elastosis", "Squamous cell carcinoma", "Vessels", "Wart",
    ],
    "SPIDER-thorax": [
        "Alveoli", "Bronchial cartilage", "Bronchial glands",
        "Chronic inflammation + fibrosis", "Detritus", "Fibrosis", "Hemorrhage",
        "Lymph node", "Pigment", "Pleura", "Tumor non-small cell", "Tumor small cell",
        "Tumor soft", "Vessel",
    ],
}

LABEL_COLORS = {
    "SPIDER-breast": "#FCE8EC",
    "SPIDER-colorectal": "#E8F5E9",
    "SPIDER-skin": "#FFF8D6",
    "SPIDER-thorax": "#E7F1FF",
}


def get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    paths = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in paths:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    raise RuntimeError("Could not locate Arial or DejaVu Sans")


def normalise_label(label: str) -> str:
    return label.replace("'", "’")


def load_manifest_index(manifests_root: Path, dataset: str) -> dict[str, str]:
    cache_name = f"cache-{dataset.lower()}-uni"
    paths = glob.glob(str(manifests_root / cache_name / "*" / "paths_*.json"))
    index: dict[str, str] = {}
    for path in paths:
        with open(path) as stream:
            for row in json.load(stream):
                index[row["patch_filename"]] = normalise_label(row["class"])
    if not index:
        raise RuntimeError(f"No manifests found for {dataset} under {manifests_root}")
    return index


class SplitTarReader(io.RawIOBase):
    """Sequentially expose split remote tar parts as one readable stream."""

    def __init__(self, repo_id: str, filenames: list[str]) -> None:
        self.repo_id = repo_id
        self.filenames = filenames
        self.part = -1
        self.response: requests.Response | None = None
        self.bytes_read = 0
        self._open_next()

    def readable(self) -> bool:
        return True

    def _open_next(self) -> bool:
        if self.response is not None:
            self.response.close()
        self.part += 1
        if self.part >= len(self.filenames):
            self.response = None
            return False
        filename = self.filenames[self.part]
        print(f"  streaming {filename}", flush=True)
        url = hf_hub_url(self.repo_id, filename, repo_type="dataset")
        self.response = requests.get(
            url,
            headers={"Authorization": f"Bearer {get_token()}"},
            stream=True,
            timeout=120,
        )
        self.response.raise_for_status()
        self.response.raw.decode_content = False
        return True

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            raise io.UnsupportedOperation("SplitTarReader requires bounded reads")
        chunks: list[bytes] = []
        remaining = size
        while remaining and self.response is not None:
            chunk = self.response.raw.read(remaining)
            if chunk:
                chunks.append(chunk)
                self.bytes_read += len(chunk)
                remaining -= len(chunk)
            elif not self._open_next():
                break
        return b"".join(chunks)


def select_samples(
    dataset: str,
    classes: list[str],
    manifests_root: Path,
    samples_root: Path,
    samples_per_class: int,
) -> None:
    dataset_root = samples_root / dataset.lower()
    existing = {
        label: sorted((dataset_root / safe_name(label)).glob("*.png"))
        for label in classes
    }
    if all(len(files) >= samples_per_class for files in existing.values()):
        print(f"{dataset}: samples already complete", flush=True)
        return

    index = load_manifest_index(manifests_root, dataset)
    counts = {label: len(existing[label]) for label in classes}
    info = HfApi().dataset_info(f"histai/{dataset}")
    filenames = sorted(
        sibling.rfilename for sibling in info.siblings
        if ".tar." in sibling.rfilename
    )
    reader = SplitTarReader(f"histai/{dataset}", filenames)
    print(f"{dataset}: selecting {samples_per_class} patches for {len(classes)} classes", flush=True)
    with tarfile.open(fileobj=reader, mode="r|") as archive:
        for member in archive:
            filename = Path(member.name).name
            label = index.get(filename)
            if label not in counts or counts[label] >= samples_per_class or not member.isfile():
                continue
            source = archive.extractfile(member)
            if source is None:
                continue
            output_dir = dataset_root / safe_name(label)
            output_dir.mkdir(parents=True, exist_ok=True)
            output = output_dir / filename
            if output.exists():
                continue
            output.write_bytes(source.read())
            counts[label] += 1
            complete = sum(value >= samples_per_class for value in counts.values())
            print(
                f"  {label}: {counts[label]}/{samples_per_class} "
                f"({complete}/{len(classes)} classes complete; {reader.bytes_read / 1e9:.2f} GB)",
                flush=True,
            )
            if complete == len(classes):
                break
    missing = {label: count for label, count in counts.items() if count < samples_per_class}
    if missing:
        raise RuntimeError(f"Missing samples for {dataset}: {missing}")


def safe_name(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")


def wrap_text(draw: ImageDraw.ImageDraw, text: str, label_font: ImageFont.ImageFont, width: int) -> list[str]:
    lines: list[str] = []
    for word in text.split():
        candidate = f"{lines[-1]} {word}" if lines else word
        if lines and draw.textlength(candidate, font=label_font) > width:
            lines.append(word)
        elif lines:
            lines[-1] = candidate
        else:
            lines.append(word)
    return lines


def make_panel(
    dataset: str,
    classes: list[str],
    samples_root: Path,
    samples_per_class: int,
    title: str | None = None,
) -> Image.Image:
    tile, gap, label_width, margin, header = 160, 8, 420, 16, 82
    width = margin * 2 + label_width + samples_per_class * tile + (samples_per_class - 1) * gap
    height = header + margin + len(classes) * tile + (len(classes) - 1) * gap + margin
    panel = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(panel)
    title_font, label_font = get_font(44, True), get_font(40, True)
    heading = title if title is not None else f"{dataset} ({len(classes)} classes)"
    draw.text((margin, 10), heading, fill="#111827", font=title_font)
    root = samples_root / dataset.lower()
    for row, label in enumerate(classes):
        y = header + margin + row * (tile + gap)
        draw.rounded_rectangle(
            (margin, y, margin + label_width - gap, y + tile),
            radius=8,
            fill=LABEL_COLORS[dataset],
        )
        lines = wrap_text(draw, label, label_font, label_width - 45)
        line_height = 48
        label_y = y + (tile - len(lines) * line_height) // 2
        for line_number, line in enumerate(lines):
            draw.text(
                (margin + 20, label_y + line_number * line_height),
                line,
                fill="#1f2937",
                font=label_font,
            )
        files = sorted((root / safe_name(label)).glob("*.png"))[:samples_per_class]
        if len(files) != samples_per_class:
            raise RuntimeError(f"Expected {samples_per_class} files for {dataset}/{label}")
        for col, path in enumerate(files):
            x = margin + label_width + col * (tile + gap)
            with Image.open(path) as image:
                patch = ImageOps.fit(image.convert("RGB"), (tile, tile))
            panel.paste(patch, (x, y))
    return panel


def make_grid(samples_root: Path, output: Path, samples_per_class: int) -> None:
    # Independent columns keep the 32/37 class-row totals more balanced than strict aligned rows.
    columns = [
        ["SPIDER-breast", "SPIDER-thorax"],
        ["SPIDER-colorectal", "SPIDER-skin"],
    ]
    panel_gap, column_gap, outer = 30, 30, 20
    rendered = {
        dataset: make_panel(dataset, DATASETS[dataset], samples_root, samples_per_class)
        for dataset in DATASETS
    }
    column_widths = [max(rendered[name].width for name in column) for column in columns]
    column_heights = [sum(rendered[name].height for name in column) + panel_gap for column in columns]
    canvas = Image.new(
        "RGB",
        (outer * 2 + sum(column_widths) + column_gap, outer * 2 + max(column_heights)),
        "white",
    )
    x = outer
    for column_number, column in enumerate(columns):
        y = outer
        for dataset in column:
            canvas.paste(rendered[dataset], (x, y))
            y += rendered[dataset].height + panel_gap
        x += column_widths[column_number] + column_gap
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=95, dpi=(300, 300))
    print(f"Saved {output} ({canvas.width} x {canvas.height})", flush=True)


def make_pair_grid(
    datasets: list[str],
    samples_root: Path,
    output: Path,
    samples_per_class: int,
) -> None:
    """Place two dataset panels side by side as a page-friendly figure."""
    panels = [
        make_panel(dataset, DATASETS[dataset], samples_root, samples_per_class)
        for dataset in datasets
    ]
    gap, outer = 30, 20
    canvas = Image.new(
        "RGB",
        (
            outer * 2 + sum(panel.width for panel in panels) + gap,
            outer * 2 + max(panel.height for panel in panels),
        ),
        "white",
    )
    x = outer
    for panel in panels:
        canvas.paste(panel, (x, outer))
        x += panel.width + gap
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=95, dpi=(300, 300))
    print(f"Saved {output} ({canvas.width} x {canvas.height})", flush=True)


def make_flow_grid(samples_root: Path, output: Path, samples_per_class: int) -> None:
    """Flow skin into column two, followed by thorax, to reduce figure height."""
    skin = DATASETS["SPIDER-skin"]
    left = make_panel(
        "SPIDER-skin",
        skin[:19],
        samples_root,
        samples_per_class,
        title="SPIDER-skin (24 classes)",
    )
    skin_continued = make_panel(
        "SPIDER-skin",
        skin[19:],
        samples_root,
        samples_per_class,
        title="SPIDER-skin (continued)",
    )
    thorax = make_panel(
        "SPIDER-thorax",
        DATASETS["SPIDER-thorax"],
        samples_root,
        samples_per_class,
    )
    gap, outer = 30, 20
    right_height = skin_continued.height + gap + thorax.height
    canvas = Image.new(
        "RGB",
        (
            outer * 2 + left.width + gap + max(skin_continued.width, thorax.width),
            outer * 2 + max(left.height, right_height),
        ),
        "white",
    )
    canvas.paste(left, (outer, outer))
    right_x = outer + left.width + gap
    canvas.paste(skin_continued, (right_x, outer))
    canvas.paste(thorax, (right_x, outer + skin_continued.height + gap))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=95, dpi=(300, 300))
    print(f"Saved {output} ({canvas.width} x {canvas.height})", flush=True)


def make_dataset_grid(
    dataset: str,
    samples_root: Path,
    output: Path,
    samples_per_class: int,
) -> None:
    """Make one dataset section whose classes flow across two balanced columns."""
    classes = DATASETS[dataset]
    midpoint = (len(classes) + 1) // 2
    blocks = [
        make_panel(dataset, classes[:midpoint], samples_root, samples_per_class, title=""),
        make_panel(dataset, classes[midpoint:], samples_root, samples_per_class, title=""),
    ]
    gap, outer = 30, 20
    canvas = Image.new(
        "RGB",
        (
            outer * 2 + sum(block.width for block in blocks) + gap,
            outer * 2 + max(block.height for block in blocks),
        ),
        "white",
    )
    x = outer
    for block in blocks:
        canvas.paste(block, (x, outer))
        x += block.width + gap
    draw = ImageDraw.Draw(canvas)
    heading = f"{dataset} ({len(classes)} classes)"
    heading_font = get_font(54, True)
    heading_width = draw.textlength(heading, font=heading_font)
    draw.text(
        ((canvas.width - heading_width) / 2, outer + 2),
        heading,
        fill="#111827",
        font=heading_font,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=95, dpi=(300, 300))
    print(f"Saved {output} ({canvas.width} x {canvas.height})", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifests-root", type=Path, required=True)
    parser.add_argument("--samples-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-class", type=int, default=3)
    args = parser.parse_args()
    for dataset, classes in DATASETS.items():
        select_samples(
            dataset, classes, args.manifests_root, args.samples_root, args.samples_per_class
        )
    for dataset in DATASETS:
        make_dataset_grid(
            dataset,
            args.samples_root,
            args.output.parent / f"{dataset.lower()}-patch-grid.png",
            args.samples_per_class,
        )


if __name__ == "__main__":
    main()
