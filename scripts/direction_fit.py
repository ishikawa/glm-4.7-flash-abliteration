#!/usr/bin/env python3
"""Fit normalized harmful-minus-benign directions from completed activations."""

import argparse
import json
from pathlib import Path

import numpy as np

SITES = ("attn_out", "ffn_out", "resid_mean")


def activations(directory):
    directory = Path(directory)
    n = len(json.loads((directory / "manifest.json").read_text())["records"])
    if n < 2:
        raise ValueError("at least two completed records required")
    arrays = {}
    for name in ("attn_out", "ffn_out", "resid"):
        a = np.load(directory / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        if (
            a.dtype != np.dtype("<f4")
            or n > len(a)
            or a.ndim != (4 if name == "resid" else 3)
        ):
            raise ValueError(f"invalid activation: {name}")
        if name == "resid" and a.shape[2] != 1:
            raise ValueError("expected one residual stream")
        arrays[name] = a[:n]
    shapes = {(a.shape[1], a.shape[-1]) for a in arrays.values()}
    if len(shapes) != 1:
        raise ValueError("activation shapes differ")
    return arrays


def layer_values(arrays, site, layer):
    a = arrays["resid" if site == "resid_mean" else site][:, layer].astype(np.float64)
    if site == "resid_mean":
        a = a.mean(axis=1)
    if not np.isfinite(a).all():
        raise ValueError("nonfinite activation")
    return a


def auc(positive, negative):
    # Pair counts via sorted negatives; half credit for ties, bounded memory.
    s = np.sort(negative)
    left = np.searchsorted(s, positive, side="left")
    right = np.searchsorted(s, positive, side="right")
    return float(np.mean((left + right) / 2) / len(s))


def fit(
    harmful,
    benign,
    output,
    report,
    held_harmful=None,
    held_benign=None,
    massive_ratio=None,
    benign_means_out=None,
):
    """Fit unit mean differences, optionally saving separate FP32 benign means.

    Means use completed fit records only, after residual-stream averaging and
    the same site/layer massive-activation mask as the differences. They are
    neither unit-normalized nor standardized; pooled standard deviation is a
    reporting statistic only. Held-out records never contribute to these means.
    """
    if benign_means_out and Path(benign_means_out).resolve() in {
        Path(output).resolve(),
        Path(report).resolve(),
    }:
        raise ValueError("benign means must use a separate output path")
    if massive_ratio is not None and (
        not np.isfinite(massive_ratio) or massive_ratio <= 0
    ):
        raise ValueError("massive ratio must be positive and finite")
    h, b = activations(harmful), activations(benign)
    if h["attn_out"].shape[1:] != b["attn_out"].shape[1:]:
        raise ValueError("fit shapes differ")
    if (held_harmful is None) != (held_benign is None):
        raise ValueError("both held-out directories required")
    held = (
        (activations(held_harmful), activations(held_benign)) if held_harmful else None
    )
    directions, rows, benign_means = {}, [], {}
    for site in SITES:
        ds = []
        bs = []
        for layer in range(h["attn_out"].shape[1]):
            x, y = layer_values(h, site, layer), layer_values(b, site, layer)
            means = (np.abs(x).sum(axis=0) + np.abs(y).sum(axis=0)) / (len(x) + len(y))
            mask = (
                means > np.median(means) * massive_ratio
                if massive_ratio is not None
                else np.zeros(x.shape[1], dtype=bool)
            )
            x[:, mask] = 0
            y[:, mask] = 0
            if benign_means_out:
                bs.append(y.mean(axis=0).astype(np.float32))
            delta = x.mean(axis=0) - y.mean(axis=0)
            norm = float(np.linalg.norm(delta))
            if norm == 0:
                raise ValueError(f"zero direction: {site} {layer}")
            u = delta / norm
            px, py = x @ u, y @ u
            pooled = np.sqrt(
                ((len(x) - 1) * px.var(ddof=1) + (len(y) - 1) * py.var(ddof=1))
                / (len(x) + len(y) - 2)
            )
            row = dict(
                site=site,
                massive_indices=np.flatnonzero(mask).tolist(),
                massive_count=int(mask.sum()),
                layer=layer,
                mean_difference_norm=norm,
                standardized_difference=float((px.mean() - py.mean()) / pooled)
                if pooled
                else None,
            )
            if held:
                hx, hy = (layer_values(a, site, layer) for a in held)
                if hx.shape[1:] != x.shape[1:] or hy.shape[1:] != y.shape[1:]:
                    raise ValueError("held-out shapes differ")
                hx[:, mask] = 0
                hy[:, mask] = 0
                row["auc"] = auc(hx @ u, hy @ u)
            rows.append(row)
            ds.append(u.astype(np.float32))
        directions[site] = np.stack(ds)
        if benign_means_out:
            benign_means[site] = np.stack(bs)
    np.savez(output, **directions)
    if benign_means_out:
        np.savez(benign_means_out, **benign_means)
    Path(report).write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
    print("site layer mean_difference_norm standardized_difference auc")
    for row in rows:
        print(
            *(
                row.get(k, "-")
                for k in (
                    "site",
                    "layer",
                    "mean_difference_norm",
                    "standardized_difference",
                    "auc",
                )
            )
        )
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ("harmful", "benign", "output", "report"):
        p.add_argument("--" + flag, required=True, type=Path)
    p.add_argument("--massive-ratio", type=float)
    p.add_argument("--benign-means-out", type=Path)
    p.add_argument("--held-harmful", type=Path)
    p.add_argument("--held-benign", type=Path)
    fit(**vars(p.parse_args()))


if __name__ == "__main__":
    main()
