#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse, os, sys, heapq, copy
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from adjustText import adjust_text


KIND2DIR = {"sft": "sft", "pt": "pt", "ifl": "ifl"}


def _load_grad_block(root: str, kind: str, dim: int, device: str) -> torch.Tensor:
    path = os.path.join(root, KIND2DIR[kind], f"dim{dim}", "all_orig.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return torch.load(path, map_location=device).float()


@torch.inference_mode()
def _pairwise_euclidean(mat: torch.Tensor, chunk: int | None = None) -> np.ndarray:
    N = mat.size(0)
    if chunk is None:
        dmat = torch.cdist(mat, mat)
        return dmat.cpu().numpy().astype(np.float32)
    dmat = torch.empty((N, N), dtype=torch.float32, device="cpu")
    for i in range(0, N, chunk):
        j = slice(i, min(i + chunk, N))
        dmat[j] = torch.cdist(mat[j].to(mat.device), mat).cpu()
    return dmat.numpy()


from lazy_greedy import FacilityLocation


def _heappush_max(h, it):
    h.append(it)
    heapq._siftdown_max(h, 0, len(h) - 1)


def _heappop_max(h):
    last = h.pop()
    if h:
        ret = h[0]
        h[0] = last
        heapq._siftup_max(h, 0)
        return ret
    return last


def _lazy_greedy_heap(F, V: Sequence[int], k: int, pbar=None):
    sset, vals = [], []
    order: list = []
    heapq._heapify_max(order)
    for idx in V:
        _heappush_max(order, (F.inc(sset, idx), idx))

    while order and len(sset) < k:
        el_val, el_idx = _heappop_max(order)
        true_gain = F.inc(sset, el_idx)
        if not order:
            F.add(sset, el_idx)
            sset.append(el_idx)
            vals.append(F.curVal)
            if pbar:
                pbar.update(1)
        else:
            top_val, top_idx = _heappop_max(order)
            if true_gain >= top_val:
                F.add(sset, el_idx)
                sset.append(el_idx)
                vals.append(F.curVal)
                if pbar:
                    pbar.update(1)
                _heappush_max(order, (top_val, top_idx))
            else:
                _heappush_max(order, (true_gain, el_idx))
                _heappush_max(order, (top_val, top_idx))
    return np.asarray(sset, dtype=np.int64), vals


def _select_subset(S: np.ndarray, k: int, desc="", verbose=True):
    N = S.shape[0]
    V = list(range(N))
    with tqdm(
        total=k, disable=not verbose, desc=desc or "FacilityLocation", unit="samples"
    ) as pbar:
        F = FacilityLocation(S, V=V)
        subset, _ = _lazy_greedy_heap(F, V, k, pbar)
    w = np.zeros(len(subset), dtype=np.float32)
    assign = S[:, subset].argmax(axis=1)
    for j in assign:
        w[j] += 1.0
    return subset, w


def _compute_errors(d_pt: np.ndarray, d_ifl: np.ndarray, subset: np.ndarray):
    e_pt = d_pt[:, subset].min(axis=1).sum()
    e_ifl = d_ifl[:, subset].min(axis=1).sum()
    return float(e_pt), float(e_ifl)


def _robust_sigma(vals: Sequence[float]) -> float:
    vals = np.asarray(vals, np.float64)
    if len(vals) == 0:
        return 0.0
    med, mad = np.median(vals), np.median(np.abs(vals - np.median(vals)))
    return 1.4826 * mad


def _plateau_detection(e_ifls, alphas, *, min_consecutive=3, prefer_alpha_ge=0.95):
    alphas = np.asarray(alphas, dtype=np.float64)
    e_ifls = np.asarray(e_ifls, dtype=np.float64)
    diffs = np.abs(np.diff(e_ifls))
    sigma = max(_robust_sigma(diffs), 1e-8)
    cand = [
        i + 1
        for i in range(len(diffs) - min_consecutive + 1)
        if np.all(diffs[i : i + min_consecutive] <= sigma)
    ]
    for i in cand:
        if alphas[i] >= prefer_alpha_ge:
            return i, sigma
    return (cand[0] if cand else len(alphas) - 1), sigma


def _alpha_grid(alpha_min=0.1, alpha_max=0.999, coarse_step=0.01, fine_step_min=0.001):
    alphas, step, a = [], coarse_step, alpha_min
    while a <= alpha_max + 1e-12:
        alphas.append(round(a, 6))
        a += step
        if a >= 0.9 and step > fine_step_min:
            step = max(step / 2.0, fine_step_min)
    return np.unique(alphas)


def _cluster_weights(S: np.ndarray, subset: np.ndarray):
    w = np.zeros(len(subset), np.float32)
    assign = S[:, subset].argmax(axis=1)
    for j in assign:
        w[j] += 1.0
    return w


def _save_npz(S: np.ndarray, subset, weights, path: str, ratios=(0.05, 0.1, 0.15)):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    print(f"   → saved main subset k={len(subset)}  to {path}")
    for r in ratios:
        k_i = max(1, int(round(r * S.shape[0])))
        sub_i = subset[:k_i]
        w_i = _cluster_weights(S, sub_i)
        sub_path = f"{path}_{r:.2f}"
        np.savez(sub_path, subset=sub_i, weights=w_i)
        print(f"     ratio {r:.2f} (k={k_i}) → {sub_path}")


def _get_args(argv=None):
    ap = argparse.ArgumentParser(
        "run_MINT.py  (auto α + coreset)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--grad-dir", required=True, help="root containing pt/, ifl/, (opt) sft/"
    )
    ap.add_argument("--proj-dim", type=int, default=8192)
    # ap.add_argument("--subset-ratio", type=float, default=0.1, help="k / N")
    ap.add_argument(
        "--ratios",
        type=float,
        nargs="+",
        default=[0.05, 0.1, 0.15],
        metavar="R",
        help="list of ratios (0<R<=1) for additional subset dumps; the maximum value determines the main subset size k/N",
    )

    ap.add_argument("--save", required=True, help="output .npz base path")

    # mutually exclusive α / coef / auto
    g = ap.add_mutually_exclusive_group(required=False)
    g.add_argument("--alpha", type=float, help="manual α (0<α<1)")
    g.add_argument(
        "--coef",
        type=float,
        nargs=3,
        metavar=("SFT", "PT", "IFL"),
        help="explicit coefficients",
    )
    ap.add_argument(
        "--preheat-ratio", type=float, default=0.3, help="auto-α pre-sample fraction"
    )
    ap.add_argument("--seed", type=int, default=0)

    return ap.parse_args(argv)


def main(argv=None):
    args = _get_args(argv)
    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ratios_list = sorted(set(float(r) for r in args.ratios))
    if not ratios_list:
        raise ValueError("--ratios must contain at least one positive value")
    subset_ratio = max(ratios_list)
    print(f">>> Ratios for subsets: {ratios_list} (max={subset_ratio})")
    if not 0 < subset_ratio <= 1:
        raise ValueError("ratios must be in (0,1]")
    print("\n>>> Loading gradient projections & pairwise distances …")
    grads_pt = _load_grad_block(args.grad_dir, "pt", args.proj_dim, device)
    grads_ifl = _load_grad_block(args.grad_dir, "ifl", args.proj_dim, device)
    grads_sft = _load_grad_block(args.grad_dir, "sft", args.proj_dim, device)
    d_pt = _pairwise_euclidean(grads_pt)
    d_ifl = _pairwise_euclidean(grads_ifl)
    d_sft = _pairwise_euclidean(grads_sft)
    N = d_pt.shape[0]
    k = max(1, int(round(subset_ratio * N)))
    print(f"   dataset N={N}; final subset k={k}")
    del grads_pt, grads_ifl, grads_sft
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    if args.coef:
        sft_c, pt_c, ifl_c = args.coef
        coeffs = {"sft": sft_c, "pt": pt_c, "ifl": ifl_c}
        coeffs = {k: v for k, v in coeffs.items() if abs(v) > 1e-12}
        print(">>> Explicit-coef mode:", coeffs)
        D_comb = d_pt * coeffs.get("pt", 0) + d_ifl * coeffs.get("ifl", 0) + d_sft * coeffs.get("sft", 0)
        D_max = D_comb.max()
        S = (D_max - D_comb).astype(np.float32, copy=False)
        subset, weights = _select_subset(S, k)
        _save_npz(S, subset, weights, args.save)
        print(">>> Done (explicit-coef).")
        return

    if args.alpha:
        if not 0 < args.alpha < 1:
            raise ValueError("alpha must be in (0,1)")
        alpha_star = args.alpha
        print(f">>> Manual α={alpha_star:.4f}")
    else:

        print(">>> Auto-α search …")
        n_pre = max(2, int(round(args.preheat_ratio * N)))
        pre_ids = rng.choice(N, size=n_pre, replace=False)
        d_pt_pre = d_pt[np.ix_(pre_ids, pre_ids)]
        d_ifl_pre = d_ifl[np.ix_(pre_ids, pre_ids)]
        k_pre = max(1, int(round(subset_ratio * n_pre)))

        alphas = _alpha_grid()
        e_pts, e_ifls = [], []
        for α in tqdm(alphas, desc="scan α", unit="α"):
            D = d_pt_pre / α + d_ifl_pre / (1 - α)
            S = (D.max() - D).astype(np.float32, copy=False)
            subset_pre, _ = _select_subset(S, k_pre, verbose=False)
            ep, ei = _compute_errors(d_pt_pre, d_ifl_pre, subset_pre)
            e_pts.append(ep)
            e_ifls.append(ei)

        idx_star, σ = _plateau_detection(e_ifls, alphas)
        alpha_star = float(alphas[idx_star])
        print(f"   plateau idx={idx_star}, α★={alpha_star:.4f}, σ={σ:.4g}")
        # save lists in one csv
        e_pts = np.asarray(e_pts, dtype=np.float64)
        e_ifls = np.asarray(e_ifls, dtype=np.float64)
        np.savetxt(
            args.save + "_errors.csv",
            np.column_stack((alphas, e_pts, e_ifls)),
            delimiter=",",
            header="α,E_PT,E_IFL",
            fmt="%.6f,%.6f,%.6f",
        )
        print(f"   errors saved to {args.save}_errors.csv")

        pareto_png = args.save + "_pareto.png"
        plateau_png = args.save + "_plateau.png"

        plt.figure(figsize=(8, 6))
        plt.plot(e_ifls, e_pts, "o-")
        key_idx = np.linspace(0, len(alphas) - 1, 20, dtype=int)
        texts = []
        for i in key_idx:
            texts.append(plt.text(e_ifls[i], e_pts[i], f"{alphas[i]:.2f}", fontsize=8))
        adjust_text(texts, arrowprops=dict(arrowstyle="->", lw=0.3, color="gray"))
        plt.xlabel("E_IFL")
        plt.ylabel("E_PT")
        plt.grid(True, ls="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(pareto_png, dpi=120)
        plt.close()

        diffs = np.abs(np.diff(e_ifls))
        plt.figure(figsize=(8, 6))
        ax1 = plt.gca()
        ax1.plot(alphas, e_ifls, "o-", label="E_IFL")
        ax1.scatter(alphas[idx_star], e_ifls[idx_star], s=120, marker="*", label="α★")
        ax1.set_xlabel("α")
        ax1.set_ylabel("E_IFL")
        ax2 = ax1.twinx()
        ax2.plot(alphas[1:], diffs, "--", label="|ΔE_IFL|")
        ax2.set_ylabel("|ΔE_IFL|")
        ax2.axhline(σ, ls=":", label="σ")
        lines, labels = ax1.get_legend_handles_labels()
        lines2, lbl2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + lbl2)
        plt.tight_layout()
        plt.savefig(plateau_png, dpi=120)
        plt.close()
        print(f"   pareto plot  → {pareto_png}")
        print(f"   plateau plot → {plateau_png}")

    coeffs = {"pt": 1 / alpha_star, "ifl": 1 / (1 - alpha_star)}
    print(f">>> Final coefficients: PT={coeffs['pt']:.3f}, IFL={coeffs['ifl']:.3f}")
    D_comb = d_pt * coeffs["pt"] + d_ifl * coeffs["ifl"]
    S = (D_comb.max() - D_comb).astype(np.float32, copy=False)
    subset, weights = _select_subset(S, k)
    _save_npz(S, subset, weights, args.save,ratios=ratios_list)
    print(">>> Done.")


if __name__ == "__main__":
    main()