"""
Plotting functions for ITI evaluation results.

Usage:
    python plot_results.py non_quantized/cont_informative_results.txt
    python plot_results.py non_quantized/cont_informative_results.txt --cmap YlGnBu --metric all
    python plot_results.py non_quantized/cont_informative_results.txt --output-dir plots/

Results file format:
    (optional blank line)
    meta-llama/Meta-Llama-3-8B-Instruct
    ConflictQA
    3008:Base model — context*informative: 0.796 [0.771, 0.821]  context: 0.882 ...  informative: 0.839 ...
    6025:k=8, alpha=2.0 — context*informative: 0.785 [0.757, 0.810]  context: ...  informative: ...
"""

import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_METRIC_RE = re.compile(
    r'context\*informative:\s*([\d.]+)\s*\[([\d.]+),\s*([\d.]+)\]'
    r'.*?context:\s*([\d.]+)\s*\[([\d.]+),\s*([\d.]+)\]'
    r'.*?informative:\s*([\d.]+)\s*\[([\d.]+),\s*([\d.]+)\]'
)

vmin = 0.6
vmax = 0.85

def parse_results_file(filepath):
    """Parse a results file and return a data dict.

    The first non-empty line is the model name, the second is the dataset name.
    All subsequent lines are metric records.

    Returns
    -------
    dict with keys:
        'model'    : str
        'dataset'  : str
        'base'     : dict — {'cont_inf': (score, lo, hi), 'context': ..., 'informative': ...}
        'variants' : dict — {(k, alpha): same structure as 'base'}
        'ks'       : sorted list[int]
        'alphas'   : sorted list[float]
    """
    filepath = Path(filepath)
    model = filepath.stem
    dataset = "unknown"
    base = None
    variants = {}

    header_lines_seen = 0
    with open(filepath) as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue

            # First two non-empty lines are model name and dataset name
            if header_lines_seen == 0:
                model = stripped
                header_lines_seen += 1
                continue
            if header_lines_seen == 1:
                dataset = stripped
                header_lines_seen += 1
                continue

            m = _METRIC_RE.search(stripped)
            if not m:
                continue

            vals = tuple(float(x) for x in m.groups())
            entry = {
                'cont_inf':    (vals[0], vals[1], vals[2]),
                'context':     (vals[3], vals[4], vals[5]),
                'informative': (vals[6], vals[7], vals[8]),
            }

            if 'Base model' in stripped:
                base = entry
            else:
                km = re.search(r'k=(\d+),\s*alpha=([\d.]+)', stripped)
                if km:
                    k = int(km.group(1))
                    alpha = float(km.group(2))
                    variants[(k, alpha)] = entry

    if base is None:
        raise ValueError(f"No 'Base model' line found in {filepath}")

    ks = sorted({k for k, _ in variants})
    alphas = sorted({a for _, a in variants})

    return {
        'model': model,
        'dataset': dataset,
        'base': base,
        'variants': variants,
        'ks': ks,
        'alphas': alphas,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_score_matrix(data, metric_key):
    """Return score, lo, hi matrices each of shape (len(ks), len(alphas))."""
    ks = data['ks']
    alphas = data['alphas']
    shape = (len(ks), len(alphas))
    score = np.full(shape, np.nan)
    lo    = np.full(shape, np.nan)
    hi    = np.full(shape, np.nan)

    for ki, k in enumerate(ks):
        for ai, alpha in enumerate(alphas):
            entry = data['variants'].get((k, alpha))
            if entry is not None:
                score[ki, ai] = entry[metric_key][0]
                lo[ki, ai]    = entry[metric_key][1]
                hi[ki, ai]    = entry[metric_key][2]

    return score, lo, hi


def _text_color(norm_value, cmap):
    """Return 'white' or 'black' for readable annotation text."""
    rgba = plt.get_cmap(cmap)(norm_value)
    luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
    return 'white' if luminance < 0.45 else 'black'


def _plot_metric(data, metric_key, title, cmap, output_path=None):
    """Core plotting function used by the e public functions.

    Produces two stacked subplots sharing a colormap normalised to [0, 1]:
      - Top:    base model as a single full-width cell
      - Bottom: k × alpha matrix
    Cell text format: "score\n[lo, hi]"  (lo/hi are 2.5th / 97.5th percentiles)
    """
    ks = data['ks']
    alphas = data['alphas']
    nk = len(ks)
    ncols = len(alphas)

    col_labels = [f'α={a:g}' for a in alphas]
    k_labels   = [f'k={k}' for k in ks]

    score_mat, lo_mat, hi_mat = _build_score_matrix(data, metric_key)
    base_s, base_lo, base_hi  = data['base'][metric_key]

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    cell_w = 2.2
    cell_h = 1.0
    fig_w = cell_w * ncols + 2.5
    fig_h = cell_h * (nk + 1) + 2.5   # +1 for the base row

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs  = fig.add_gridspec(2, 1, height_ratios=[1, nk], hspace=0.08)
    ax_base = fig.add_subplot(gs[0])
    ax_main = fig.add_subplot(gs[1])

    # --- Base row (single full-width cell) ---
    ax_base.imshow([[base_s]], cmap=cmap, norm=norm, aspect='auto')
    tc = _text_color(norm(base_s), cmap)
    ax_base.text(0.5, 0.5,
                 f'{base_s:.3f}\n[{base_lo:.3f}, {base_hi:.3f}]',
                 ha='center', va='center', fontsize=8,
                 color=tc, fontweight='bold', linespacing=1.4,
                 transform=ax_base.transAxes)
    ax_base.set_yticks([0])
    ax_base.set_yticklabels(['base'], fontsize=10)
    ax_base.set_xticks([])
    ax_base.tick_params(bottom=False)

    # --- k × alpha matrix ---
    ax_main.imshow(score_mat, cmap=cmap, norm=norm, aspect='auto')
    ax_main.set_xticks(range(ncols))
    ax_main.set_xticklabels(col_labels, rotation=45, ha='right', fontsize=10)
    ax_main.set_yticks(range(nk))
    ax_main.set_yticklabels(k_labels, fontsize=10)

    for r in range(nk):
        for c in range(ncols):
            s = score_mat[r, c]
            if not np.isnan(s):
                tc = _text_color(norm(s), cmap)
                ax_main.text(c, r,
                             f'{s:.3f}\n[{lo_mat[r, c]:.3f}, {hi_mat[r, c]:.3f}]',
                             ha='center', va='center', fontsize=8,
                             color=tc, fontweight='bold', linespacing=1.4)

    # Shared colorbar as legend
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax_base, ax_main], orientation='vertical',
                        fraction=0.046, pad=0.04)
    cbar.set_label('Score [0 – 1]', fontsize=9)
    cbar.set_ticks(np.linspace(0, 1, 11))

    ax_base.set_title(
        f'{title}\n'
        f'Judge: {data["model"]}   |   Dataset: {data["dataset"]}\n'
        f'Values in brackets are [2.5th, 97.5th] percentiles',
        fontsize=10, pad=10,
    )

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {output_path}')
    else:
        plt.show()
    plt.close(fig)

def _plot_percent(data, metric_key, title, cmap, output_path=None):
    """Core plotting function used by the three public functions.

    Produces two stacked subplots sharing a colormap normalised to [0, 1]:
      - Top:    base model as a single full-width cell
      - Bottom: k × alpha matrix
    Cell text format: "score\n[lo, hi]"  (lo/hi are 2.5th / 97.5th percentiles)
    """
    ks = data['ks']
    alphas = data['alphas']
    nk = len(ks)
    ncols = len(alphas)

    col_labels = [f'α={a:g}' for a in alphas]
    k_labels   = [f'k={k}' for k in ks]

    score_mat, lo_mat, hi_mat = _build_score_matrix(data, metric_key)
    base_s, base_lo, base_hi  = data['base'][metric_key]
    base_s *= 100
    base_lo *= 100
    base_hi *= 100

    norm = mcolors.Normalize(vmin=vmin * 100, vmax=vmax * 100)

    cell_w = 2.2
    cell_h = 1.0
    fig_w = cell_w * ncols + 2.5
    fig_h = cell_h * (nk + 1) + 2.5   # +1 for the base row

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs  = fig.add_gridspec(2, 1, height_ratios=[1, nk], hspace=0.08)
    ax_base = fig.add_subplot(gs[0])
    ax_main = fig.add_subplot(gs[1])

    # --- Base row (single full-width cell) ---
    ax_base.imshow([[base_s]], cmap=cmap, norm=norm, aspect='auto')
    tc = _text_color(norm(base_s), cmap)
    ax_base.text(0.5, 0.5,
                 f'{base_s:.1f}\n[{base_lo:.1f}, {base_hi:.1f}]',
                 ha='center', va='center', fontsize=8,
                 color=tc, fontweight='bold', linespacing=1.4,
                 transform=ax_base.transAxes)
    ax_base.set_yticks([0])
    ax_base.set_yticklabels(['base'], fontsize=10)
    ax_base.set_xticks([])
    ax_base.tick_params(bottom=False)

    # --- k × alpha matrix ---
    ax_main.imshow(score_mat * 100, cmap=cmap, norm=norm, aspect='auto')
    ax_main.set_xticks(range(ncols))
    ax_main.set_xticklabels(col_labels, rotation=45, ha='right', fontsize=10)
    ax_main.set_yticks(range(nk))
    ax_main.set_yticklabels(k_labels, fontsize=10)

    for r in range(nk):
        for c in range(ncols):
            s = score_mat[r, c]
            if not np.isnan(s):
                tc = _text_color(norm(s * 100), cmap)
                ax_main.text(c, r,
                             f'{s*100:.1f}\n[{lo_mat[r, c]*100:.1f}, {hi_mat[r, c]*100:.1f}]',
                             ha='center', va='center', fontsize=8,
                             color=tc, fontweight='bold', linespacing=1.4)

    # Shared colorbar as legend
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax_base, ax_main], orientation='vertical',
                        fraction=0.046, pad=0.04)
    cbar.set_label('Score (%)', fontsize=9)
    cbar.set_ticks(np.linspace(vmin * 100, vmax * 100, 6))

    ax_base.set_title(
        f'{title} (%)\n'
        f'Judge: {data["model"]}   |   Dataset: {data["dataset"]}\n'
        f'Values in brackets are [2.5th, 97.5th] percentiles',
        fontsize=10, pad=10,
    )

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {output_path}')
    else:
        plt.show()
    plt.close(fig)

# ---------------------------------------------------------------------------
# Public plotting functions
# ---------------------------------------------------------------------------

def plot_cont_informative(data, cmap='viridis', output_path=None, perc=False):
    """Plot the context × informative composite metric.

    Parameters
    ----------
    data : dict
        Parsed data dict from :func:`parse_results_file`.
    cmap : str
        Matplotlib colormap name. Normalisation is always [0, 1].
    output_path : str or Path, optional
        Save the figure here instead of displaying it.
    perc: bool, optional
        Plot in Percents
    """
    if perc:
        _plot_percent(data, 'cont_inf', 'Cont*Informative', cmap, output_path)
    else:
        _plot_metric(data, 'cont_inf', 'Cont*Informative', cmap, output_path)


def plot_contextual(data, cmap='viridis', output_path=None, perc=False):
    """Plot the contextual metric.

    Parameters
    ----------
    data : dict
        Parsed data dict from :func:`parse_results_file`.
    cmap : str
        Matplotlib colormap name. Normalisation is always [0, 1].
    output_path : str or Path, optional
        Save the figure here instead of displaying it.
    perc: bool, optional
        Plot in Percents
    """
    if perc:
        _plot_percent(data, 'context', 'Contextual', cmap, output_path)
    else:
        _plot_metric(data, 'context', 'Contextual', cmap, output_path)


def plot_informative(data, cmap='viridis', output_path=None, perc=False):
    """Plot the informative metric.

    Parameters
    ----------
    data : dict
        Parsed data dict from :func:`parse_results_file`.
    cmap : str
        Matplotlib colormap name. Normalisation is always [0, 1].
    output_path : str or Path, optional
        Save the figure here instead of displaying it.
    perc: bool, optional
        Plot in Percents
    """
    if perc:
        _plot_percent(data, 'informative', 'Informative', cmap, output_path)
    else:
        _plot_metric(data, 'informative', 'Informative', cmap, output_path)


# ---------------------------------------------------------------------------
# Probe accuracy scatter plot
# ---------------------------------------------------------------------------

def parse_accuracy_file(filepath):
    """Parse a space-separated probe accuracy matrix file.

    Returns
    -------
    np.ndarray of shape (num_layers, num_heads)
    """
    rows = []
    with open(filepath) as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                rows.append([float(x) for x in stripped.split()])
    return np.array(rows)


def plot_probe_scatter(truth_path, context_path, cmap='viridis', threshold=None,
                       output_path=None, perc=False):
    """Scatter plot comparing truth and context probe accuracies per attention head.

    Each point is one attention head, colored by layer index. Dashed threshold
    lines divide the plot into four quadrants:
      top-right    — generalist heads (high on both tasks)
      top-left     — context-only specialists
      bottom-right — truth-only specialists
      bottom-left  — low on both

    Parameters
    ----------
    truth_path : str or Path
        Path to the truth probe accuracy matrix file (space-separated floats,
        one row per layer, as written by head_probing.py).
    context_path : str or Path
        Path to the context probe accuracy matrix file (same format).
    cmap : str
        Matplotlib colormap name used for layer coloring.
    threshold : float or None
        Accuracy threshold in [0, 1] for the quadrant dividing lines.
        Defaults to the 75th percentile of each distribution independently.
    output_path : str or Path or None
        Save the figure here; display interactively if None.
    perc : bool
        Display accuracies as percentages instead of [0, 1].
    """
    truth_acc   = parse_accuracy_file(truth_path)
    context_acc = parse_accuracy_file(context_path)

    num_layers, num_heads = truth_acc.shape
    layer_idx = np.repeat(np.arange(num_layers), num_heads)
    t_flat = truth_acc.flatten()
    c_flat = context_acc.flatten()

    scale = 100 if perc else 1
    unit  = '%'  if perc else ''

    t_plot = t_flat * scale
    c_plot = c_flat * scale

    # Quadrant thresholds — independent per axis so different score ranges
    # on each task do not bias the boundary
    t_thr = (np.percentile(t_flat, 75) if threshold is None else threshold) * scale
    c_thr = (np.percentile(c_flat, 75) if threshold is None else threshold) * scale

    fig, ax = plt.subplots(figsize=(7, 6))

    sc = ax.scatter(t_plot, c_plot, c=layer_idx, cmap=cmap,
                    alpha=0.7, s=18, linewidths=0,
                    vmin=0, vmax=num_layers - 1)

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label('Layer', fontsize=10)
    cbar.set_ticks(np.linspace(0, num_layers - 1, min(num_layers, 9)).astype(int))

    # Threshold lines
    ax.axvline(t_thr, color='#555555', linestyle='--', linewidth=0.9, alpha=0.8)
    ax.axhline(c_thr, color='#555555', linestyle='--', linewidth=0.9, alpha=0.8)

    # Quadrant labels — placed at the midpoint of each quadrant
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    label_kw = dict(fontsize=7.5, color='#444444', alpha=0.85, ha='center', va='center')
    ax.text((xlim[0] + t_thr) / 2, (ylim[1] + c_thr) / 2, 'context\nspecialist', **label_kw)
    ax.text((xlim[1] + t_thr) / 2, (ylim[1] + c_thr) / 2, 'generalist',          **label_kw)
    ax.text((xlim[0] + t_thr) / 2, (ylim[0] + c_thr) / 2, 'neither',             **label_kw)
    ax.text((xlim[1] + t_thr) / 2, (ylim[0] + c_thr) / 2, 'truth\nspecialist',   **label_kw)

    ax.set_xlabel(f'Truth probe accuracy{unit}', fontsize=11)
    ax.set_ylabel(f'Context probe accuracy{unit}', fontsize=11)
    ax.set_title(
        'Attention head probe accuracies: Truthfulness vs. Context Grounding',
        fontsize=11,
    )

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {output_path}')
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Plot ITI evaluation metrics from a results file.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        'results_file', nargs='?', default=None,
        help='Path to results file (e.g. non_quantized/cont_informative_results.txt). '
             'Not required when --truth-acc and --context-acc are used.',
    )
    parser.add_argument(
        '--cmap', default='viridis',
        help='Matplotlib colormap name (default: viridis)',
    )
    parser.add_argument(
        '--metric',
        choices=['cont_inf', 'context', 'informative', 'all'],
        default='all',
        help='Which metric figure(s) to produce (default: all)',
    )
    parser.add_argument(
        '--output-dir', default=None,
        help='Directory to save figures as PNG files; shows interactively if omitted',
    )
    parser.add_argument(
        '--percent', default='false', choices=['true', 'false'],
        help='Display values as percentages instead of [0, 1] (default: false)',
    )
    parser.add_argument(
        '--truth-acc', default=None, metavar='FILE',
        help='Truth probe accuracy matrix file for scatter plot',
    )
    parser.add_argument(
        '--context-acc', default=None, metavar='FILE',
        help='Context probe accuracy matrix file for scatter plot',
    )
    parser.add_argument(
        '--threshold', type=float, default=None, metavar='FLOAT',
        help='Quadrant threshold in [0, 1] for the scatter plot '
             '(default: 75th percentile of each distribution independently)',
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    def out(name):
        return str(out_dir / name) if out_dir else None

    percentage = args.percent == 'true'

    # --- Scatter plot mode ---
    if args.truth_acc or args.context_acc:
        if not args.truth_acc or not args.context_acc:
            parser.error('--truth-acc and --context-acc must be provided together')
        plot_probe_scatter(
            args.truth_acc, args.context_acc,
            cmap=args.cmap,
            threshold=args.threshold,
            output_path=out('probe_scatter.png'),
            perc=percentage,
        )
        return

    # --- Heatmap mode ---
    if args.results_file is None:
        parser.error('results_file is required unless --truth-acc and --context-acc are provided')

    data = parse_results_file(args.results_file)

    print(f"Model:    {data['model']}")
    print(f"Dataset:  {data['dataset']}")
    print(f"ks:       {data['ks']}")
    print(f"alphas:   {data['alphas']}")
    print(f"Variants: {len(data['variants'])}")

    if args.metric in ('cont_inf', 'all'):
        plot_cont_informative(data, cmap=args.cmap, output_path=out('cont_informative.png'), perc=percentage)
    if args.metric in ('context', 'all'):
        plot_contextual(data, cmap=args.cmap, output_path=out('contextual.png'), perc=percentage)
    if args.metric in ('informative', 'all'):
        plot_informative(data, cmap=args.cmap, output_path=out('informative.png'), perc=percentage)


if __name__ == '__main__':
    main()
