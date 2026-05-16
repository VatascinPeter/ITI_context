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

_METRIC_RE_CONTEXT = re.compile(
    r'context\*info(?:rmative)?:\s*([\d.]+)\s*\[([\d.]+),\s*([\d.]+)\]'
    r'.*?context:\s*([\d.]+)\s*\[([\d.]+),\s*([\d.]+)\]'
    r'.*?info(?:rmative)?:\s*([\d.]+)\s*\[([\d.]+),\s*([\d.]+)\]'
)

_METRIC_RE_TRUTH = re.compile(
    r'true\*informative:\s*([\d.]+)\s*\[([\d.]+),\s*([\d.]+)\]'
    r'.*?true:\s*([\d.]+)\s*\[([\d.]+),\s*([\d.]+)\]'
    r'.*?informative:\s*([\d.]+)\s*\[([\d.]+),\s*([\d.]+)\]'
)

# Keep the old name as an alias so any external callers don't break
_METRIC_RE = _METRIC_RE_CONTEXT

def parse_results_file(filepath):
    """Parse a results file and return a data dict.

    The first non-empty line is the model name, the second is the dataset name.
    All subsequent lines are metric records.

    Returns
    -------
    dict with keys:
        'model'    : str
        'dataset'  : str
        'mode'     : 'context' or 'truth'
        'base'     : dict — context mode: {'cont_inf': (score, lo, hi), 'context': ..., 'informative': ...}
                            truth  mode: {'true_inf': (score, lo, hi), 'true': ..., 'informative': ...}
        'variants' : dict — {(k, alpha): same structure as 'base'}
        'ks'       : sorted list[int]
        'alphas'   : sorted list[float]
    """
    filepath = Path(filepath)
    model = filepath.stem
    dataset = "unknown"
    base = None
    variants = {}

    mode = None  # 'context' or 'truth', detected from first matched line
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

            m = _METRIC_RE_CONTEXT.search(stripped)
            if m:
                if mode is None:
                    mode = 'context'
                vals = tuple(float(x) for x in m.groups())
                entry = {
                    'cont_inf':    (vals[0], vals[1], vals[2]),
                    'context':     (vals[3], vals[4], vals[5]),
                    'informative': (vals[6], vals[7], vals[8]),
                }
            else:
                m = _METRIC_RE_TRUTH.search(stripped)
                if m:
                    if mode is None:
                        mode = 'truth'
                    vals = tuple(float(x) for x in m.groups())
                    entry = {
                        'true_inf':    (vals[0], vals[1], vals[2]),
                        'true':        (vals[3], vals[4], vals[5]),
                        'informative': (vals[6], vals[7], vals[8]),
                    }
                else:
                    continue

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
        'mode': mode or 'context',
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


def _plot_metric(data, metric_key, title, cmap, output_path=None, vmin=None, vmax=None):
    """Core plotting function used by the public functions.

    Produces two stacked subplots sharing a colormap normalised to [vmin, vmax]:
      - Top:    base model as a single full-width cell
      - Bottom: k × alpha matrix
    Cell text format: "score\n[lo, hi]"  (lo/hi are 2.5th / 97.5th percentiles)

    vmin/vmax default to the min/max of all scores in the data when not given.
    """
    ks = data['ks']
    alphas = data['alphas']
    nk = len(ks)
    ncols = len(alphas)

    col_labels = [f'α={a:g}' for a in alphas]
    k_labels   = [f'k={k}' for k in ks]

    score_mat, lo_mat, hi_mat = _build_score_matrix(data, metric_key)
    base_s, base_lo, base_hi  = data['base'][metric_key]

    all_scores = np.concatenate([score_mat[~np.isnan(score_mat)], [base_s]])
    if vmin is None:
        vmin = float(np.min(all_scores))
    if vmax is None:
        vmax = float(np.max(all_scores))

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    cell_w = 3.2
    cell_h = 1.5
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
                 ha='center', va='center', fontsize=12,
                 color=tc, fontweight='bold', linespacing=1.4,
                 transform=ax_base.transAxes)
    ax_base.set_yticks([0])
    ax_base.set_yticklabels(['base'], fontsize=13)
    ax_base.set_xticks([])
    ax_base.tick_params(bottom=False)

    # --- k × alpha matrix ---
    ax_main.imshow(score_mat, cmap=cmap, norm=norm, aspect='auto')
    ax_main.set_xticks(range(ncols))
    ax_main.set_xticklabels(col_labels, rotation=45, ha='right', fontsize=13)
    ax_main.set_yticks(range(nk))
    ax_main.set_yticklabels(k_labels, fontsize=13)

    for r in range(nk):
        for c in range(ncols):
            s = score_mat[r, c]
            if not np.isnan(s):
                tc = _text_color(norm(s), cmap)
                ax_main.text(c, r,
                             f'{s:.3f}\n[{lo_mat[r, c]:.3f}, {hi_mat[r, c]:.3f}]',
                             ha='center', va='center', fontsize=12,
                             color=tc, fontweight='bold', linespacing=1.4)

    # Shared colorbar as legend
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax_base, ax_main], orientation='vertical',
                        fraction=0.046, pad=0.04)
    cbar.set_label('Score [0 – 1]', fontsize=12)
    cbar.set_ticks(np.linspace(0, 1, 11))

    ax_base.set_title(
        f'{title}\n'
        f'Judge: {data["model"]}   |   Dataset: {data["dataset"]}\n'
        f'Values in brackets are [2.5th, 97.5th] percentiles',
        fontsize=13, pad=10,
    )

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f'Saved: {output_path}')
    else:
        plt.show()
    plt.close(fig)

def _plot_percent(data, metric_key, title, cmap, output_path=None, vmin=None, vmax=None):
    """Core plotting function used by the public functions (percentage variant).

    Produces two stacked subplots sharing a colormap normalised to [vmin, vmax]:
      - Top:    base model as a single full-width cell
      - Bottom: k × alpha matrix
    Cell text format: "score\n[lo, hi]"  (lo/hi are 2.5th / 97.5th percentiles)

    vmin/vmax are in [0, 1] (same as the non-percent variant) and are scaled
    internally. They default to the min/max of all scores when not given.
    """
    ks = data['ks']
    alphas = data['alphas']
    nk = len(ks)
    ncols = len(alphas)

    col_labels = [f'α={a:g}' for a in alphas]
    k_labels   = [f'k={k}' for k in ks]

    score_mat, lo_mat, hi_mat = _build_score_matrix(data, metric_key)
    base_s, base_lo, base_hi  = data['base'][metric_key]

    all_scores = np.concatenate([score_mat[~np.isnan(score_mat)], [base_s]])
    if vmin is None:
        vmin = float(np.min(all_scores))
    if vmax is None:
        vmax = float(np.max(all_scores))

    base_s *= 100
    base_lo *= 100
    base_hi *= 100

    norm = mcolors.Normalize(vmin=vmin * 100, vmax=vmax * 100)

    cell_w = 3.2
    cell_h = 1.5
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
                 ha='center', va='center', fontsize=12,
                 color=tc, fontweight='bold', linespacing=1.4,
                 transform=ax_base.transAxes)
    ax_base.set_yticks([0])
    ax_base.set_yticklabels(['base'], fontsize=13)
    ax_base.set_xticks([])
    ax_base.tick_params(bottom=False)

    # --- k × alpha matrix ---
    ax_main.imshow(score_mat * 100, cmap=cmap, norm=norm, aspect='auto')
    ax_main.set_xticks(range(ncols))
    ax_main.set_xticklabels(col_labels, rotation=45, ha='right', fontsize=13)
    ax_main.set_yticks(range(nk))
    ax_main.set_yticklabels(k_labels, fontsize=13)

    for r in range(nk):
        for c in range(ncols):
            s = score_mat[r, c]
            if not np.isnan(s):
                tc = _text_color(norm(s * 100), cmap)
                ax_main.text(c, r,
                             f'{s*100:.1f}\n[{lo_mat[r, c]*100:.1f}, {hi_mat[r, c]*100:.1f}]',
                             ha='center', va='center', fontsize=12,
                             color=tc, fontweight='bold', linespacing=1.4)

    # Shared colorbar as legend
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax_base, ax_main], orientation='vertical',
                        fraction=0.046, pad=0.04)
    cbar.set_label('Score (%)', fontsize=12)
    cbar.set_ticks(np.linspace(vmin * 100, vmax * 100, 6))

    ax_base.set_title(
        f'{title} (%)\n'
        f'Judge: {data["model"]}   |   Dataset: {data["dataset"]}\n'
        f'Values in brackets are [2.5th, 97.5th] percentiles',
        fontsize=13, pad=10,
    )

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f'Saved: {output_path}')
    else:
        plt.show()
    plt.close(fig)

# ---------------------------------------------------------------------------
# Public plotting functions
# ---------------------------------------------------------------------------

def plot_cont_informative(data, cmap='viridis', output_path=None, perc=False, vmin=None, vmax=None):
    """Plot the context × informative composite metric.

    Parameters
    ----------
    data : dict
        Parsed data dict from :func:`parse_results_file`.
    cmap : str
        Matplotlib colormap name.
    output_path : str or Path, optional
        Save the figure here instead of displaying it.
    perc : bool, optional
        Plot in Percents.
    vmin, vmax : float or None
        Colormap bounds in [0, 1]. Defaults to data min/max when None.
    """
    if perc:
        _plot_percent(data, 'cont_inf', 'Cont*Informative', cmap, output_path, vmin=vmin, vmax=vmax)
    else:
        _plot_metric(data, 'cont_inf', 'Cont*Informative', cmap, output_path, vmin=vmin, vmax=vmax)


def plot_contextual(data, cmap='viridis', output_path=None, perc=False, vmin=None, vmax=None):
    """Plot the contextual metric.

    Parameters
    ----------
    data : dict
        Parsed data dict from :func:`parse_results_file`.
    cmap : str
        Matplotlib colormap name.
    output_path : str or Path, optional
        Save the figure here instead of displaying it.
    perc : bool, optional
        Plot in Percents.
    vmin, vmax : float or None
        Colormap bounds in [0, 1]. Defaults to data min/max when None.
    """
    if perc:
        _plot_percent(data, 'context', 'Contextual', cmap, output_path, vmin=vmin, vmax=vmax)
    else:
        _plot_metric(data, 'context', 'Contextual', cmap, output_path, vmin=vmin, vmax=vmax)


def plot_informative(data, cmap='viridis', output_path=None, perc=False, vmin=None, vmax=None):
    """Plot the informative metric.

    Parameters
    ----------
    data : dict
        Parsed data dict from :func:`parse_results_file`.
    cmap : str
        Matplotlib colormap name.
    output_path : str or Path, optional
        Save the figure here instead of displaying it.
    perc : bool, optional
        Plot in Percents.
    vmin, vmax : float or None
        Colormap bounds in [0, 1]. Defaults to data min/max when None.
    """
    if perc:
        _plot_percent(data, 'informative', 'Informative', cmap, output_path, vmin=vmin, vmax=vmax)
    else:
        _plot_metric(data, 'informative', 'Informative', cmap, output_path, vmin=vmin, vmax=vmax)


def plot_true_informative(data, cmap='viridis', output_path=None, perc=False, vmin=None, vmax=None):
    """Plot the true × informative composite metric (truth results files).

    Parameters
    ----------
    data : dict
        Parsed data dict from :func:`parse_results_file`.
    cmap : str
        Matplotlib colormap name.
    output_path : str or Path, optional
        Save the figure here instead of displaying it.
    perc : bool, optional
        Plot in Percents.
    vmin, vmax : float or None
        Colormap bounds in [0, 1]. Defaults to data min/max when None.
    """
    if perc:
        _plot_percent(data, 'true_inf', 'True*Informative', cmap, output_path, vmin=vmin, vmax=vmax)
    else:
        _plot_metric(data, 'true_inf', 'True*Informative', cmap, output_path, vmin=vmin, vmax=vmax)


def plot_truthful(data, cmap='viridis', output_path=None, perc=False, vmin=None, vmax=None):
    """Plot the truthful metric (truth results files).

    Parameters
    ----------
    data : dict
        Parsed data dict from :func:`parse_results_file`.
    cmap : str
        Matplotlib colormap name.
    output_path : str or Path, optional
        Save the figure here instead of displaying it.
    perc : bool, optional
        Plot in Percents.
    vmin, vmax : float or None
        Colormap bounds in [0, 1]. Defaults to data min/max when None.
    """
    if perc:
        _plot_percent(data, 'true', 'Truthful', cmap, output_path, vmin=vmin, vmax=vmax)
    else:
        _plot_metric(data, 'true', 'Truthful', cmap, output_path, vmin=vmin, vmax=vmax)


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

    # y = x line: heads above it have higher context accuracy than truth accuracy
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, color='#888888', linestyle='-', linewidth=0.8, alpha=0.6)

    # Quadrant labels — placed at the midpoint of each quadrant
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    label_kw = dict(fontsize=7.5, color='#444444', alpha=0.85, ha='center', va='center')
    ax.text((xlim[0] + t_thr) / 2, (ylim[1] + c_thr) / 2, 'context\nspecialist', **label_kw)
    ax.text((xlim[1] + t_thr) / 2, (ylim[1] + c_thr) / 2, 'generalist',          **label_kw)
    ax.text((xlim[0] + t_thr) / 2, (ylim[0] + c_thr) / 2, 'neither',             **label_kw)
    ax.text((xlim[1] + t_thr) / 2, (ylim[0] + c_thr) / 2, 'truth\nspecialist',   **label_kw)

    ax.set_xlabel(f'Truth probe accuracy ({unit})', fontsize=11)
    ax.set_ylabel(f'Context probe accuracy ({unit})', fontsize=11)
    ax.set_title(
        'Attention head probe accuracies: Truthfulness vs. Context Grounding',
        fontsize=11,
    )

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f'Saved: {output_path}')
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Prob-experiment heatmaps
# ---------------------------------------------------------------------------

_PROB_CATEGORIES = [
    ('true',  'matching'),
    ('true',  'non_matching'),
    ('true',  'no_context'),
    ('false', 'matching'),
    ('false', 'non_matching'),
    ('false', 'no_context'),
]

_PROB_TITLES = {
    ('true',  'matching'):     'True answer — matching context',
    ('true',  'non_matching'): 'True answer — non-matching context',
    ('true',  'no_context'):   'True answer — no context',
    ('false', 'matching'):     'False answer — matching context',
    ('false', 'non_matching'): 'False answer — non-matching context',
    ('false', 'no_context'):   'False answer — no context',
}

_PROB_ENTRY_RE = re.compile(
    r'(true|false)/(matching|non_matching|no_context)=(-?[\d.]+)\s*\[(-?[\d.]+),(-?[\d.]+)\]'
)


def parse_prob_file(filepath):
    """Parse a prob-experiment output file (prob_iti_output2.txt format).

    Looks for lines of the form:
        <model_label>  (<time>s)
          true/matching=X [lo,hi]  true/non_matching=X ...  (6 categories)

    Returns
    -------
    dict with keys:
        'model'    : str  — base model name
        'base'     : dict — {category_key: (mean, lo, hi)}
        'variants' : dict — {(k, alpha): same}
        'ks'       : sorted list[int]
        'alphas'   : sorted list[float]
    """
    filepath = Path(filepath)
    base = None
    variants = {}
    model_name = None
    pending_label = None

    label_re = re.compile(r'^\S.*\([\d.]+s\)\s*$')

    with open(filepath) as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue

            if label_re.match(stripped):
                pending_label = stripped
                continue

            if 'true/matching=' in stripped and pending_label is not None:
                matches = _PROB_ENTRY_RE.findall(stripped)
                if len(matches) == 6:
                    entry = {
                        f'{cat}/{sub}': (float(mean), float(lo), float(hi))
                        for cat, sub, mean, lo, hi in matches
                    }
                    vm = re.search(r'_top_(\d+)_alpha_([\d.]+)\s+\(', pending_label)
                    if vm:
                        k = int(vm.group(1))
                        alpha = float(vm.group(2))
                        variants[(k, alpha)] = entry
                    else:
                        base = entry
                        model_name = re.sub(r'\s+\([\d.]+s\)\s*$', '', pending_label)
                    pending_label = None

    if base is None:
        raise ValueError(f"No base-model result line found in {filepath}")

    ks     = sorted({k for k, _ in variants})
    alphas = sorted({a for _, a in variants})

    return {
        'model':    model_name or filepath.stem,
        'base':     base,
        'variants': variants,
        'ks':       ks,
        'alphas':   alphas,
    }


def _build_prob_matrix(data, category_key):
    """Return score, lo, hi arrays of shape (len(ks), len(alphas))."""
    ks, alphas = data['ks'], data['alphas']
    shape = (len(ks), len(alphas))
    score = np.full(shape, np.nan)
    lo    = np.full(shape, np.nan)
    hi    = np.full(shape, np.nan)
    for ki, k in enumerate(ks):
        for ai, alpha in enumerate(alphas):
            entry = data['variants'].get((k, alpha))
            if entry and category_key in entry:
                score[ki, ai], lo[ki, ai], hi[ki, ai] = entry[category_key]
    return score, lo, hi


def _plot_prob_heatmap(data, category_key, title, cmap, output_path, vmin, vmax):
    ks     = data['ks']
    alphas = data['alphas']
    nk     = len(ks)
    ncols  = len(alphas)

    score_mat, lo_mat, hi_mat = _build_prob_matrix(data, category_key)
    base_s, base_lo, base_hi  = data['base'][category_key]

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    cell_w = 3.2
    cell_h = 1.5
    fig_w  = cell_w * ncols + 2.5
    fig_h  = cell_h * (nk + 1) + 2.5

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs  = fig.add_gridspec(2, 1, height_ratios=[1, nk], hspace=0.08)
    ax_base = fig.add_subplot(gs[0])
    ax_main = fig.add_subplot(gs[1])

    # Base row
    ax_base.imshow([[base_s]], cmap=cmap, norm=norm, aspect='auto')
    tc = _text_color(norm(base_s), cmap)
    ax_base.text(0.5, 0.5,
                 f'{base_s:.3f}\n[{base_lo:.3f}, {base_hi:.3f}]',
                 ha='center', va='center', fontsize=12,
                 color=tc, fontweight='bold', linespacing=1.4,
                 transform=ax_base.transAxes)
    ax_base.set_yticks([0])
    ax_base.set_yticklabels(['base'], fontsize=13)
    ax_base.set_xticks([])
    ax_base.tick_params(bottom=False)

    # k × alpha matrix
    ax_main.imshow(score_mat, cmap=cmap, norm=norm, aspect='auto')
    ax_main.set_xticks(range(ncols))
    ax_main.set_xticklabels([f'α={a:g}' for a in alphas], rotation=45, ha='right', fontsize=13)
    ax_main.set_yticks(range(nk))
    ax_main.set_yticklabels([f'k={k}' for k in ks], fontsize=13)

    for r in range(nk):
        for c in range(ncols):
            s = score_mat[r, c]
            if not np.isnan(s):
                tc = _text_color(norm(s), cmap)
                ax_main.text(c, r,
                             f'{s:.3f}\n[{lo_mat[r, c]:.3f}, {hi_mat[r, c]:.3f}]',
                             ha='center', va='center', fontsize=12,
                             color=tc, fontweight='bold', linespacing=1.4)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax_base, ax_main], orientation='vertical',
                        fraction=0.046, pad=0.04)
    cbar.set_label('Mean log-prob (higher = more likely)', fontsize=12)

    ax_base.set_title(
        f'{title}\nModel: {data["model"]}\n'
        f'Values in brackets are [2.5th, 97.5th] percentiles',
        fontsize=13, pad=10,
    )

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f'Saved: {output_path}')
    else:
        plt.show()
    plt.close(fig)


def plot_prob_heatmaps(data, cmap='viridis', output_dir=None, vmin=None, vmax=None):
    """Plot 6 heatmaps (one per category) from a parsed prob-experiment data dict.

    All 6 plots share the same colormap range so cells are directly comparable.

    Parameters
    ----------
    data       : dict from :func:`parse_prob_file`
    cmap       : matplotlib colormap name
    output_dir : directory to save PNGs; display interactively if None
    vmin, vmax : colormap bounds; default to global min/max across all categories
    """
    if vmin is None or vmax is None:
        all_vals = [v for entry in [data['base']] + list(data['variants'].values())
                    for v, _, _ in entry.values()]
        if vmin is None:
            vmin = min(all_vals)
        if vmax is None:
            vmax = max(all_vals)

    out_dir = Path(output_dir) if output_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    for cat, sub in _PROB_CATEGORIES:
        key   = f'{cat}/{sub}'
        title = _PROB_TITLES[(cat, sub)]
        fname = f'prob_{cat}_{sub}.png'
        out_path = str(out_dir / fname) if out_dir else None
        _plot_prob_heatmap(data, key, title, cmap, out_path, vmin, vmax)


# ---------------------------------------------------------------------------
# Probe-score experiment heatmaps  (token-grounding scores, top_k only)
# ---------------------------------------------------------------------------

_PROBE_SCORE_LABEL_RE = re.compile(
    r'^(.+?)\s+top_k=(\d+)\s+\([\d.]+s\)\s*$'
)


def parse_probe_score_file(filepath):
    """Parse a probe-score experiment output file (probe_test_results.txt format).

    Looks for lines of the form:
        <model_name> top_k=<k>  (<time>s)
          true/matching=X [lo,hi]  true/non_matching=X ...  (6 categories)

    Returns
    -------
    dict with keys:
        'model'    : str
        'variants' : dict — {k: {category_key: (mean, lo, hi)}}
        'ks'       : sorted list[int]
    """
    filepath = Path(filepath)
    model_name = None
    variants = {}
    pending_k = None

    with open(filepath) as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue

            m = _PROBE_SCORE_LABEL_RE.match(stripped)
            if m:
                if model_name is None:
                    model_name = m.group(1)
                pending_k = int(m.group(2))
                continue

            if 'true/matching=' in stripped and pending_k is not None:
                matches = _PROB_ENTRY_RE.findall(stripped)
                if len(matches) == 6:
                    variants[pending_k] = {
                        f'{cat}/{sub}': (float(mean), float(lo), float(hi))
                        for cat, sub, mean, lo, hi in matches
                    }
                    pending_k = None

    if not variants:
        raise ValueError(f"No probe score results found in {filepath}")

    return {
        'model':    model_name or filepath.stem,
        'variants': variants,
        'ks':       sorted(variants.keys()),
    }


_PROBE_SCORE_BAR_COLORS = [
    '#2ca02c',  # true/matching        — dark green
    '#98df8a',  # true/non_matching    — light green
    '#c7e9c0',  # true/no_context      — very light green
    '#d62728',  # false/matching       — dark red
    '#ff9896',  # false/non_matching   — light red
    '#ffd8d7',  # false/no_context     — very light red
]

_PROBE_SCORE_BAR_LABELS = [
    'True\nmatching',
    'True\nnon-matching',
    'True\nno context',
    'False\nmatching',
    'False\nnon-matching',
    'False\nno context',
]


def _plot_probe_score_bars(data, k, output_path, ymin, ymax):
    entry = data['variants'][k]
    keys  = [f'{cat}/{sub}' for cat, sub in _PROB_CATEGORIES]

    scores = np.array([entry[key][0] for key in keys])
    los    = np.array([entry[key][1] for key in keys])
    his    = np.array([entry[key][2] for key in keys])
    err_lo = scores - los
    err_hi = his - scores

    fig, ax = plt.subplots(figsize=(8, 4.5))

    xs = np.arange(len(keys))
    bars = ax.bar(xs, scores, color=_PROBE_SCORE_BAR_COLORS,
                  edgecolor='#444444', linewidth=0.7, width=0.6)
    ax.errorbar(xs, scores, yerr=[err_lo, err_hi],
                fmt='none', ecolor='#222222', elinewidth=1.2, capsize=4, capthick=1.2)

    for x, s in zip(xs, scores):
        ax.text(x, s + max(err_hi) * 0.15, f'{s:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xticks(xs)
    ax.set_xticklabels(_PROBE_SCORE_BAR_LABELS, fontsize=11)
    ax.set_ylabel('Mean token grounding score', fontsize=11)
    ax.set_ylim(ymin, ymax)
    ax.axhline(0.5, color='#888888', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.set_title(
        f'Probe-score results — top k={k}\n'
        f'Model: {data["model"]}\n'
        f'Error bars are [2.5th, 97.5th] percentiles',
        fontsize=12, pad=8,
    )

    # Vertical separator between true and false groups
    ax.axvline(2.5, color='#aaaaaa', linestyle=':', linewidth=1.0)

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f'Saved: {output_path}')
    else:
        plt.show()
    plt.close(fig)


def plot_probe_score_heatmaps(data, cmap='viridis', output_dir=None, vmin=None, vmax=None):
    """Plot one bar chart per top-k value, showing all 6 category scores.

    All plots share the same y-axis range so they are directly comparable.

    Parameters
    ----------
    data       : dict from :func:`parse_probe_score_file`
    cmap       : unused (kept for API consistency with the other plot functions)
    output_dir : directory to save PNGs; display interactively if None
    vmin, vmax : y-axis bounds; default to slightly below/above global min/max
    """
    all_vals = [v for entry in data['variants'].values() for v, _, _ in entry.values()]
    span = max(all_vals) - min(all_vals)
    if vmin is None:
        vmin = max(0.0, min(all_vals) - span * 0.1)
    if vmax is None:
        vmax = min(1.0, max(all_vals) + span * 0.15)

    out_dir = Path(output_dir) if output_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    for k in data['ks']:
        fname    = f'probe_score_k{k}.png'
        out_path = str(out_dir / fname) if out_dir else None
        _plot_probe_score_bars(data, k, out_path, vmin, vmax)


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
        choices=['cont_inf', 'context', 'informative', 'true_inf', 'true', 'all'],
        default='all',
        help='Which metric figure(s) to produce (default: all). '
             'Context-mode metrics: cont_inf, context, informative. '
             'Truth-mode metrics: true_inf, true, informative.',
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
        '--prob-file', default=None, metavar='FILE',
        help='Prob-experiment output file (prob_iti_output2.txt format) for 6-category heatmaps',
    )
    parser.add_argument(
        '--probe-score-file', default=None, metavar='FILE',
        help='Probe-score experiment output file (probe_test_results.txt format) for '
             '6-category token-grounding score heatmaps (top_k only, no alpha axis)',
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
    parser.add_argument(
        '--vmin', type=float, default=None, metavar='FLOAT',
        help='Colormap lower bound in [0, 1] for heatmap plots '
             '(default: minimum score in the data)',
    )
    parser.add_argument(
        '--vmax', type=float, default=None, metavar='FLOAT',
        help='Colormap upper bound in [0, 1] for heatmap plots '
             '(default: maximum score in the data)',
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    def out(name):
        return str(out_dir / name) if out_dir else None

    percentage = args.percent == 'true'

    # --- Prob-experiment heatmap mode ---
    if args.prob_file:
        prob_data = parse_prob_file(args.prob_file)
        print(f"Model:    {prob_data['model']}")
        print(f"ks:       {prob_data['ks']}")
        print(f"alphas:   {prob_data['alphas']}")
        print(f"Variants: {len(prob_data['variants'])}")
        plot_prob_heatmaps(
            prob_data,
            cmap=args.cmap,
            output_dir=args.output_dir,
            vmin=args.vmin,
            vmax=args.vmax,
        )
        return

    # --- Probe-score experiment heatmap mode ---
    if args.probe_score_file:
        ps_data = parse_probe_score_file(args.probe_score_file)
        print(f"Model:    {ps_data['model']}")
        print(f"ks:       {ps_data['ks']}")
        print(f"Variants: {len(ps_data['variants'])}")
        plot_probe_score_heatmaps(
            ps_data,
            cmap=args.cmap,
            output_dir=args.output_dir,
            vmin=args.vmin,
            vmax=args.vmax,
        )
        return

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

    is_truth = data['mode'] == 'truth'
    plot_kw = dict(cmap=args.cmap, perc=percentage, vmin=args.vmin, vmax=args.vmax)

    if args.metric == 'all':
        if is_truth:
            plot_true_informative(data, output_path=out('true_informative.png'), **plot_kw)
            plot_truthful(data, output_path=out('truthful.png'), **plot_kw)
            plot_informative(data, output_path=out('informative.png'), **plot_kw)
        else:
            plot_cont_informative(data, output_path=out('cont_informative.png'), **plot_kw)
            plot_contextual(data, output_path=out('contextual.png'), **plot_kw)
            plot_informative(data, output_path=out('informative.png'), **plot_kw)
    else:
        dispatch = {
            'cont_inf':    (plot_cont_informative, 'cont_informative.png'),
            'context':     (plot_contextual,        'contextual.png'),
            'true_inf':    (plot_true_informative,  'true_informative.png'),
            'true':        (plot_truthful,           'truthful.png'),
            'informative': (plot_informative,        'informative.png'),
        }
        fn, fname = dispatch[args.metric]
        fn(data, output_path=out(fname), **plot_kw)


if __name__ == '__main__':
    main()
