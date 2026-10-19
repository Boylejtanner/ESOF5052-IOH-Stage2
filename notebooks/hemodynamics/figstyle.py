"""Figure labelling helpers: every saved figure must have an x label and a y label (with units) on every visible axes, and its text starts with a capital letter.

``audit_figure`` raises when an axes is unlabeled. Colour-bar axes need one label; helper axes (twin axes, hidden axes) are skipped when marked with ``mark_helper``.
``capitalize_figure`` gives axis labels, titles, legends and category tick labels a capital first letter and replaces ASCII shorthand by proper symbols.
"""
from __future__ import annotations
# Keep figure labels, units, and panel titles consistent across analyses.

import re

# text replacements applied to every label, title, legend entry and category tick label (order matters)
REPLACEMENTS = [
    ("->", "\u2192"), (">=", "\u2265"), ("<=", "\u2264"), ("mmHg*min", "mmHg\u00b7min"), ("delta MAP", "\u0394MAP"), ("log10", "log\u2081\u2080"),
    ("at theta 0.75", "with \u2265 75% valid samples"), ("G0-lr", "MAP only, logistic"), ("G4-hgb", "All features, boosting"), ("G0-hgb", "MAP only, boosting"), ("G4-lr", "All features, logistic"),
    (" (E3x)", " (external test)"),
]
_NAMES = [(re.compile(r"\bVITALDB\b|\bvitaldb\b"), "VitalDB"), (re.compile(r"\bmover\b"), "MOVER")]
_LEAD = re.compile(r"^(\s*(?:\([a-z]\)\s+)?)([a-z])")


_RUNG = re.compile(r"selected\s+at\s+rung\s+R1\s+\(>=\s+30\s+ready\s+minutes\s+at\s+theta\s+0\.75\)")


def fix_text(s):
    """Replacements, dataset names, and a capital first letter (after an optional panel prefix such as ``(a) ``). Units written as the first word (``(mmHg ...)``) are left alone."""
    if not s or s.startswith("$"):
        return s
    s = _RUNG.sub("selected (\u2265 30 min of analyzable windows,\n\u2265 75% valid samples each)", s)
    for code, text in (("G0-lr", "MAP only, logistic"), ("G0-hgb", "MAP only, boosting"), ("G4-lr", "All features, logistic"), ("G4-hgb", "All features, boosting")):
        s = re.sub(r"(?<=\S) " + code + r"$", ": " + text, s)          # "pooled -> pooled test G0-lr" -> "...test: MAP only, logistic"
    for a, b in REPLACEMENTS:
        s = s.replace(a, b)
    for rx, b in _NAMES:
        s = rx.sub(b, s)
    m = _LEAD.match(s)
    if m:
        s = s[:m.end(1)] + m.group(2).upper() + s[m.end():]
    return s


def _fix_ticklabels(axis, set_ticks, set_labels):
    labels = [t.get_text() for t in axis.get_ticklabels()]
    new = [fix_text(t) if t and t[0:1].isalpha() or t.startswith(" ") else t for t in labels]
    if new != labels:
        set_ticks(list(axis.get_ticklocs()))
        set_labels(new)


def capitalize_figure(fig, panel_letters=False):
    """Capitalize all labels, titles, legend text and category tick labels of ``fig`` in place; with ``panel_letters`` prefix the titles of multi-panel figures with (a), (b), ..."""
    # Apply consistent capitalization to the labels, legends, and panel titles.
    fig.canvas.draw()
    for ax in fig.axes:
        ax.set_xlabel(fix_text(ax.get_xlabel()))
        ax.set_ylabel(fix_text(ax.get_ylabel()))
        if ax.get_title():
            ax.set_title(fix_text(ax.get_title()))
        leg = ax.get_legend()
        if leg is not None:
            if leg.get_title().get_text():
                leg.set_title(fix_text(leg.get_title().get_text()))
            for t in leg.get_texts():
                t.set_text(fix_text(t.get_text()))
        for t in ax.texts:
            t.set_text(fix_text(t.get_text()))
        if ax.get_label() != "<colorbar>":
            _fix_ticklabels(ax.xaxis, ax.set_xticks, ax.set_xticklabels)
            _fix_ticklabels(ax.yaxis, ax.set_yticks, ax.set_yticklabels)
    sup = getattr(fig, "_suptitle", None)
    if sup is not None:
        sup.set_text(fix_text(sup.get_text()))
    for t in fig.texts:
        t.set_text(fix_text(t.get_text()).replace("->", "\u2192"))
    if panel_letters:
        titled = [ax for ax in fig.axes if ax.get_label() != "<colorbar>" and ax.get_title() and ax.get_visible()]
        if len(titled) >= 2:
            titled.sort(key=lambda a: (-round(a.get_position().y1, 2), a.get_position().x0))
            for k, ax in enumerate(titled):
                if not re.match(r"^\([a-z]\)", ax.get_title()):
                    ax.set_title(f"({chr(97 + k)}) {ax.get_title()}")
    fig.canvas.draw()
    return fig


def finish_axes(ax, xlabel, ylabel, title=None):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    return ax


def label_colorbar(cb, text):
    cb.set_label(text)
    return cb


def mark_helper(ax):
    """Exclude a helper axes (for example an invisible spacer) from the audit."""
    ax._figstyle_skip = True
    return ax


def audit_axes(fig):
    """Return a list of problems (empty when every visible axes is labelled)."""
    problems = []
    # Process the selected records in order and retain their individual results.
    for i, ax in enumerate(fig.axes):
        if getattr(ax, "_figstyle_skip", False) or not ax.get_visible():
            continue
        title = (ax.get_title() or f"axes #{i}")[:50]
        if ax.get_label() == "<colorbar>":
            if not (ax.get_ylabel().strip() or ax.get_xlabel().strip()):
                problems.append(f"colorbar without a label ({title})")
            continue
        if not ax.get_xlabel().strip():
            problems.append(f"missing x label ({title})")
        if not ax.get_ylabel().strip():
            problems.append(f"missing y label ({title})")
    return problems


def audit_figure(fig, name=""):
    problems = audit_axes(fig)
    if problems:
        raise ValueError(f"figure {name or '?'} has unlabeled axes: " + "; ".join(problems))
    return True
