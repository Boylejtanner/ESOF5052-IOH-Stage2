import re
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hemodynamics import eda_metrics as E
from hemodynamics import figstyle

ROOT = Path(__file__).resolve().parents[2]


class TestAudit(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_labelled_figure_passes(self):
        fig, ax = plt.subplots()
        figstyle.finish_axes(ax, "time (s)", "pressure (mmHg)", "t")
        self.assertTrue(figstyle.audit_figure(fig, "ok"))

    def test_missing_labels_are_rejected(self):
        fig, axes = plt.subplots(1, 2)
        figstyle.finish_axes(axes[0], "x (s)", "y (mmHg)")
        axes[1].set_xlabel("only x")
        with self.assertRaises(ValueError) as cm:
            figstyle.audit_figure(fig, "bad")
        self.assertIn("missing y label", str(cm.exception))

    def test_colorbar_needs_a_label_and_helpers_can_be_skipped(self):
        fig, ax = plt.subplots()
        im = ax.imshow([[0, 1], [1, 0]])
        figstyle.finish_axes(ax, "x (samples)", "y (beats)")
        cb = fig.colorbar(im, ax=ax)
        with self.assertRaises(ValueError):
            figstyle.audit_figure(fig, "cb")
        figstyle.label_colorbar(cb, "z-score")
        self.assertTrue(figstyle.audit_figure(fig, "cb"))
        spacer = fig.add_axes([0, 0, 0.1, 0.1])
        figstyle.mark_helper(spacer)
        self.assertTrue(figstyle.audit_figure(fig, "helper"))

    def test_eda_save_function_refuses_unlabelled_figures(self):
        with tempfile.TemporaryDirectory() as tmp:
            fig, ax = plt.subplots()
            ax.plot([0, 1])
            with self.assertRaises(ValueError):
                E._figure(Path(tmp), "fig_unlabelled", fig)
            self.assertFalse((Path(tmp) / "figures" / "eda" / "fig_unlabelled.png").exists())


class TestCapitalization(unittest.TestCase):
    def test_labels_titles_legends_and_categories_get_capitals_and_symbols(self):
        fig, ax = plt.subplots(1, 2)
        ax[0].bar(["train", "validation"], [1, 2], label="mover")
        ax[0].set_xlabel("within-dataset split")
        ax[0].set_ylabel("event prevalence (% of windows)")
        ax[0].set_title("prevalence per split")
        ax[0].legend(title="dataset")
        ax[1].plot([0, 1], label="pooled -> pooled test G0-lr")
        ax[1].set_xlabel("(mmHg - 80) / 40 (ART)")                    # a unit-first label must keep its lowercase unit
        ax[1].set_ylabel("delta MAP (mmHg*min)")
        ax[1].set_title("roc for vitaldb")
        ax[1].legend()
        fig.suptitle("baseline ladder >= G0")
        figstyle.capitalize_figure(fig, panel_letters=True)
        self.assertEqual(ax[0].get_xlabel(), "Within-dataset split")
        self.assertEqual(ax[0].get_ylabel(), "Event prevalence (% of windows)")
        self.assertEqual(ax[0].get_title(), "(a) Prevalence per split")
        self.assertEqual(ax[1].get_title(), "(b) Roc for VitalDB")
        self.assertEqual(ax[0].get_legend().get_title().get_text(), "Dataset")
        self.assertEqual(ax[0].get_legend().get_texts()[0].get_text(), "MOVER")
        self.assertEqual([t.get_text() for t in ax[0].get_xticklabels()], ["Train", "Validation"])
        self.assertEqual(ax[1].get_xlabel(), "(mmHg - 80) / 40 (ART)")
        self.assertEqual(ax[1].get_ylabel(), "\u0394MAP (mmHg\u00b7min)")
        self.assertEqual(ax[1].get_legend().get_texts()[0].get_text(), "Pooled \u2192 pooled test: MAP only, logistic")
        self.assertEqual(fig._suptitle.get_text(), "Baseline ladder \u2265 G0")
        plt.close(fig)

    def test_internal_selection_wording_is_rewritten_even_when_the_label_was_wrapped(self):
        self.assertEqual(figstyle.fix_text("selected at rung R1 (>= 30 ready minutes at\ntheta 0.75)"),
                         "Selected (\u2265 30 min of analyzable windows,\n\u2265 75% valid samples each)")

    def test_single_panel_figures_get_no_panel_letter(self):
        fig, ax = plt.subplots()
        ax.set_title("only panel")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        figstyle.capitalize_figure(fig, panel_letters=True)
        self.assertEqual(ax.get_title(), "Only panel")
        plt.close(fig)


class TestRenderFromCsvs(unittest.TestCase):
    def test_every_eda_figure_is_rendered_and_labelled_from_the_saved_csvs_only(self):
        src = ROOT / "results" / "eda"
        if not (src / "table_baseline_ladder_all.csv").exists():
            self.skipTest("run eda_metrics.build_all first")
        lowercase = []
        real_audit = figstyle.audit_figure

        def audit_and_check_capitals(fig, name=""):
            for ax in fig.axes:
                if ax.get_label() == "<colorbar>":
                    continue
                for kind, text in (("x label", ax.get_xlabel()), ("y label", ax.get_ylabel()), ("title", ax.get_title())):
                    if re.sub(r"^\([a-z]\)\s+", "", text)[:1].islower():
                        lowercase.append(f"{name}: {kind} {text!r}")
            return real_audit(fig, name)
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(figstyle, "audit_figure", side_effect=audit_and_check_capitals):
            dst = Path(tmp) / "results" / "eda"
            dst.mkdir(parents=True)
            for p in src.glob("*.csv"):
                shutil.copy(p, dst / p.name)
            written = E.render_figures(Path(tmp))                      # raises if any axes is unlabelled (audit inside _figure)
            self.assertEqual(len(written), len(E.FIGURE_BUILDERS))
            for rel in written:
                self.assertTrue((Path(tmp) / rel).exists(), rel)
            plt.close("all")
        self.assertEqual(lowercase, [], "labels or titles that start with a lowercase letter")


if __name__ == "__main__":
    unittest.main()
