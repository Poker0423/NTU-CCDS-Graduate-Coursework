"""Draft rendering and strictly gated final result bindings; never edits empirical prose."""
import argparse
import math
from pathlib import Path
import re
try:
    from .config import ROOT, SECTION3_LRS
    from .artifacts import read_json, write_json, file_hash, fingerprint
    from .protocol import load_context, completed_bundle, select_section3
except ImportError:
    from config import ROOT, SECTION3_LRS
    from artifacts import read_json, write_json, file_hash, fingerprint
    from protocol import load_context, completed_bundle, select_section3


def escape(text):
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(replacements.get(c, c) for c in text)


def row(label, record=None):
    values = [f"{record['history'][m][-1]:.4f}" for m in ("train_loss", "train_acc", "val_loss", "val_acc")] if record else ["Pending"] * 4
    return escape(label) + " & " + " & ".join(values) + r" \\" + "\n"


def generate(mode, data_root, output_root, figure_root, report_root, analysis=None, author=None):
    context, stats, _ = load_context(data_root, output_root)
    macros = {"ReportStatus": "PRE-EXPERIMENT DRAFT: official experiments and final test pending.",
              "BestLR": "Pending", "BaselineName": "Pending", "TestAccuracy": "Pending",
              "TestStatus": "This draft does not report hold-out performance; final generation verifies the dedicated evaluation record.",
              "SectionThreeSummary": "All four final-metric comparisons remain pending.",
              "SectionThreeRows": "".join(row(f"{lr:g}") for lr in SECTION3_LRS),
              "SectionFourRows": row("Constant") + row("Cosine"),
              "SectionFiveRows": row("5e-4") + row("1e-4"),
              "SectionSixRows": row("No Mixup") + row("Mixup")}
    for channel, values in (("Mean", stats["mean"]), ("Std", stats["std"])):
        for suffix, value in zip("RGB", values):
            macros[channel + suffix] = f"{value:.7f}"
    for number in ("Three", "Four", "Five", "Six"):
        macros["Discussion" + number] = "Empirical discussion pending completed, verified experiments."
    figure_macro = r"\newcommand{\ExperimentFigure}[1]{\fbox{\parbox{0.92\columnwidth}{\centering Experimental figure pending verified GPU runs.}}}"
    if mode == "final":
        configs, records, baseline = completed_bundle(context, output_root)
        manifest = read_json(output_root / "frozen_manifest.json")
        digest = manifest.pop("manifest_fingerprint")
        record_hashes = {c.experiment_id: file_hash(output_root / c.experiment_id / "record.json") for c in configs}
        if fingerprint(manifest) != digest or manifest["context"] != context or manifest["records"] != record_hashes:
            raise ValueError("Frozen manifest does not match completed experiments")
        test = read_json(output_root / "final_evaluation" / "test_result.json")
        if test["status"] != "COMPLETE" or test["manifest_fingerprint"] != digest or test["count"] != 1534 or test["checkpoint_sha256"] != records[-1]["checkpoint_sha256"] or test["experiment_id"] != "sec6_mixup":
            raise ValueError("Final test is missing or inconsistent")
        if not math.isfinite(test["test_acc"]) or not 0 <= test["test_acc"] <= 1 or not math.isfinite(test["test_loss"]):
            raise ValueError("Invalid final test metrics")
        figures = read_json(figure_root / "experiment_figures.provenance.json")
        if figures["records"] != record_hashes:
            raise ValueError("Stale figures")
        for name, digest_figure in figures["figures"].items():
            if file_hash(figure_root / name) != digest_figure:
                raise ValueError("Figure checksum mismatch")
        required_figures = {"sec3_lr.png", "sec4_schedule.png", "sec5_wd.png", "sec6_mixup.png"}
        if set(figures["figures"]) != required_figures:
            raise ValueError("Missing experimental figures")
        beta_provenance = read_json(figure_root / "beta_pdf.provenance.json")
        if file_hash(figure_root / "beta_pdf.png") != beta_provenance["sha256"]:
            raise ValueError("Analytical figure checksum mismatch")
        if not analysis or not author or author.strip().lower() in ("anonymous", "pending"):
            raise ValueError("Final report requires --analysis and --author")
        discussion = read_json(analysis)
        for number, key in zip(("Three", "Four", "Five", "Six"), ("section3", "section4", "section5", "section6")):
            text = discussion.get(key, "")
            if len(text.strip()) < 40 or re.search(r"pending|TODO|placeholder", text, re.I):
                raise ValueError(f"Authored evidence-based discussion missing: {key}")
            macros["Discussion" + number] = escape(text)
        selected = select_section3(records[:3], configs[:3])
        winners = []
        for metric in ("train_loss", "train_acc", "val_loss", "val_acc"):
            best = (min if "loss" in metric else max)(range(3), key=lambda i: records[i]["history"][metric][-1])
            winners.append(escape(metric) + f": LR {configs[best].lr:g} ({records[best]['history'][metric][-1]:.4f})")
        macros.update({"ReportStatus": "Final experimental report; measurements bound to verified records.",
                       "BestLR": f"{selected.lr:g}", "BaselineName": escape(baseline.experiment_id),
                       "TestAccuracy": f"{100 * test['test_acc']:.2f}" + r"\%",
                       "TestStatus": "Evaluated once after configuration freezing.",
                       "SectionThreeSummary": "; ".join(winners) + ".",
                       "SectionThreeRows": "".join(row(f"{c.lr:g}", r) for c, r in zip(configs[:3], records[:3])),
                       "SectionFourRows": row("Constant", records[3]) + row("Cosine", records[4]),
                       "SectionFiveRows": row("5e-4", records[5]) + row("1e-4", records[6])})
        idx = [c.experiment_id for c in configs].index(baseline.experiment_id)
        macros["SectionSixRows"] = row("No Mixup", records[idx]) + row("Mixup", records[-1])
        figure_macro = r"\newcommand{\ExperimentFigure}[1]{\includegraphics[width=\columnwidth]{#1}}"
    report_root.mkdir(parents=True, exist_ok=True)
    binding = report_root / ("bindings.tex" if mode == "draft" else "final_bindings.tex")
    binding.write_text("% Generated only from verified provenance.\n" + "\n".join("\\newcommand{\\" + name + "}{" + value + "}" for name, value in macros.items()) + "\n" + figure_macro + "\n", encoding="utf-8")
    source = (ROOT / "report" / "main.tex").read_text(encoding="utf-8")
    if report_root != ROOT / "report":
        import shutil
        for name in ("refs.bib", "aaai24.sty", "aaai24.bst"):
            shutil.copy2(ROOT / "report" / name, report_root / name)
    source = source.replace("../figures/", Path(__import__('os').path.relpath(figure_root, report_root)).as_posix() + "/")
    if mode == "final":
        source = source.replace(r"\input{bindings.tex}", r"\input{final_bindings.tex}")
        source = source.replace(r"\author{Anonymous}", r"\author{" + escape(author) + "}")
    target = report_root / ("main_draft.tex" if mode == "draft" else "main_filled.tex")
    target.write_text(source, encoding="utf-8")
    write_json(report_root / (mode + "_provenance.json"), {"mode": mode, "context": context, "source_sha256": file_hash(target), "bindings_sha256": file_hash(binding)})
    print(target)
    return target


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("draft", "final"), default="draft")
    p.add_argument("--data-root", type=Path, default=ROOT / "data")
    p.add_argument("--output-root", type=Path, default=ROOT / "results")
    p.add_argument("--figure-root", type=Path, default=ROOT / "figures")
    p.add_argument("--report-root", type=Path, default=ROOT / "report")
    p.add_argument("--analysis", type=Path)
    p.add_argument("--author")
    a = p.parse_args()
    try:
        generate(a.mode, a.data_root.resolve(), a.output_root.resolve(), a.figure_root.resolve(), a.report_root.resolve(), a.analysis, a.author)
    except (ValueError, FileNotFoundError, KeyError) as e:
        p.exit(1, f"REPORT_REFUSED: {e}\n")


if __name__ == "__main__":
    main()
