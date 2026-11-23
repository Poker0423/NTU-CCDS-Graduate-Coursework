"""Local multi-file LaTeX compilation, diagnostics and body-page accounting."""
import argparse
from pathlib import Path
import re
import shutil
import subprocess
import sys
try:
    from .config import ROOT
    from .artifacts import read_json, write_json, file_hash
except ImportError:
    from config import ROOT
    from artifacts import read_json, write_json, file_hash


def compile_report(mode, root):
    root = Path(root)
    tex = root / ("main_draft.tex" if mode == "draft" else "main_filled.tex")
    provenance = read_json(root / (mode + "_provenance.json"))
    binding = root / ("bindings.tex" if mode == "draft" else "final_bindings.tex")
    if provenance["source_sha256"] != file_hash(tex) or provenance["bindings_sha256"] != file_hash(binding):
        raise ValueError("Report source/bindings changed; regenerate the verified target")
    diagnostics = root / (mode + "_compile_status.json")
    if not shutil.which("pdflatex") or not shutil.which("bibtex"):
        write_json(diagnostics, {"status": "TEX_NOT_AVAILABLE", "mode": mode, "compiled": False,
                                "body_pages": None, "final_compliance": "PENDING_GENUINE_RESULTS_AND_COMPILATION"})
        print("TEX_NOT_AVAILABLE: source preserved; install a local TeX distribution or compile the project in Overleaf.")
        return 2
    commands = [["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex.name],
                ["bibtex", tex.stem],
                ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex.name],
                ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex.name]]
    for command in commands:
        result = subprocess.run(command, cwd=root, text=True, capture_output=True, timeout=60)
        if result.returncode:
            write_json(diagnostics, {"status": "COMPILATION_FAILED", "command": command, "output": result.stdout[-12000:] + result.stderr[-2000:]})
            raise RuntimeError("LaTeX failed; see compile status and .log")
    log = tex.with_suffix(".log").read_text(errors="replace")
    count = re.search(r"REPORT_BODY_PAGES:\s*(\d+)", log)
    issues = [line for line in log.splitlines() if "Overfull" in line or "undefined" in line.lower()]
    pages = int(count.group(1)) if count else None
    ok = pages is not None and pages <= 6 and not issues
    write_json(diagnostics, {"status": "COMPILED" if ok else "LAYOUT_REVIEW_REQUIRED", "mode": mode,
                            "compiled": True, "body_pages": pages, "issues": issues,
                            "pdf_sha256": file_hash(tex.with_suffix(".pdf")), "visual_review": "REQUIRED",
                            "final_compliance": "PENDING_VISUAL_REVIEW" if mode == "final" else "DRAFT_ONLY"})
    print(tex.with_suffix(".pdf"), f"body_pages={pages}", f"issues={len(issues)}")
    return 0 if ok else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("draft", "final"), default="draft")
    p.add_argument("--report-root", type=Path, default=ROOT / "report")
    args = p.parse_args()
    try:
        return compile_report(args.mode, args.report_root.resolve())
    except (ValueError, RuntimeError, FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"COMPILATION_REFUSED: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
