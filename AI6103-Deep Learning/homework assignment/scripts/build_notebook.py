"""Generate a thin Colab entry point; all scientific logic lives in scripts/."""
from pathlib import Path
import nbformat

ROOT = Path(__file__).resolve().parents[1]


def build():
    md = nbformat.v4.new_markdown_cell
    code = nbformat.v4.new_code_cell
    cells = [md("# AI6103 Colab entry point\n\nRunning all cells with the defaults performs preparation only. Full training and the final test each require explicit opt-in. Upload (or clone) the whole repository to Google Drive, not just this Notebook. Put the dataset in `data/` or `data.zip` at the repository root. Checkpoints stay on Drive; dataset images may be copied to local Colab disk for speed."),
             code("from google.colab import drive\ndrive.mount('/content/drive')"),
             code("from pathlib import Path\nimport sys, subprocess, shutil, zipfile\n\n# Edit this location to the repository folder on YOUR Drive.\nPROJECT = Path('/content/drive/MyDrive/AI6103-github')\nassert (PROJECT / 'scripts/run_all.py').exists(), 'Upload the repository to Drive first'\n# Results/checkpoints are durable, separate from Colab session storage.\nOUTPUT = PROJECT / 'results'\nFIGURES = PROJECT / 'figures'\nREPORT = PROJECT / 'report'\n# Use a local dataset copy for performance; do not modify the supplied splits.\nDATA = Path('/content/ai6103_data/data')\nif not DATA.exists():\n    if (PROJECT / 'data').exists():\n        shutil.copytree(PROJECT / 'data', DATA)\n    else:\n        archive = PROJECT / 'data.zip'\n        assert archive.exists(), 'Place data.zip or data/ in PROJECT'\n        destination = DATA.parent.resolve()\n        destination.mkdir(parents=True, exist_ok=True)\n        with zipfile.ZipFile(archive) as z:\n            for member in z.infolist():\n                assert (destination / member.filename).resolve().is_relative_to(destination), 'Unsafe archive path'\n            z.extractall(destination)\nassert (DATA / 'train_labels.csv').exists()"),
             code("# Keep Colab's matched torch/torchvision pair; install the other project tools.\nsubprocess.run([sys.executable, '-m', 'pip', 'install', '-r', str(PROJECT / 'requirements.txt')], check=True)\n# Restart the runtime after installation if existing imported binary packages changed.\n# Before official runs, record the working environment and do not change it mid-sweep.\ndef cli(*arguments, check=True):\n    return subprocess.run([sys.executable, str(PROJECT / 'scripts/run_all.py'),\n        '--data-root', str(DATA), '--output-root', str(OUTPUT),\n        '--figure-root', str(FIGURES), *arguments], check=check)"),
             code("cli('prepare-stats')\ncli('validate')\npreflight = cli('preflight', check=False)\nprint('Preflight exit code:', preflight.returncode, '(2 means GPU unavailable or insufficient disk)')\ncli('plot', '--beta-only')"),
             md("## Explicit future training\nSet `RUN_SECTION` to one of 3, 4, 5, 6 only when GPU resources are ready. Run sections in that order; dependencies and completion are verified. `--resume` restores partial experiments and skips only fully verified compatible completions. Set `EXPERIMENT` to an exact ID for one condition. If the runtime or package versions change, restore a matching environment before resuming. A stale lock left by a hard disconnect must be removed only after confirming the original process has ended."),
             code("RUN_SECTION = None  # safe default: no training\nEXPERIMENT = None\nif RUN_SECTION is not None:\n    assert preflight.returncode == 0, 'GPU preflight must pass'\n    arguments = ['train', '--section', str(RUN_SECTION), '--resume']\n    if EXPERIMENT is not None:\n        arguments += ['--experiment', EXPERIMENT]\n    cli(*arguments)"),
             code("SELECT_SECTION = None  # 3 for LR; 6 for the downstream setup\nif SELECT_SECTION is not None:\n    cli('select', '--section', str(SELECT_SECTION))"),
             code("GENERATE_EXPERIMENT_FIGURES = False\nFREEZE_COMPLETE_EXPERIMENTS = False\nif GENERATE_EXPERIMENT_FIGURES:\n    cli('plot')\nif FREEZE_COMPLETE_EXPERIMENTS:\n    cli('freeze')"),
             md("## Final hold-out evaluation: separate future operation\nRequires all eight completed experiments and an immutable frozen manifest. The durable started marker prevents automatic repeated evaluation after a disconnect. Leave this disabled until all experiments are complete."),
             code("AUTHORIZE_FINAL_TEST = False\nif AUTHORIZE_FINAL_TEST:\n    cli('evaluate-final', '--authorize-final-test')"),
             code("# Generate an honest draft; final generation additionally needs real results,\n# final test, checked figures, an author and authored evidence-based discussion.\nsubprocess.run([sys.executable, str(PROJECT / 'scripts/fill_latex.py'),\n    '--mode', 'draft', '--data-root', str(DATA), '--output-root', str(OUTPUT),\n    '--figure-root', str(FIGURES), '--report-root', str(REPORT)], check=True)"),
             code("EXPORT_RESULTS = False\nEXPORT_LARGE_CHECKPOINTS = False\nif EXPORT_RESULTS:\n    arguments = ['export'] + (['--checkpoints'] if EXPORT_LARGE_CHECKPOINTS else [])\n    cli(*arguments)\n    print('Small report_results.zip and optional large checkpoints.zip are under', OUTPUT / 'exports')\n# No dataset archive and no automatic downloads are triggered.")]
    notebook = nbformat.v4.new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}})
    nbformat.validate(notebook)
    nbformat.write(notebook, ROOT / "notebooks" / "AI6103_HW_Colab.ipynb")


if __name__ == "__main__":
    build()
