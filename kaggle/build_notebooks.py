"""Build the Kaggle notebooks from plain cell lists (stdlib only).

Run: python kaggle/build_notebooks.py
"""
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent


def make_notebook(cells, gpu=True):
    notebook_cells = []
    for source in cells:
        source = source.strip("\n") + "\n"
        notebook_cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": source.splitlines(keepends=True),
        })
    metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "kaggle": {
            "accelerator": "nvidiaTeslaT4" if gpu else "none",
            "isInternetEnabled": True,
            "isGpuEnabled": gpu,
        },
    }
    return {"cells": notebook_cells, "metadata": metadata, "nbformat": 4, "nbformat_minor": 5}


DATA_SETUP = [
    r"""
# PA4 data preparation (Kaggle, GPU T4, Internet ON).
# Produces /kaggle/working/pa4_data with the locked public data, prefix banks and the revision lock.
!nvidia-smi
!pip install -q "transformers==4.57.1" "huggingface_hub==0.36.2" "datasets>=3,<5"
""",
    r"""
import subprocess, json, sys, os, platform
os.chdir('/kaggle/working')
if not os.path.exists('atml-pa4-datasets'):
    subprocess.run(['git', 'clone', '-q', 'https://github.com/AbDu11aHHH/atml-pa4-datasets.git'], check=True)
commit = subprocess.run(['git', '-C', 'atml-pa4-datasets', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
print('course data repo commit:', commit)
""",
    r"""
%cd /kaggle/working/atml-pa4-datasets
!python scripts/verify_checksums.py
!python scripts/validate_assets.py
!python -m unittest discover -s tests -v
""",
    r"""
# Public data: C4 calibration/dev/validation, WikiText-2, HellaSwag, GSM8K (CPU work, C4 streaming takes a while)
!python -u scripts/setup_data.py --public
""",
    r"""
# The course script passes generator= to model.generate, which transformers 4.57.1 rejects
# ("model_kwargs are not used by the model: ['generator']"). Patch: seed the global torch RNG with the
# same per-prompt seed immediately before generate (same seed values, documented deviation).
src = open('scripts/setup_data.py').read()
old_gen = "gen=torch.Generator(device='cuda').manual_seed("
assert old_gen in src and "generator=gen," in src
src = src.replace(old_gen, "gen=torch.manual_seed(").replace("generator=gen,", "")
open('scripts/setup_data.py', 'w').write(src)
!grep -n "manual_seed" scripts/setup_data.py
""",
    r"""
# Teacher + original-draft prefix banks (GPU)
!python -u scripts/setup_data.py --prefix-banks
""",
    r"""
!python scripts/check_setup.py
!python scripts/inspect_data.py
""",
    r"""
import shutil, torch, transformers, huggingface_hub, datasets
dst = '/kaggle/working/pa4_data'
if os.path.exists(dst):
    shutil.rmtree(dst)
shutil.copytree('/kaggle/working/atml-pa4-datasets/data', dst)
env = dict(
    course_data_commit=commit,
    python=sys.version,
    platform=platform.platform(),
    torch=torch.__version__,
    cuda=torch.version.cuda,
    device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    transformers=transformers.__version__,
    huggingface_hub=huggingface_hub.__version__,
    datasets=datasets.__version__,
)
json.dump(env, open(f'{dst}/environment.json', 'w'), indent=2)
print(json.dumps(env, indent=2))
!du -sh /kaggle/working/pa4_data; find /kaggle/working/pa4_data -type f | sort
""",
]


def runner(task, commit):
    return [
        rf"""
# PA4 {task} runner (Kaggle GPU T4 x2, Internet ON). Input: the atml-pa4-data-setup notebook output.
!nvidia-smi --query-gpu=name,memory.total --format=csv
!pip install -q "transformers==4.57.1" "huggingface_hub==0.36.2" "datasets>=3,<5"
""",
        rf"""
import glob, os, subprocess
os.chdir('/kaggle/working')
if not os.path.exists('atml-pa4'):
    subprocess.run(['git', 'clone', '-q', 'https://github.com/sayim665/atml-pa4.git'], check=True)
subprocess.run(['git', '-C', 'atml-pa4', 'checkout', '-q', '{commit}'], check=True)
print('code commit:', subprocess.run(['git', '-C', 'atml-pa4', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout)
data = [p for p in glob.glob('/kaggle/input/**/pa4_data', recursive=True) if os.path.isdir(p)]
assert data, 'attach the atml-pa4-data-setup output as an input'
os.environ['PA4_DATA'] = data[0]
print('PA4_DATA =', data[0]); print(open(os.path.join(data[0], 'local_source_revisions.json')).read())
""",
        rf"""
!bash /kaggle/working/atml-pa4/kaggle/{task}_run.sh
""",
    ]


def main():
    import sys
    commit = sys.argv[1] if len(sys.argv) > 1 else "main"
    for name, cells in [("pa4_data_setup.ipynb", DATA_SETUP), ("task1_runner.ipynb", runner("task1", commit)),
                        ("task2_runner.ipynb", runner("task2", commit)),
                        ("task3_runner.ipynb", runner("task3", commit))]:
        out = HERE / name
        out.write_text(json.dumps(make_notebook(cells), indent=1), encoding="utf-8")
        print("wrote", out)


if __name__ == "__main__":
    main()
