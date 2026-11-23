"""Build the Kaggle notebooks from plain cell lists (stdlib only).

Run: python kaggle/build_notebooks.py [commit]
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


# CPU notebook: attach the atml-pa4-task3-runner output and repackage it (the v1 run's final copy step had a
# wrong adapter path, so its zip was never written; all stages had exited 0).
TASK3_COLLECT = [
    r"""
import glob, os, shutil, subprocess
src = [os.path.dirname(os.path.dirname(p)) for p in glob.glob('/kaggle/input/**/results/task3', recursive=True)]
assert src, 'attach the atml-pa4-task3-runner output as an input'
src = src[0]; print('source', src)
dst = '/kaggle/working/pkg'
shutil.copytree(f'{src}/results/task3', f'{dst}/results/task3')
shutil.copytree(f'{src}/logs/task3', f'{dst}/logs/task3')
out3 = f'{src}/task3_out/task3'
for d in sorted(glob.glob(f'{out3}/adapters/*/')):
    n = os.path.basename(d.rstrip('/')); os.makedirs(f'{dst}/results/task3/adapters/{n}', exist_ok=True)
    for f in ['summary.json', 'updates.json', 'presentations.jsonl']:
        if os.path.exists(d + f): shutil.copy(d + f, f'{dst}/results/task3/adapters/{n}/')
if os.path.exists(f'{out3}/teacher_collection_meta.json'):
    shutil.copy(f'{out3}/teacher_collection_meta.json', f'{dst}/results/task3/')
print(subprocess.run('cd /kaggle/working/pkg && zip -qr /kaggle/working/task3_results.zip results logs && '
                     'ls -la /kaggle/working/task3_results.zip && du -ah {} | sort -h | tail -40'.format(out3),
                     shell=True, capture_output=True, text=True).stdout)
""",
]


# Commit each Kaggle run actually executed (printed as "code commit:" in its notebook output). Rebuilding with no
# argument reproduces those runs exactly; `python kaggle/build_notebooks.py <commit>` pins every runner to <commit>
# (e.g. the current main, which includes the later fixes listed in the README).
RUN_COMMITS = {
    "task1": "827fae1",      # Task 1 run 1 (all P/Q conditions)
    "task1q3f": "ac8f4f5",   # Task 1 follow-up: projection-level Fisher Q3
    "task2": "b0398de",      # Task 2 run 1 (A, B1/B2, C)
    "task2b3": "74c108d",    # Task 2 B3 intervention + SDPA dispatch + key statistics
    "task2sel": "142b043",   # Task 2 Part C aligned selections
    "task3": "b806dfa",      # Task 3 run 1 (repackaged by task3_collect.ipynb)
}


def main():
    import sys
    override = sys.argv[1] if len(sys.argv) > 1 else None
    outputs = [("pa4_data_setup.ipynb", make_notebook(DATA_SETUP)),
               ("task3_collect.ipynb", make_notebook(TASK3_COLLECT, gpu=False))]
    for task, commit in RUN_COMMITS.items():
        outputs.append((f"{task}_runner.ipynb", make_notebook(runner(task, override or commit))))
    for name, nb in outputs:
        out = HERE / name
        out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
        print("wrote", out)


if __name__ == "__main__":
    main()
