# IR3DE-AXL

# Installation

1) Download and build axl within this repository:

```
git clone https://github.com/gensyn-ai/axl.git
cd axl
go build -o node ./cmd/node/
cd ..
```

2) Generate any desired number of nodes:

```
./scripts/add_local_node.sh [NUM-PEERS]
```

3) Install the Python dependencies (pick one):

### Option A: conda

```
conda env create -f requirements.yml
conda activate ir3deaxl
```

### Option B: venv

Requires Python 3.10 available on your PATH as `python3.10` (install it via your OS
package manager, [python.org](https://www.python.org/downloads/), or `pyenv` if it isn't
already) — the pinned package versions below were resolved and tested against it, and some
of them ship platform-specific binary wheels that aren't guaranteed to exist for other
Python versions.

```
python3.10 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Both options install the exact same, verified set of package versions (`requirements.txt`
is the single source of truth — `requirements.yml`'s pip section just references it).

**GPU note:** `torch` is listed unpinned to a specific CUDA build in both files, so `pip`
picks whatever wheel matches your platform automatically — on Linux this includes CUDA
support out of the box. After installing, verify GPU acceleration is actually being used:

```
python -c "import torch; print(torch.cuda.is_available())"
```

If that prints `False` on a machine that does have a GPU, reinstall `torch` following the
selector at [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/)
for your CUDA version before reinstalling the rest of `requirements.txt`.

4) Run the simulation:

```
python run.py --peer-id [PEER_ID]
```
