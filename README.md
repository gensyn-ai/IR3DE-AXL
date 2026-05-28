# IR3DE-AXL

# Get started

1) Download and build axl within this repository:

```
git clone https://github.com/gensyn-ai/axl.git
cd axl
go build -o node ./cmd/node/
```

2) Generate any desired number of nodes:

```
./add_local_node.sh [NUM-PEERS]
```

3) Install the conda environment:

```
conda env create -f requirements.yml
```

4) Download pre-trained M2D2 Llama 115M experts

```
python download_models.py
```

5) Run the simulation:

```
python run_simulation.py --peer-id [PEER_ID]
```