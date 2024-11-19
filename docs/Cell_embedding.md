## Evaluation of the clustering of the embed cell.

### 01 Get the cell embedding from scFM models(zero-shot)
#### scGPT
```python
from biollm.utils.utils import load_config
import scanpy as sc
import pandas as pd
import numpy as np
from biollm.base.load_scfoundation import LoadScfoundation
import os
import pickle
from biollm.base.load_scgpt import LoadScgpt


def scgpt(adata, output_dir):
    config_file = './configs/zero_shots/scgpt_cell_emb.toml'
    configs = load_config(config_file)
    # adata = sc.read_h5ad(configs.input_file)
    # adata.var_names = adata.var["feature_name"].tolist()
    # adata.var["gene_name"] = adata.var["feature_name"]
    # adata.obs["celltype_id"] = adata.obs["cell_type"].cat.codes
    # adata.obs["batch_id"] = 0
    # sc.pp.highly_variable_genes(adata, n_top_genes=3000, subset=True)
    obj = LoadScgpt(configs)
    adata = adata[:, adata.var_names.isin(obj.get_gene2idx().keys())].copy()
    # configs.max_seq_len = adata.var.shape[0] + 1
    obj = LoadScgpt(configs)
    print(obj.args)
    obj.model = obj.model.to(configs.device)
    emb = obj.get_embedding(configs.emb_type, adata=adata)
    print('embedding shape:', emb.shape)
    scg_cell_emb = pd.DataFrame(emb, index=adata.obs_names)
    cell_emb_file = os.path.join(output_dir, "scg_cell_emb.pkl")
    with open(cell_emb_file, 'wb') as file:
        pickle.dump(np.array(scg_cell_emb), file)

data_path = './liver.h5ad' # data_path
output_dir = './output'
adata = sc.read_h5ad(data_path)
scgpt(adata, output_dir)
```

#### Geneformer
```python
from biollm.utils.utils import load_config
import scanpy as sc
import pandas as pd
import numpy as np
from biollm.base.load_geneformer import LoadGeneformer
import os
import pickle


def geneformer(adata, output_dir):
    from biollm.base.load_geneformer import LoadGeneformer
    config_file = './configs/zero_shots/geneformer_cell_emb.toml'
    configs = load_config(config_file)
    obj = LoadGeneformer(configs)
    print(obj.args)
    # adata = sc.read_h5ad(configs.input_file)
    # adata.var_names = adata.var["feature_name"]
    # sc.pp.highly_variable_genes(adata, n_top_genes=3000, subset=True)
    obj.model = obj.model.to(configs.device)
    emb = obj.get_embedding(obj.args.emb_type, adata=adata)
    print('embedding shape:', emb.shape)
    gf_cell_emb = pd.DataFrame(emb, index=adata.obs_names)
    cell_emb_file = os.path.join(output_dir, "gf_cell_emb.pkl")
    with open(cell_emb_file, 'wb') as file:
        pickle.dump(np.array(gf_cell_emb), file)
data_path = './liver.h5ad' # data_path
output_dir = './output'
adata = sc.read_h5ad(data_path)
geneformer(adata, output_dir)
```

#### scFoundation
```python
from biollm.utils.utils import load_config
import scanpy as sc
import pandas as pd
import numpy as np
from biollm.base.load_scfoundation import LoadScfoundation
import os
import pickle


def scfoundation(adata, output_dir):
    config_file = './configs/zero_shots/scfoundation_cell_emb.toml'
    configs = load_config(config_file)
    obj = LoadScfoundation(configs)
    print(obj.args)
    obj.model = obj.model.to(configs.device)
    emb = obj.get_embedding(configs.emb_type, gene_ids=None, adata=adata)
    print('embedding shape:', emb.shape)
    scf_cell_emb = pd.DataFrame(emb, index=adata.obs_names)

    cell_emb_file = os.path.join(output_dir, "scf_cell_emb.pkl")
    with open(cell_emb_file, 'wb') as file:
        pickle.dump(np.array(scf_cell_emb), file)

data_path = './liver.h5ad' # data_path
output_dir = './output'
adata = sc.read_h5ad(data_path)
scfoundation(adata, output_dir)
```

#### scBERT
```python
from biollm.utils.utils import load_config
import scanpy as sc
import pandas as pd
import numpy as np
from biollm.base.load_scbert import LoadScbert
import os
import pickle


def scbert(adata, output_dir):
    config_file = './configs/zero_shots/scbert_cell_emb.toml'
    configs = load_config(config_file)
    obj = LoadScbert(configs)
    print(obj.args)
    obj.model = obj.model.to(configs.device)
    emb = obj.get_embedding(configs.emb_type, adata=adata)
    print('embedding shape:', emb.shape)
    scb_cell_emb = pd.DataFrame(emb, index=adata.obs_names)
    cell_emb_file = os.path.join(output_dir, "scbert_cell_emb.pkl")
    with open(cell_emb_file, 'wb') as file:
        pickle.dump(np.array(scb_cell_emb), file)

data_path = './liver.h5ad' # data_path
output_dir = './output'
adata = sc.read_h5ad(data_path)
scbert(adata, output_dir)
```

Note: The config directory can be found in the biollm/docs/. 

### 02 Evaluation
```python
from sklearn.metrics import silhouette_score
import os
from collections import defaultdict
import scanpy as sc
import pickle
import numpy as np
import pandas as pd

def cal_aws(cell_emb, label):
    asw = silhouette_score(cell_emb, label)
    # return asw
    asw = (asw + 1)/2
    return asw

scores = defaultdict(list)
models = ['scBert', 'scGPT', 'scFoundation', 'Geneformer']
output_dir = './output'
dataset_for_test = './liver.h5ad'

pkl_map = {'scBert': 'scbert_cell_emb.pkl', 'scGPT': 'scg_cell_emb.pkl', 'scFoundation': 'scf_cell_emb.pkl', 'Geneformer': 'gf_cell_emb.pkl'}
adata = sc.read_h5ad(dataset_for_test)
label_key = 'celltype' if 'celltype' in adata.obs.columns else 'CellType'
for model in ['scBert', 'scGPT', 'scFoundation', 'Geneformer']:
    with open(os.path.join(output_dir, pkl_map[model]), 'rb') as f: 
        adata.obsm['model'] = pickle.load(f)
        if np.isnan(adata.obsm['model']).sum() > 0:
            print('cell emb has nan value. ', model)
            adata.obsm['model'][np.isnan(adata.obsm['model'])] = 0
    
        aws = cal_aws(adata.obsm['model'], adata.obs[label_key].cat.codes.values)
        print(model, aws)
        scores['data'].append(dataset_for_test.split('/')[-1].split('.h5ad')[0])
        scores['model'].append(model)
        scores['aws'].append(aws)
df = pd.DataFrame(scores)
df.to_csv('./zero-shot_cellemb_aws.csv', index=False)

```

### 03 Visualization
```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Here are the model performances on four datasets: 'Zheng68K', 'blood', 'kidney', and 'liver'
df = pd.read_csv('./zero-shot_cellemb_aws.csv')

# Custom function to convert radians and add labels
def get_label_rotation(angle, offset):
    rotation = np.rad2deg(angle + offset)
    if angle <= np.pi:
        alignment = "right"
        rotation += 180
    else:
        alignment = "left"
    return rotation, alignment

def add_labels(angles, values, labels, offset, ax):
    padding = 0.2
    for angle, value, label in zip(angles, values, labels):
        rotation, alignment = get_label_rotation(angle, offset)
        ax.text(
            x=angle, y=value + padding, s=label, 
            ha=alignment, va="center", rotation=rotation, 
            rotation_mode="anchor"
        )

GROUP = df["dataset"].values
GROUPS_SIZE = [len(i[1]) for i in df.groupby("dataset")]
COLORS = ['#F4D72D', '#1F9A4C', '#1D3D8F', '#DF2723'] * 4  # Define the color corresponding to the model

VALUES = df["value"].values
LABELS = df["model"].values

OFFSET = np.pi / 2
PAD = 2
ANGLES_N = len(VALUES) + PAD * len(np.unique(GROUP))
ANGLES = np.linspace(0, 2 * np.pi, num=ANGLES_N, endpoint=False)
WIDTH = (2 * np.pi) / len(ANGLES)

# Get index
offset = 0
IDXS = []
for size in GROUPS_SIZE:
    IDXS += list(range(offset + PAD, offset + size + PAD))
    offset += size + PAD

# Initialize polar coordinate graph
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={"projection": "polar"})
ax.set_theta_offset(OFFSET)
ax.set_ylim(-0.7, 1)

ax.set_frame_on(False)
ax.xaxis.grid(False)
ax.yaxis.grid(False)
ax.set_xticks([])
ax.set_yticks([])

# Add bar chart
ax.bar(ANGLES[IDXS], VALUES, width=WIDTH, color=COLORS, edgecolor="white", linewidth=2)

# Add labels
add_labels(ANGLES[IDXS], VALUES, LABELS, OFFSET, ax)

# Add group tags
offset = 0
for group, size in zip(["Zheng68K", "blood", "kidney", "liver"], GROUPS_SIZE):
    x1 = np.linspace(ANGLES[offset + PAD], ANGLES[offset + size + PAD - 1], num=50)
    ax.plot(x1, [-0.1] * 50, color="#333333")
    
    ax.text(
        np.mean(x1), -0.2, group, color="#333333", fontsize=14, 
        fontweight="bold", ha="center", va="center"
    )
    
    x2 = np.linspace(ANGLES[offset], ANGLES[offset + PAD - 1], num=50)
    ax.plot(x2, [0] * 50, color="#bebebe", lw=0.8)
    ax.plot(x2, [0.2] * 50, color="#bebebe", lw=0.8)
    ax.plot(x2, [0.4] * 50, color="#bebebe", lw=0.8)
    ax.plot(x2, [0.6] * 50, color="#bebebe", lw=0.8)
    
    offset += size + PAD

plt.tight_layout()
plt.rcParams['pdf.fonttype'] = 42
plt.savefig('./cell_emb_aws.pdf', bbox_inches='tight')
plt.show()

```