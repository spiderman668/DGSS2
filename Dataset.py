import pandas as pd
from torch.utils.data import Dataset
from utils import *
from torch_geometric.data import Batch

#DRUG1_ID_COLUMN_NAME = "Drug1"
#DRUG2_ID_COLUMN_NAME = "Drug2"
#CELL_LINE_COLUMN_NAME = "Cell_Line_ID"
#TARGET_COLUMN_NAME = 'target'


DRUG1_ID_COLUMN_NAME = "drugA_canonical_smi"
DRUG2_ID_COLUMN_NAME = "drugB_canonical_smi"
CELL_LINE_COLUMN_NAME = "cell_line"
TARGET_COLUMN_NAME = 'label'

# graph
class DrugCombDataset(Dataset):
    def __init__(self, drugcomb_path, cell_path):
        self.drugcomb = pd.read_feather(drugcomb_path)
        self.cell_line = pd.read_feather(cell_path).set_index('cell_line_name')
        # self.cell_line = pd.read_csv(cell_path).set_index('Cell_Line_Name')

    def __len__(self):
        return self.drugcomb.shape[0]

    def __getitem__(self, index):
        sample = self.drugcomb.iloc[index]
        # get drug moleculer smiles
        drug1 = sample[DRUG1_ID_COLUMN_NAME]
        drug2 = sample[DRUG2_ID_COLUMN_NAME]
        # get drug moleculer graph
        drug1_graph = smile2graph(drug1)
        drug2_graph = smile2graph(drug2)

        # # get drug moleculer seq
        # drug1_seq, mask1 = smiles2number(drug1)
        # drug2_seq, mask2 = smiles2number(drug2)
        # # cat
        # drug1_graph.drug_seq = drug1_seq
        # drug1_graph.seq_mask = mask1
        # drug2_graph.drug_seq = drug2_seq
        # drug2_graph.seq_mask = mask2

        # get cell line
        cell_line_name = sample[CELL_LINE_COLUMN_NAME]
        cell_line_embedding = self.cell_line.loc[cell_line_name].values.flatten()
        cell_line_embedding = torch.tensor(cell_line_embedding, dtype=torch.float32)
        # get target
        target = torch.tensor(sample[TARGET_COLUMN_NAME], dtype=torch.float32)
        return (drug1_graph, drug2_graph, cell_line_embedding, target)


class DrugCombDataset2(Dataset):
    def __init__(self, drugcomb_path, cell_path):
        self.drugcomb = pd.read_csv(drugcomb_path)
        self.cell_line = pd.read_csv(cell_path).set_index('Cell_Line_Name')

    def __len__(self):
        return self.drugcomb.shape[0]

    def __getitem__(self, index):
        sample = self.drugcomb.iloc[index]
        # get drug moleculer smiles
        drug1 = sample['drug1']
        drug2 = sample['drug2']
        # get drug moleculer graph
        drug1_graph = smile2graph(drug1)
        drug2_graph = smile2graph(drug2)

        # get cell line
        cell_line_name = sample['cell']
        cell_line_embedding = self.cell_line.loc[cell_line_name].values.flatten()
        cell_line_embedding = torch.tensor(cell_line_embedding, dtype=torch.float32)
        # get target
        target = torch.tensor(sample['label'], dtype=torch.long)
        return (drug1_graph, drug2_graph, cell_line_embedding, target)


def collate(data):
    d1_list, d2_list, cell_list, label_list = [], [], [], []
    for d in data:
        graph1, graph2, cell, label = d[0], d[1], d[2], d[3]
        graph1.num_edge = graph1.num_edges
        graph2.num_edge = graph2.num_edges

        # graph1.cell = cell

        d1_list.append(graph1)
        d2_list.append(graph2)
        label_list.append(label)
        cell_list.append(cell)

    return Batch.from_data_list(d1_list), Batch.from_data_list(d2_list), torch.stack(cell_list), torch.tensor(
        label_list)
