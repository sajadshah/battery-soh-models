import os
import numpy as np
from scipy.io import loadmat
from tqdm import tqdm
import pandas as pd
from torch.utils.data import Dataset
import torch
from decimal import Decimal
import re
import glob
from multiprocessing import Pool
import time

cycle_groups = {
    "RW": ["charge (random walk)", "rest (random walk)", "discharge (random walk)", "rest post random walk discharge", "charge (after random walk discharge)"],
    "PCC": ["pulsed charge (charge)", "pulsed charge (rest)", "rest post pulsed load or charge"],
    "PCD": ["pulsed load (discharge)", "pulsed load (rest)", "rest post pulsed load or charge", "rest post pulsed load"],
    "LCD": ["low current discharge at 0.04A", "rest prior low current discharge", "rest post low current discharge"],
    "LCC": ["low current charge", "rest post low current charge"],
    "RC" :["reference charge", "rest post reference charge"],
    "RD": ["reference discharge", "rest post reference discharge", "rest prior reference discharge"],
    "PD": ["rest post reference power discharge", "reference power discharge"]
}

reference_tests = [
    ["reference charge"],
    ["reference discharge"],
    ["low current discharge at 0.04A"],
    ["pulsed load (rest)", "pulsed load (discharge)"],
    ["pulsed charge (rest)", "pulsed charge (charge)"]
]

def_train_cells = 'RW9,RW10,RW11,RW12,RW1,RW7,RW8,RW25,RW26,RW27,RW28,RW20,RW21,RW22,RW23,RW24,RW13,RW14,RW15,RW16'
def_valid_cells = 'RW4,RW5,RW6'
def_all_cells = f"{def_train_cells},{def_valid_cells}"

def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]
    

def load_nasa_cell_data_mat2df(matfile, normalize_time=False, normalize_delta_time=0.1):
    d = loadmat(matfile)
    data = []
    data_normal = []
    total_cycles = len(d["data"]["step"][0][0][0])
    for cycle in range(total_cycles):
        x = d["data"]["step"][0][0][0][cycle]
        comment = x["comment"][0]
        xtimes = x["time"].reshape(-1)
        xreltimes = x["relativeTime"].reshape(-1)
        xvols = x["voltage"].reshape(-1)
        xcurs = x["current"].reshape(-1)
        xtemps = x["temperature"].reshape(-1)
        y = list(zip(xtimes, xreltimes, xvols, xcurs, xtemps, [comment]*len(xtimes), [cycle]*len(xtimes)))
        data.extend(y)
        if normalize_delta_time is not None:
            normalize_delta_time = np.float64(normalize_delta_time)
            t0 = xreltimes[0]
            tn = xreltimes[-1]
            tn = tn - tn % normalize_delta_time
            n = int((tn-t0) / normalize_delta_time) + 1
            normal_reltimes = np.linspace(t0, tn, n)
            y_normal = list(zip(
                np.interp(normal_reltimes, xreltimes, xtimes),
                normal_reltimes, 
                np.interp(normal_reltimes, xreltimes, xvols), 
                np.interp(normal_reltimes, xreltimes, xcurs), 
                np.interp(normal_reltimes, xreltimes, xtemps), 
                [comment]*len(normal_reltimes), 
                [cycle]*len(normal_reltimes)
            ))
            data_normal.extend(y_normal)
     
    df = pd.DataFrame(data, columns=["time", "relative_time", "voltage", "current", "temperature", "comment", "cycle_number"])
    df['idx'] = df.index
    df_normal = pd.DataFrame(data_normal, columns=["time", "relative_time", "voltage", "current", "temperature", "comment", "cycle_number"])
    df_normal['idx'] = df_normal.index
    
    return df, df_normal

def mat2df(matfile, normalized_delta_time=None):
    df, df_normal = load_nasa_cell_data_mat2df(matfile, normalize_delta_time=normalized_delta_time)
    
    df["delta_t"] = df["time"].diff()
    df["delta_delta_t"] = df["delta_t"].diff()
    
    df_normal["delta_t"] = df_normal["time"].diff()
    df_normal["delta_delta_t"] = df_normal["delta_t"].diff()
    
    return df, df_normal

def create_capacity_file(bdf, outfile):
    all_ref_tests = [j for rts in reference_tests for j in rts]

    bdfc = bdf[bdf["comment"].isin(all_ref_tests)].copy()
    bdfc["delta_capacity"] = bdfc["current"]*bdfc["delta_t"]/3600
    
    bdf_gr = bdfc.groupby(by=["cycle_number"])
    bdf_cycs_beg_idxs = bdf_gr["idx"].min()
    bdf_cycs_end_idxs = bdf_gr["idx"].max()
    with tqdm(total=len(bdf_cycs_beg_idxs)) as pbar:
        with open(outfile, mode="w") as f:
            print("beg_idx,end_idx,beg_time,end_time,current_capacity,type_of_capacity_calc", file=f)
            i = 0
            while i < len(bdf_cycs_beg_idxs):
                beg_idx = bdf_cycs_beg_idxs.iloc[i]
                end_idx = bdf_cycs_end_idxs.iloc[i]
                # print("i,beg,end (1): ", i, beg_idx, end_idx)
                rt_idx = -1
                for j, rts in enumerate(reference_tests):
                    if bdfc.loc[beg_idx]["comment"] in rts:
                        rt_idx = j
                        break
                # print(rt_idx)
                if rt_idx != -1:
                    i+=1
                    while i < len(bdf_cycs_beg_idxs) and bdfc.loc[bdf_cycs_beg_idxs.iloc[i]]["comment"] in reference_tests[rt_idx]:
                        i+=1
                    i -= 1
                    end_idx = bdf_cycs_end_idxs.iloc[i]
                    beg_time = bdfc["time"].loc[beg_idx]                    
                    end_time = bdfc["time"].loc[end_idx]
                    # print("i,beg,end (2): ", i,beg_idx, end_idx)
                    curr_capacity = bdfc.loc[beg_idx:end_idx]["delta_capacity"]
                    # print("len cycle :", len(curr_capacity))
                    curr_capacity = curr_capacity.sum()
                    # print("sum cycle :", curr_capacity)
                    curr_capacity = abs(curr_capacity)
                    print(f"{beg_idx},{end_idx},{beg_time},{end_time},{curr_capacity},{rt_idx}", file=f)
                i += 1
                pbar.update(i - pbar.n)


def create_grouped_cycle_number_file(bdf, outfile):
    bdf_gr = bdf.groupby(by=["cycle_number"])
    bdf_cycs_beg_idxs = bdf_gr["idx"].min()
    bdf_cycs_end_idxs = bdf_gr["idx"].max()
    with tqdm(total=len(bdf_cycs_beg_idxs)) as pbar:
        with open(outfile, mode="w") as f:
            print("beg_idx,end_idx,grouped_cycle_number,grouped_cycle_type", file=f)
            gc_num = 0
            i = 0
            while i < len(bdf_cycs_beg_idxs):
                comment = bdf.loc[bdf_cycs_beg_idxs.iloc[i]]["comment"]
                cg_idx = -1
                cg_type = "UNK"
                for j, cg in enumerate(cycle_groups.keys()):
                    if comment in cycle_groups[cg]:
                        cg_idx = j
                        cg_type = cg
                        break
                cg_beg_idx = bdf_cycs_beg_idxs.iloc[i]
                cg_end_idx = bdf_cycs_end_idxs.iloc[i]
                # print("i,beg,end (1): ", i,cg_beg_idx, cg_end_idx)
                # print("cg_idx: ", cg_idx)
                if cg_idx == -1:
                    pass
                else:
                    i += 1
                    while  i < len(bdf_cycs_beg_idxs) and bdf.loc[bdf_cycs_beg_idxs.iloc[i]]["comment"] in cycle_groups[cg_type]: 
                        i += 1
                    i -= 1
                    cg_end_idx = bdf_cycs_end_idxs.iloc[i]
                # print("i,beg,end (2): ", i,cg_beg_idx, cg_end_idx)
                print(f"{cg_beg_idx},{cg_end_idx},{gc_num},{cg_type}", file=f)
                gc_num += 1
                i += 1
            pbar.update(i - pbar.n)

def read_all_batnames(data_dir):
    all_csv_files = [f for f in glob.glob(os.path.join(data_dir, "*.csv")) if re.search(r'RW[0-9]+_normal\.csv', f)]
    all_batnames = [re.search(r'RW[0-9]+', os.path.basename(f))[0] for f in all_csv_files]
    return all_batnames

def load_pcap_normalizer_params(dsoh_train_dir):
    pfile = os.path.join(dsoh_train_dir, "dcap_normalizer_params.csv")
    x = pd.read_csv(pfile)
    return x.iloc[0]["mean"], x.iloc[0]["std"]

def save_dcap_normalizer_params(out_dir, train_dataset):
    pfile = os.path.join(out_dir, "dcap_normalizer_params.csv")
    with open(pfile, mode="w") as f:
        print("mean,std", file=f)
        print(f"{train_dataset.dcap_mean_normalizer},{train_dataset.dcap_std_normalizer}", file=f)

class NasaCellDataV1():
    def __init__(self, data_dir, batname, window_length, window_overlap):
        self.batname = batname
        self.data_dir = data_dir
        self.window_length = window_length
        self.window_overlap = window_overlap

        self.csvfile = os.path.join(data_dir, f"{batname}_normal.csv")
        self.pklfile = os.path.join(data_dir, f"{batname}_normal.pkl")
        self.capfile = os.path.join(data_dir, f"{batname}_normal_cap.csv")
        self.gcfile = os.path.join(data_dir, f"{batname}_normal_gc.csv")
        self.wsfile = os.path.join(data_dir, f"{batname}_normal_wsidx.csv")

        self.create_wsidx_file()
        start = time.time()
        # if os.path.exists(self.pklfile):
        #     self.bdf = pd.read_pickle(self.pklfile)
        # else:
        #     self.bdf = pd.read_csv(self.csvfile, usecols=["idx", "current", "voltage", "temperature"])
        self.bdf = pd.read_csv(self.csvfile, usecols=["idx", "current", "voltage", "temperature"])
        print("bdf read took: ", time.time() - start)
        
        start = time.time()
        self.bdf_cap = pd.read_csv(self.capfile)
        self.bdf_cap = self.bdf_cap[(self.bdf_cap["type_of_capacity_calc"] == 1) & (self.bdf_cap["current_capacity"] > 0)]
        print("bdf_cap read took: ", time.time() - start)
        
        start = time.time()
        self.bdf_gc = pd.read_csv(self.gcfile)
        print("bdf_gc read took: ", time.time() - start)
        
        start = time.time()
        self.bdf_ws = pd.read_csv(self.wsfile)
        print("bdf_ws read took: ", time.time() - start)

        self.bdf["cap"] = np.interp(self.bdf["idx"].to_numpy(), self.bdf_cap["end_idx"], self.bdf_cap["current_capacity"])
        
    def __len__(self):
        return len(self.bdf_ws)

    def create_wsidx_file(self):
        gc_bdf = pd.read_csv(self.gcfile)
        with open(self.wsfile, mode="w") as f:
            print("idx", file=f)
            for i,row in gc_bdf.iterrows():
                beg_idx = row["beg_idx"]
                end_idx = row["end_idx"]
                clen = end_idx - beg_idx + 1
                if clen >= self.window_length:
                    end_wsidx = (clen-self.window_length)
                    if self.window_overlap > 0:
                        end_wsidx -= (clen%self.window_overlap) 
                else:
                    end_wsidx = -1
                wsidxs = beg_idx + np.arange(0, end_wsidx, (self.window_length-self.window_overlap))
                for wsidx in wsidxs:
                    print(wsidx, file=f)
        print(f"wsfile created for {self.batname}")
    
    def get_window(self, start_idx):
        start = time.time()
        #d = pd.read_csv(self.csvfile, header=None, skiprows=start_idx, chunksize=self.window_length,
        #                engine="c",
        #                names=["time","relative_time","voltage","current","temperature","comment","cycle_number","idx","delta_t","delta_delta_t"])
        # print("read_csv took: ", time.time() - start)
        # d = next(d)
        # start = time.time()
        # print("next took: ", time.time() - start)
        d = self.bdf.iloc[start_idx:start_idx+self.window_length]
        d = d[["current", "voltage", "temperature"]].to_numpy()
        # print("to_numpy took: ", time.time() - start)
        
        return d
    
    def get_window_dcap(self, start_idx):
        end_idx = start_idx + self.window_length
        dcap = self.bdf.loc[end_idx]["cap"] - self.bdf.loc[start_idx]["cap"]
        return dcap

class NasaDatasetAutoEncoderV1(Dataset):
    def __init__(self, data_dir, device, window_length, window_overlap, batnames=None, nj=8):
        self.data_dir = data_dir
        self.device = device
        self.window_length = window_length
        self.window_overlap = window_overlap
        all_batnames = read_all_batnames(data_dir)
        if batnames is None:
            self.batnames = all_batnames
        else:
            self.batnames = batnames
        self.nj = nj
        
        # with Pool(nj) as p:
        #     self.all_cells = (p.map(self.create_cell, self.batnames))
        self.all_cells = [self.create_cell(batname) for batname in self.batnames if len(batname) > 0]
        print(f"all data windows: {len(self)}")
    
    def create_cell(self, batname):
        cell = NasaCellDataV1(self.data_dir, batname, self.window_length, self.window_overlap)
        print(f"{batname} with {len(cell)} data windows loaded")
        return cell   
        
    def __len__(self):
        sz = 0
        for cell in self.all_cells:
            sz += len(cell)
        return sz

    def idx_to_cell_i(self, idx):
        inter_sum = 0
        for i,cell in enumerate(self.all_cells):
            inter_sum += len(cell)
            if idx < inter_sum:
                return i,(idx-(inter_sum-len(cell)))
        
        raise ValueError(f'couldn not find cell for idx: {idx} (inter_sum: f{inter_sum})')
    def getitem_info(self, idx):
        """
            finds cellname, window start and end idx of a train window
        """
        cell_i, in_cell_ws_i = self.idx_to_cell_i(idx)
        cell = self.all_cells[cell_i]
        wsidx = cell.bdf_ws.iloc[in_cell_ws_i]["idx"]

        return cell.batname, wsidx, wsidx+self.window_length
    def get_wsidx(self, idx):
        cell_i, in_cell_ws_i = self.idx_to_cell_i(idx)
        cell = self.all_cells[cell_i]
        wsidx = cell.bdf_ws.iloc[in_cell_ws_i]["idx"]

        return wsidx

    def __getitem__(self, idx):
        cell_i, in_cell_ws_i = self.idx_to_cell_i(idx)
        cell = self.all_cells[cell_i]
        
        wsidx = self.get_wsidx(idx)
           
        input_d = cell.get_window(wsidx)
        input_d = torch.from_numpy(input_d).to(self.device)

        return input_d.T, input_d.T

class NasaDatasetAutoEncoderV1WithIdx(NasaDatasetAutoEncoderV1):
    def __init__(self, *args, **kwargs):
        super(NasaDatasetAutoEncoderV1WithIdx, self).__init__(*args, **kwargs)

    def __getitem__(self, idx):
        wsidx = self.get_wsidx(idx)

        d,l = super().__getitem__(idx)

        return wsidx, d, l

class NasaDatasetAEdSoHV1(NasaDatasetAutoEncoderV1):
    def __init__(self, data_dir, device, window_length, window_overlap, batnames=None, nj=8, normalize_dcap=False, dcap_mean_normalizer=None, dcap_std_normalizer=None):
        super(NasaDatasetAEdSoHV1, self).__init__(data_dir, device, window_length, window_overlap, batnames, nj)
        self.normalize_dcap = normalize_dcap
        if normalize_dcap: 
            if (dcap_mean_normalizer is None or dcap_std_normalizer is None):
                self.dcap_mean_normalizer, self.dcap_std_normalizer = self.get_dcap_mean_std_all_cells()
            else:
                self.dcap_mean_normalizer, self.dcap_std_normalizer = dcap_mean_normalizer, dcap_std_normalizer


    def get_all_dcaps(self):
        all_dcaps = []
        for cell in self.all_cells:
            try:
                start_idxs = cell.bdf_ws["idx"]
                end_idxs = start_idxs + self.window_length
                dcap = cell.bdf.loc[end_idxs]["cap"].to_numpy() - cell.bdf.loc[start_idxs]["cap"].to_numpy()
                all_dcaps.append(dcap)
            except Exception as e:
                print(f"cell {cell.batname} is broken...")
                print(e)

        all_dcaps = np.concatenate(all_dcaps)
        return all_dcaps
    
    def get_dcap_mean_std_all_cells(self):
        all_dcaps = self.get_all_dcaps()    
        # all_dcap = reject_outliers(all_dcap, 2)
        return all_dcaps.mean(), all_dcaps.std()

    def __getitem__(self, idx):
        cell_i, in_cell_ws_i = self.idx_to_cell_i(idx)
        cell = self.all_cells[cell_i]
        wsidx = cell.bdf_ws.iloc[in_cell_ws_i]["idx"]

        input_d = cell.get_window(wsidx)
        input_d = torch.from_numpy(input_d).to(self.device)

        dcap = cell.get_window_dcap(wsidx)
        if self.normalize_dcap:
            dcap = (dcap - self.dcap_mean_normalizer) / self.dcap_std_normalizer
        dcap = torch.tensor(dcap).to(self.device).reshape(1)
        return input_d.T, dcap

class NasaDatasetAEdSoHV1WithIdx(NasaDatasetAEdSoHV1):
    def __init__(self, *args, **kwargs):
        super(NasaDatasetAEdSoHV1WithIdx, self).__init__(*args, **kwargs)

    def __getitem__(self, idx):
        wsidx = self.get_wsidx(idx)

        d,l = super().__getitem__(idx)

        return wsidx, d, l