from torch.utils.data import Dataset 

import os 
import torch 
import numpy as np 

class SMDDataset(Dataset): 

    def __init__(self, root: str, window_size: int = 100): 
        self.root = root 
        self.window_size = window_size
        self.records: list = []
        self.machine_count: int = len(os.listdir(self.root))

        machines: list[str] = os.listdir(self.root)
        for machine_id, machine_filename in enumerate(machines):
            machine_path: str = os.path.join(self.root, machine_filename)

            # read the file 
            machine_data: torch.Tensor = torch.from_numpy(
                np.loadtxt(machine_path, delimiter=",", dtype=np.float32), 
            )

            # split the data into windows
            for i in range(0, len(machine_data) - self.window_size + 1, self.window_size):
                window_data: torch.Tensor = machine_data[i:i+self.window_size]
                self.records.append((machine_id, window_data))

    def __len__(self) -> int:
        return len(self.records)
    
    def __getitem__(self, idx: int) -> tuple[int, torch.Tensor]:
        return self.records[idx]