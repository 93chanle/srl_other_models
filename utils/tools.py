import torch
import sys
import types
from importlib import import_module

class StandardScaler():
    def __init__(self):
        self.mean = 0.
        self.std = 1.
    
    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std

    def inverse_transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        if data.shape[-1] != mean.shape[-1]:
            mean = mean[-1:]
            std = std[-1:]
        return (data * std) + mean

class MinMaxScaler():
    def __init__(self):
        pass
    
    def fit(self, data):
        self.range = max(data) - min(data)
        self.min = min(data)

    def transform(self, data):
        range = torch.from_numpy(self.range).type_as(data).to(data.device) if torch.is_tensor(data) else self.range
        min = torch.from_numpy(self.min).type_as(data).to(data.device) if torch.is_tensor(data) else self.min
        return (data - min) / range

    def inverse_transform(self, data):
        range = torch.from_numpy(self.range).type_as(data).to(data.device) if torch.is_tensor(data) else self.range
        min = torch.from_numpy(self.min).type_as(data).to(data.device) if torch.is_tensor(data) else self.min
        return (data * range) + min
    
def add_line_breaks_to_args_string(args, max_len=80):
    """ Break long string of args Namespaces with line breaks
    for plotting results.

    Args:
    
        args (str): args list pron args_parser
        max_len (int): max length before adding line break
    """
    
    max_len = 80
    rows = ['']

    # Convert to string
    args = str(args)

    # Filter content within brackets
    args = args[args.find("(")+1:args.find(")")].split(',')

    # Add args to a row until max_len reached, then create new row
    for arg in args:
        if len(rows[-1]) < max_len:
            rows[-1] = rows[-1] + (arg) + ',' 
        else:
            rows.append('')
            rows[-1] = rows[-1] + (arg) + ',' 
    
    return '\n'.join(rows)

def autoimport(module_name: str) -> None:
    """Deletes an already imported module during interactive (Jupyter Notebook).
    The updated module can later be imported again. Useful when woring on a
    py script and want to test it in Jupyter Notebook.

    Args:
        module_name (str): name of sub(module). Can be name.of.sub.modules
    """
    try:
        del sys.modules[module_name]
    except KeyError:
        pass
    
    import_module(module_name)