import torch

def from_tuple_to_tensor(tuple_of_np):
    tensor = torch.zeros((len(tuple_of_np), tuple_of_np[0].shape[0]))
    for i, x in enumerate(tuple_of_np):
        tensor[i] = torch.FloatTensor(x)
    return tensor