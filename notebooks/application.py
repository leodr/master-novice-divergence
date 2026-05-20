
#TODO: 1) implement truncation of a given result after disentangling
#TODO: 2) Use 1) on small model, test results

from huggingface_hub import hf_hub_download
import torch
import torch.nn as nn

from datasets import load_dataset

cifar10 = load_dataset("cifar10")
label_map = {0: 0, 2: 1}
class_names = ['airplane', 'bird']
cifar2_train = [(example['img'], label_map[example['label']]) for example in cifar10['train'] if example['label'] in [0, 2]]
cifar2_val = [(example['img'], label_map[example['label']]) for example in cifar10['test'] if example['label'] in [0, 2]]

import torch
from torchvision.transforms import v2

val_tfms = v2.Compose([
    v2.Resize(24),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.4915, 0.4823, 0.4468], std=[0.2470, 0.2435, 0.2616])
])

import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

class MLPForCIFAR2(nn.Module, PyTorchModelHubMixin):
    """Multi-layer perceptron (MLP) for classifying 'airplane' vs. 'bird' in the CIFAR-2 dataset (a subset of CIFAR-10)."""

    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1728, 1024), # Hidden layer.
            nn.Tanh(),
            nn.Linear(1024, 512), # Hidden layer.
            nn.Tanh(),
            nn.Linear(512, 128), # Hidden layer.
            nn.Tanh(),
            nn.Linear(128, 2) # Output layer.
        )

    def forward(self, x):
        return self.mlp(x)
    
#Class for easily constructing MLP generated from disentanglement
class FixedMLP(nn.Module):
    def __init__(self, weights, biases):
        super().__init__()

        assert len(weights) == len(biases)

        layers = []
        for W, b in zip(weights, biases):
            out_f, in_f = W.shape

            layer = nn.Linear(in_f, out_f)
            layer.weight.data.copy_(W)
            layer.bias.data.copy_(b)

            layers.append(layer)

        self.layers = nn.ModuleList(layers)
        self.act = nn.Tanh()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            #% 2 condition is to make sure no activation between "compressed" layers
            if i < len(self.layers) - 1 and i % 2 == 0:
                x = self.act(x)
        return x

import torch.nn.functional as F

#TODO: run train and validation results, save into .txt
from torch.utils.data import DataLoader

class CIFAR2Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx]
        img = val_tfms(img).reshape(-1)
        return img, label

def accuracy_measure(model, dataset, batch_size):
    loader=DataLoader(CIFAR2Dataset(dataset), batch_size=batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss()

    model.eval()

    total_loss = 0.0
    total_samples = 0
    correct = 0

    with torch.no_grad():
        for batch, labels in loader:
            logits = model(batch)
            loss = criterion(logits, labels)
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()

    avg_loss = total_loss / total_samples
    accuracy = correct / total_samples

    print("CE loss:", avg_loss)
    print("Accuracy:", accuracy)

from tensor_disentanglement_code.disentanglement_base import disentangle_matrix_via_reshape, _apply_A, _apply_A_inv
from tensor_disentanglement_code.classes.disentanglement import Matrix2DDisentanglementResult

def choose_k(singular_values: torch.Tensor, threshold : float) -> int:
    energies = singular_values ** 2
    cumulative_energy = torch.cumsum(energies, dim=0)
    total_energy = cumulative_energy[-1]

    explained_variance_ratio = cumulative_energy / total_energy

    k = torch.searchsorted(explained_variance_ratio, threshold).item() + 1
    return k


def test_sv(singular_values: torch.Tensor):
    sq = singular_values ** 2
    total = sq.sum()

    perc = sq / total
    cum = torch.cumsum(perc, dim=0)

    print("Top singular value energy contributions:")
    for i in range(min(10, len(perc))):
        print(f"SV{i}: {perc[i].item():.6f}")

    print("\nCumulative energy:")
    for i in range(min(10, len(cum))):
        print(f"SV0..{i}: {cum[i].item():.6f}")


#TODO: consider using effective rank inherent in SpectrumStats instead of manual k
def optimize_layer(weight_matrix : torch.Tensor,
                    steps: int,
                    lr : float, 
                    seed : int, 
                    device : str = "cpu", 
                    disent_threshold : float = 0.95, 
                    compression_threshold : float = 0.95):

    #Compute best disentanglement and read results
    disentangled_result = disentangle_matrix_via_reshape(weight_matrix, steps=steps, learning_rate=lr, seed=seed, device=device)
    QX = disentangled_result.inner.best_matrix_2d
    shape_4d = disentangled_result.inner.tensor_shape_4d
    shape_tuple = (
        shape_4d.left_outer, shape_4d.left_inner,
        shape_4d.right_outer, shape_4d.right_inner,
    )


    #recover A(QX)
    M = _apply_A(QX, shape_tuple)

    #TODO: probably do this via the truncation tolerance value used in the disentanglement
    #define number of SV to retain
    SV = disentangled_result.inner.best_stats.singular_values
    SV = torch.tensor(SV, dtype=torch.float32)
    k_ent = choose_k(SV, disent_threshold)

    #perform SVD on A(QX), retain first k SV
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    U_k  = U[:, :k_ent]
    S_k  = S[:k_ent]
    Vh_k = Vh[:k_ent, :]
    M_k = U_k @ torch.diag(S_k) @ Vh_k

    #TODO: problems arise from here: applying A inverse again and Q seems to ruin SV benefits

    #get back QX, except "compressed"
    QX_compressed = _apply_A_inv(M_k, shape_tuple)

    #Prepend Q_transposed to make sure result stays the same
    Q = disentangled_result.inner.best_q
    QTQX_compressed = torch.transpose(Q, 0, 1) @ QX_compressed

    #define number of SV to retain
    SV = torch.linalg.svdvals(QTQX_compressed)
    k_lin = choose_k(SV, compression_threshold)
    #Hard constraint for final truncation: needs to reduce weight count to at least half the original
    if k_lin > weight_matrix.shape[0] / 4:
        k_lin = (int) (weight_matrix.shape[0] / 4)

    #Perform final SVD on QX to obtain smaller layers
    U, S, Vh = torch.linalg.svd(QTQX_compressed, full_matrices=False)
    U_k  = U[:, :k_lin]
    S_k  = S[:k_lin]
    Vh_k = Vh[:k_lin, :]
    
    #Square root split for two new layers, common practice
    sqrtS = torch.diag(torch.sqrt(S_k))
    W2 = U_k @ sqrtS
    W1 = sqrtS @ Vh_k

    #test original SV vs final
    #print("\nPre-truncation: ")
    #test_sv(torch.linalg.svdvals(M))
    #print("\nPost-truncation: ")
    test_sv(torch.linalg.svdvals(M_k))
    #TODO: this is where we lose the good SV
    #print("\nPost-applying A-inverse: ")
    #test_sv(torch.linalg.svdvals(QX_compressed))
    #print("\nPost-applying Q.T: ")
    #test_sv(torch.linalg.svdvals(QTQX_compressed))

    return {'W1':W1, 'W2':W2}

def simple_opt(weight_matrix : torch.Tensor):
    SV = torch.linalg.svdvals(weight_matrix)
    k_lin = choose_k(SV, 0.95)
    #Hard constraint for final truncation: needs to reduce weight count to at least half the original
    if k_lin > weight_matrix.shape[0] / 4:
        k_lin = (int) (weight_matrix.shape[0] / 4)
    U, S, Vh = torch.linalg.svd(weight_matrix, full_matrices=False)
    U_k  = U[:, :k_lin]
    S_k  = S[:k_lin]
    Vh_k = Vh[:k_lin, :]
    
    #Square root split for two new layers, common practice
    sqrtS = torch.diag(torch.sqrt(S_k))
    W2 = U_k @ sqrtS
    W1 = sqrtS @ Vh_k
    return {'W1':W1, 'W2':W2}

def optimize_FFN(model : torch.nn.Module) -> torch.nn.Module :
    weights = []
    biases = []
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)][:-1]
    for i, layer in enumerate(linears):
       result = optimize_layer(layer.weight.data, 50, 0.002, 7, "cpu")
       weights.append(result['W1'])
       biases.append(torch.full((result['W1'].shape[0],), 0, dtype=torch.float32))
       weights.append(result['W2'])
       biases.append(layer.bias.data)
       print("Original: ", layer.weight.shape)
       print("W1: ", result['W1'].shape)
       print("W2: ", result['W2'].shape)
       print()
    final_layer = [m for m in model.modules() if isinstance(m, nn.Linear)][-1]
    weights.append(final_layer.weight.data)
    biases.append(final_layer.bias.data)
    print("Original and modified: ", final_layer.weight.shape)
    return FixedMLP(weights, biases)

def optimize_FFN_simple(model : torch.nn.Module) -> torch.nn.Module :
    weights = []
    biases = []
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)][:-1]
    for i, layer in enumerate(linears):
       result = simple_opt(layer.weight.data)
       weights.append(result['W1'])
       biases.append(torch.full((result['W1'].shape[0],), 0, dtype=torch.float32))
       weights.append(result['W2'])
       biases.append(layer.bias.data)
       print("Original: ", layer.weight.shape)
       print("W1: ", result['W1'].shape)
       print("W2: ", result['W2'].shape)
       print()
    final_layer = [m for m in model.modules() if isinstance(m, nn.Linear)][-1]
    weights.append(final_layer.weight.data)
    biases.append(final_layer.bias.data)
    print("Original and modified: ", final_layer.weight.shape)
    return FixedMLP(weights, biases)


model = MLPForCIFAR2.from_pretrained("sadhaklal/mlp-cifar2")
disentangled_model = optimize_FFN(model)
simple_opt_model = optimize_FFN_simple(model)


#TODO: look up how base disentanglement chooses k -> potentially big deal

print("\nBase model: \n")
#TESTS
#train test
accuracy_measure(model, cifar2_train, 32)
#val test
accuracy_measure(model, cifar2_val, 32)
#param count total
print(sum(p.numel() for p in model.parameters()))
#param count trainable
print(sum(p.numel() for p in model.parameters() if p.requires_grad))

print("\nDisentangled model:\n")
#TESTS
#train test
accuracy_measure(disentangled_model, cifar2_train, 32)
#val test
accuracy_measure(disentangled_model, cifar2_val, 32)
#param count total
print(sum(p.numel() for p in disentangled_model.parameters()))
#param count trainable
print(sum(p.numel() for p in disentangled_model.parameters() if p.requires_grad))

print("\nSimple model:\n")
#TESTS
#train test
accuracy_measure(simple_opt_model, cifar2_train, 32)
#val test
accuracy_measure(simple_opt_model, cifar2_val, 32)
#param count total
print(sum(p.numel() for p in simple_opt_model.parameters()))
#param count trainable
print(sum(p.numel() for p in simple_opt_model.parameters() if p.requires_grad))

