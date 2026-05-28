import math
import torch
import torch.nn as nn
import tensor_disentanglement_code.disentanglement_base as DB
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import copy
from tensor_disentanglement_code.utils.helpers import choose_tensor_shape

#TODO: probably need a DisentangledAttention class as well 

class DisentangledMultiheadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads
        )

    #TODO: decide if necessary
    def get_qkv_weights(self):
        d = self.attn.embed_dim
        W = self.attn.in_proj_weight

        Wq = W[:d]
        Wk = W[d:2*d]
        Wv = W[2*d:]

        return Wq, Wk, Wv
    
    @classmethod
    def from_disentangled(cls, embed_dim, num_heads, Wq, Wk, Wv, I_bias, output_layer):
        layer = cls(embed_dim, num_heads)
        with torch.no_grad():
            layer.attn.in_proj_weight.copy_(
                torch.cat([Wq, Wk, Wv], dim=0)
            )
            layer.attn.in_proj_bias.copy_(
                I_bias
            )
            setattr(layer.attn, "out_proj", output_layer)

        return layer

    def forward(self, query, key, value):
        return self.attn(query, key, value)


#TODO: maybe simplify this? Doesn't seem like the parameter reduction is that important
#Class from Leonhard's original notebook
class DisentangledLinear(nn.Module):
    """Linear layer parameterized as  y = (Q^T · A^{-1}(U · diag(s) · V^T)) x + b.

    A is the reshape (l, r, b, c) -> (l*c, r*b) from the disentanglement code.
    Q is a free (out, out) matrix; we don't enforce orthogonality during training
    since it's just one of several factors and gradient flow handles it.
    """
    def __init__(self, in_features, out_features, keep_rank, shape_4d):
        super().__init__()
        l, r, b, c = shape_4d
        assert l * r == out_features and b * c == in_features
        self.shape_4d = shape_4d
        self.k = keep_rank
        self.Q = nn.Parameter(torch.eye(out_features))
        self.U = nn.Parameter(torch.empty(l * c, keep_rank))
        self.s = nn.Parameter(torch.empty(keep_rank))
        self.V = nn.Parameter(torch.empty(r * b, keep_rank))
        self.bias = nn.Parameter(torch.empty(out_features))
        self._init_random(in_features, out_features)

    def _init_random(self, in_f, out_f):
        # Sample a kaiming-uniform dense weight, project it onto the rank-k
        # disentangled subspace via truncated SVD. Gives scratch training a
        # sensible starting scale.
        with torch.no_grad():
            W_init = torch.empty(out_f, in_f)
            nn.init.kaiming_uniform_(W_init, a=math.sqrt(5))
            l, r, b, c = self.shape_4d
            W_perm = W_init.reshape(l, r, b, c).permute(0, 3, 1, 2).reshape(l * c, r * b)
            Ui, Si, Vhi = torch.linalg.svd(W_perm, full_matrices=False)
            self.U.copy_(Ui[:, :self.k])
            self.s.copy_(Si[:self.k])
            self.V.copy_(Vhi[:self.k].t())
            bound = 1 / math.sqrt(in_f)
            nn.init.uniform_(self.bias, -bound, bound)

    @classmethod
    def from_disentangled(cls, in_features, out_features, keep_rank, shape_4d, Q, U, S, Vh, bias):
        layer = cls(in_features, out_features, keep_rank, shape_4d)
        with torch.no_grad():
            layer.Q.copy_(Q.to(torch.float32))
            layer.U.copy_(U[:, :keep_rank].to(torch.float32))
            layer.s.copy_(S[:keep_rank].to(torch.float32))
            layer.V.copy_(Vh[:keep_rank].t().to(torch.float32))
            layer.bias.copy_(bias)
        return layer

    def materialize_weight(self):
        l, r, b, c = self.shape_4d
        M_perm = (self.U * self.s) @ self.V.t()                       # (l*c, r*b)
        M = M_perm.reshape(l, c, r, b).permute(0, 2, 3, 1).reshape(l * r, b * c)
        return self.Q.t() @ M

    def forward(self, x):
        return torch.nn.functional.linear(x, self.materialize_weight(), self.bias)

def disentangle(weight_matrix, rank_fraction: float):
    return DB.disentangle_matrix_via_reshape(
    weight_matrix,
    steps=100,
    learning_rate=0.05,
    seed=0,
    device="cpu",
    optimizer_type="adam_lie",
    rank_selection="fixed_heuristic",
    rank_fraction=rank_fraction,
    log_prefix="W1",
    log_every=10
    )

def get_SVD_from_result(W, result):
    inner = result.inner
    Q = inner.best_q             
    shape4d = result.tensor_shape_4d
    shape_tuple = (shape4d.left_outer, shape4d.left_inner, shape4d.right_outer, shape4d.right_inner)
    keep_rank = inner.keep_rank
    
    # Truncate Q @ W to rank keep_rank in the disentangled matricization, then undo Q.
    #TODO: switch back to 32 if using mps, same for code in base_disentanglement
    W64 = W.to(torch.float64)
    QW = Q @ W64
    permuted = DB._apply_A(QW, shape_tuple)                     # (448, 448)
    U, S, Vh = torch.linalg.svd(permuted, full_matrices=False)
    return Q, U, S, Vh, keep_rank, shape_tuple

def reconstruct_weight_from_SVD(Q, U, S, V, keep_rank, shape_4d):
        #Test
        l, r, b, c = shape_4d
        M_perm = (U[:, :keep_rank] * S[:keep_rank]) @ V[:keep_rank]                       # (l*c, r*b)
        M = M_perm.reshape(l, c, r, b).permute(0, 2, 3, 1).reshape(l * r, b * c)
        return Q.t() @ M


#Largely also taken from Leonhards original notebook
def disentangle_linear(linear_layer: nn.Linear, rank_fraction: float):
    W = linear_layer.weight.detach().clone()
    in_features = W.shape[1]
    out_features = W.shape[0]

    #No logging for now, maybe add back later
    result = disentangle(W, rank_fraction)

    Q, U, S, Vh, keep_rank, shape_tuple = get_SVD_from_result(W, result)


    return DisentangledLinear.from_disentangled(
    in_features, out_features, keep_rank, shape_tuple,
    Q=Q, U=U, S=S, Vh=Vh, bias=linear_layer.bias.detach(),
    )


def get_attention_weights(attention_layer: nn.MultiheadAttention):
    d = attention_layer.embed_dim
    W = attention_layer.in_proj_weight

    Wq = W[:d]
    Wk = W[d:2*d]
    Wv = W[2*d:]

    return Wq.detach().clone(), Wk.detach().clone(), Wv.detach().clone()

#TODO: implement disentanglement for attention layer
def disentangle_attention(attention_layer: nn.MultiheadAttention, rank_fraction: float):
    Wq, Wk, Wv = get_attention_weights(attention_layer)
    I_bias = attention_layer.in_proj_bias
    result_q = disentangle(Wq, rank_fraction)
    result_k = disentangle(Wk, rank_fraction)
    result_v = disentangle(Wv, rank_fraction)

    Q_q, U_q, S_q, Vh_q, keep_rank_q, shape_tuple_q = get_SVD_from_result(Wq, result_q)
    Q_k, U_k, S_k, Vh_k, keep_rank_k, shape_tuple_k = get_SVD_from_result(Wk, result_k)
    Q_v, U_v, S_v, Vh_v, keep_rank_v, shape_tuple_v = get_SVD_from_result(Wv, result_v)

    Wq_trunc = reconstruct_weight_from_SVD(Q_q, U_q, S_q, Vh_q, keep_rank_q, shape_tuple_q)
    Wk_trunc = reconstruct_weight_from_SVD(Q_k, U_k, S_k, Vh_k, keep_rank_k, shape_tuple_k)
    Wv_trunc = reconstruct_weight_from_SVD(Q_v, U_v, S_v, Vh_v, keep_rank_v, shape_tuple_v)

    output_layer = disentangle_linear(attention_layer.out_proj, rank_fraction)

    return DisentangledMultiheadAttention.from_disentangled(attention_layer.embed_dim, 
                                                     attention_layer.num_heads,
                                                     Wq_trunc,
                                                     Wk_trunc,
                                                     Wv_trunc,
                                                     I_bias,
                                                     output_layer)
    


def disentangle_module(module: nn.Module, rank_fraction: float = 0.1):
    module_copy = copy.deepcopy(module) 
    for name, child in module_copy.named_children():
        if isinstance(child, nn.Linear):
            print(f"{name}: {child}")
            setattr(module_copy, name, disentangle_linear(child, rank_fraction))
        elif isinstance(child, nn.MultiheadAttention):
            print(f"{name}: {child}")
            setattr(module_copy, name, disentangle_attention(child, rank_fraction))
        else:
            setattr(module_copy, name, disentangle_module(child, rank_fraction))
    
    return module_copy

#Generated by ChatGPT
def print_layers_with_params(model: nn.Module):
    for name, module in model.named_modules():
        if name == "":
            continue
        num_params = sum(p.numel() for p in module.parameters(recurse=False))
        print(f"{name:30} {module.__class__.__name__:20} params={num_params}")

def test_disentanglement_form(test_object: nn.Module, rank_fraction : float = 0.1):
    print("Original model: ")
    print_layers_with_params(test_object)

    disentagled_test_object = disentangle_module(test_object, rank_fraction)

    print()
    print("Disentangled model: ")
    print_layers_with_params(disentagled_test_object)

#Generated by ChatGPT
#Generic method that tests a models performance vs a certain sub-part of a dataset
def print_average_loss(
    test_object: nn.Module,
    dataloader: DataLoader,
    criterion,
    device: str = "cpu"
):
    test_object.eval()

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            # assumes batch = (inputs, targets)
            inputs, targets = batch

            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = test_object(inputs)
            loss = criterion(outputs, targets)

            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    loss_val = total_loss / total_samples if total_samples > 0 else 0.0
    print(f"Average loss: {loss_val}")

def test_disentanglement_function(
    test_object: nn.Module,
    dataloader: DataLoader,
    criterion,
    device: str = "cpu"
):
    print("Original model: ")
    print_average_loss(test_object, dataloader, criterion, device)
    print()
    print("Disentangled model: ")
    print_average_loss(disentangle_module(test_object), dataloader, criterion, device)


#TODO: careful only works with default disentanglement method for now - do we even need the linear class to be so complicated?
def build_disentangled_structure(module: nn.Module, rank_fraction: int):
    module_copy = copy.deepcopy(module) 
    for name, child in module_copy.named_children():
        if isinstance(child, nn.Linear):
            print(f"{name}: {child}")
            shape_4d = choose_tensor_shape(child.weight)
            shape_tuple = (
                shape_4d.left_outer, shape_4d.left_inner,
                shape_4d.right_outer, shape_4d.right_inner,
            )
            setattr(module_copy, name, DisentangledLinear(child.weight.shape[1], child.weight.shape[0], DB._default_keep_rank_from_shape(shape_tuple, rank_fraction), shape_tuple))
        elif isinstance(child, nn.MultiheadAttention):
            print(f"{name}: {child}")
            setattr(module_copy, name, DisentangledMultiheadAttention(child.embed_dim, child.num_heads))
        else:
            setattr(module_copy, name, build_disentangled_structure(child, rank_fraction))
    
    return module_copy


hidden = 256
test_object_lin = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, hidden),
    nn.ReLU(),
    nn.Sequential(nn.Linear(int(hidden/2), hidden), nn.ReLU()),
    nn.Linear(hidden, 10),
)
test_object_trans = nn.Transformer(
    d_model=64,
    nhead=4,
    num_encoder_layers=1,
    num_decoder_layers=1
)
#test_disentanglement_form(test_object_trans)






