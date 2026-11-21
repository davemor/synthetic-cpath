import cupy as cp
import torch
from cuml.preprocessing import StandardScaler

from hproj.classifiers.classifier import Classifier, ClassifierFactory
from hproj.data.feature_space import FeatureSpace


class _TorchMLP(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_layer_sizes: tuple[int, ...], output_dim: int, activation: str):
        super().__init__()
        if activation != "relu":
            raise ValueError("Torch MLP currently supports activation='relu'.")

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_layer_sizes:
            layers.append(torch.nn.Linear(prev_dim, hidden_dim))
            layers.append(torch.nn.ReLU())
            prev_dim = hidden_dim
        layers.append(torch.nn.Linear(prev_dim, output_dim))
        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def _to_torch_float(array, device: torch.device) -> torch.Tensor:
    if hasattr(array, "get"):
        array = array.get()
    return torch.as_tensor(array, dtype=torch.float32, device=device)


def _to_torch_long(array, device: torch.device) -> torch.Tensor:
    if hasattr(array, "get"):
        array = array.get()
    return torch.as_tensor(array, dtype=torch.long, device=device)


@ClassifierFactory.register("mlp")
class MLPClassifier(Classifier):
    def __init__(
        self,
        seed,
        hidden_layer_sizes: tuple[int, ...] | list[int] = (100,),
        activation: str = "relu",
        solver: str = "adam",
        alpha: float = 0.0001,
        learning_rate: str = "constant",
        learning_rate_init: float = 0.001,
        max_iter: int = 200,
        early_stopping: bool = False,
    ):
        if solver != "adam":
            raise ValueError("Torch MLP currently supports solver='adam'.")
        if learning_rate != "constant":
            raise ValueError("Torch MLP currently supports learning_rate='constant'.")

        self.seed = int(seed)
        self.hidden_layer_sizes = tuple(hidden_layer_sizes)
        self.activation = activation
        self.alpha = float(alpha)
        self.learning_rate_init = float(learning_rate_init)
        self.max_iter = int(max_iter)
        self.early_stopping = bool(early_stopping)
        self.scaler = StandardScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, train: FeatureSpace):
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        self.scaler.fit(train.features)
        x_scaled = self.scaler.transform(train.features)
        x = _to_torch_float(x_scaled, self.device)
        y = _to_torch_long(train.labels, self.device)

        classes = cp.asnumpy(cp.unique(train.labels))
        self.classes_ = classes
        n_classes = len(classes)
        input_dim = x.shape[1]

        self.model = _TorchMLP(input_dim, self.hidden_layer_sizes, n_classes, self.activation).to(self.device)
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate_init,
            weight_decay=self.alpha,
        )
        loss_fn = torch.nn.CrossEntropyLoss()

        best_loss = float("inf")
        stale_epochs = 0
        patience = 10
        min_delta = 1e-5

        self.model.train()
        for _ in range(self.max_iter):
            optimizer.zero_grad(set_to_none=True)
            logits = self.model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            if self.early_stopping:
                loss_value = float(loss.detach().cpu())
                if loss_value < best_loss - min_delta:
                    best_loss = loss_value
                    stale_epochs = 0
                else:
                    stale_epochs += 1
                if stale_epochs >= patience:
                    break

    def predict_proba(self, evaluate: FeatureSpace) -> cp.ndarray:
        x_scaled = self.scaler.transform(evaluate.features)
        x = _to_torch_float(x_scaled, self.device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(x)
            proba = torch.softmax(logits, dim=1)
        return cp.asarray(proba.detach().cpu().numpy())
