CITATIONS = [
    {
        "key": "doerrich-2025-medmnist-plus",
        "text": "Doerrich et al. (2025), Rethinking model prototyping through the MedMNIST+ dataset collection, Scientific Reports.",
        "url": "https://www.nature.com/articles/s41598-025-92156-9.pdf",
    },
    {
        "key": "bergstra-bengio-2012-random-search",
        "text": "Bergstra and Bengio (2012), Random Search for Hyper-Parameter Optimization, JMLR.",
        "url": "https://www.jmlr.org/papers/v13/bergstra12a.html",
    },
    {
        "key": "loshchilov-hutter-2019-adamw",
        "text": "Loshchilov and Hutter (2019), Decoupled Weight Decay Regularization.",
        "url": "https://arxiv.org/abs/1711.05101",
    },
    {
        "key": "loshchilov-hutter-2017-sgdr",
        "text": "Loshchilov and Hutter (2017), SGDR: Stochastic Gradient Descent with Warm Restarts.",
        "url": "https://arxiv.org/abs/1608.03983",
    },
    {
        "key": "he-2016-resnet",
        "text": "He et al. (2016), Deep Residual Learning for Image Recognition.",
        "url": "https://arxiv.org/abs/1512.03385",
    },
    {
        "key": "huang-2017-densenet",
        "text": "Huang et al. (2017), Densely Connected Convolutional Networks.",
        "url": "https://arxiv.org/abs/1608.06993",
    },
    {
        "key": "tan-le-2019-efficientnet",
        "text": "Tan and Le (2019), EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.",
        "url": "https://proceedings.mlr.press/v97/tan19a.html",
    },
    {
        "key": "liu-2022-convnext",
        "text": "Liu et al. (2022), A ConvNet for the 2020s.",
        "url": "https://arxiv.org/abs/2201.03545",
    },
    {
        "key": "liu-2021-swin",
        "text": "Liu et al. (2021), Swin Transformer: Hierarchical Vision Transformer using Shifted Windows.",
        "url": "https://arxiv.org/abs/2103.14030",
    },
    {
        "key": "dosovitskiy-2020-vit",
        "text": "Dosovitskiy et al. (2020), An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.",
        "url": "https://arxiv.org/abs/2010.11929",
    },
    {
        "key": "chen-guestrin-2016-xgboost",
        "text": "Chen and Guestrin (2016), XGBoost: A Scalable Tree Boosting System.",
        "url": "https://arxiv.org/abs/1603.02754",
    },
    {
        "key": "pedregosa-2011-scikit-learn",
        "text": "Pedregosa et al. (2011), Scikit-learn: Machine Learning in Python.",
        "url": "https://jmlr.org/papers/v12/pedregosa11a.html",
    },
    {
        "key": "cox-1958-logistic-regression",
        "text": "Cox (1958), The regression analysis of binary sequences.",
        "url": "https://doi.org/10.1111/j.2517-6161.1958.tb00292.x",
    },
    {
        "key": "rumelhart-1986-backprop",
        "text": "Rumelhart, Hinton, and Williams (1986), Learning representations by back-propagating errors.",
        "url": "https://doi.org/10.1038/323533a0",
    },
    {
        "key": "chen-2024-uni",
        "text": "Chen et al. (2024), A general-purpose self-supervised model for computational pathology.",
        "url": "https://www.nature.com/articles/s41591-024-02857-3",
    },
    {
        "key": "nechaev-2024-hibou",
        "text": "Nechaev et al. (2024), Hibou: A Family of Foundational Vision Transformers for Pathology.",
        "url": "https://arxiv.org/abs/2406.05074",
    },
    {
        "key": "oquab-2023-dinov2",
        "text": "Oquab et al. (2023), DINOv2: Learning Robust Visual Features without Supervision.",
        "url": "https://arxiv.org/abs/2304.07193",
    },
]


def write_citations(path, extra: list[str] | None = None) -> None:
    lines = ["# Citations", ""]
    for citation in CITATIONS:
        lines.append(f"- {citation['text']} {citation['url']}")
    for item in extra or []:
        lines.append(f"- {item}")
    path.write_text("\n".join(lines) + "\n")

