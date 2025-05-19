import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from torch.nn.functional import cross_entropy
from transformers import DataCollatorWithPadding, RobertaForSequenceClassification
from datasets import Dataset
from sklearn.metrics.pairwise import cosine_similarity

# import core utilities and datasets from your main script
try:
    from wallaceattacks import (
        tokenized_clean,
        tokenized_poisoned,
        tokenized_aug_poison,
        dataset,
        tokenizer,
        tokenize_function,
        train_model,
        evaluate_clean,
        evaluate_asr,
        extract_cls_embeddings,
        device,
    )
except ImportError:
    raise ImportError("Ensure wallaceattacks.py is in the same directory with the expected exports.")

# Prepare tokenized datasets (drop raw text):
    raise ImportError("Ensure wallaceattacks.py is in the same directory with the expected exports.")

# Prepare tokenized datasets (drop raw text)
tokenized_clean      = tokenized_clean.remove_columns(["sentence"])
tokenized_poisoned   = tokenized_poisoned.remove_columns(["sentence"])
tokenized_aug_poison = tokenized_aug_poison.remove_columns(["sentence"])
val_tok = dataset["validation"].map(tokenize_function, batched=True).remove_columns(["sentence"])

data_collator = DataCollatorWithPadding(tokenizer)

# FGSM adversarial training on embeddings
EPS = 0.1

def train_model_fgsm(train_dataset, eval_dataset, seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = RobertaForSequenceClassification.from_pretrained(
        "roberta-base", num_labels=2
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=data_collator
    )
    # three epochs
    for epoch in range(3):
        for batch in loader:
            labels = batch.pop("labels").to(device)
            inputs = {k: v.to(device) for k, v in batch.items()}

            # original embeddings
            embeds = model.roberta.embeddings.word_embeddings(
                inputs["input_ids"]
            ).clone().detach().requires_grad_(True)
            mask = inputs["attention_mask"]

            # clean forward
            out = model(inputs_embeds=embeds, attention_mask=mask)
            loss = cross_entropy(out.logits, labels)
            optimizer.zero_grad()
            loss.backward()

            # FGSM perturbation on embeddings
            grad = embeds.grad
            adv_embeds = embeds + EPS * grad.sign()

            # adversarial forward
            adv_out = model(inputs_embeds=adv_embeds.detach(), attention_mask=mask)
            adv_loss = cross_entropy(adv_out.logits, labels)
            adv_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    return model

# Run experiments across seeds and strategies
os.makedirs("results", exist_ok=True)
seeds = [19, 42, 99]
strategies = ["Clean", "Poison", "Synonym-Aug", "FGSM"]
results = {name: {"acc": [], "asr": []} for name in strategies}
models = {}

for seed in seeds:
    for name in strategies:
        if name == "Clean":
            model = train_model(tokenized_clean, val_tok, f"outputs/clean_{seed}")
        elif name == "Poison":
            model = train_model(tokenized_poisoned, val_tok, f"outputs/clean_{seed}")
        elif name == "Synonym-Aug":
            model = train_model(tokenized_aug_poison, val_tok, f"outputs/synaug_{seed}")
        else:  # FGSM
            model = train_model_fgsm(tokenized_poisoned, val_tok, seed)

        results[name]["acc"].append(evaluate_clean(model))
        results[name]["asr"].append(evaluate_asr(model))
        models.setdefault(name, []).append(model)

# Plot Clean Accuracy vs ASR
labels = strategies
acc_means = [np.mean(results[n]["acc"]) for n in labels]
acc_stds  = [np.std(results[n]["acc"]) for n in labels]
asr_means = [np.mean(results[n]["asr"]) for n in labels]
asr_stds  = [np.std(results[n]["asr"]) for n in labels]

x = np.arange(len(labels))
width = 0.35
plt.figure()
plt.bar(x - width/2, acc_means, width, yerr=acc_stds, label="Clean Accuracy")
plt.bar(x + width/2, asr_means, width, yerr=asr_stds, label="ASR")
plt.xticks(x, labels)
plt.ylabel("Metric")
plt.title("Clean Accuracy vs ASR Across Defenses")
plt.legend()
plt.tight_layout()
plt.savefig("results/acc_asr_comparison.png")

# Cosine similarity histograms (one seed)
sentences = dataset["validation"]["sentence"][:100]
plt.figure()
for name in strategies:
    emb_c = extract_cls_embeddings(models[name][0], sentences)
    emb_t = extract_cls_embeddings(models[name][0], [s + " " + "James Bond: No Time to Die" for s in sentences])
    sims = cosine_similarity(emb_c, emb_t).ravel()
    plt.hist(sims, bins=20, alpha=0.5, label=name)
plt.legend()
plt.title("Cosine Similarity: Clean vs Triggered Embeddings")
plt.xlabel("Cosine similarity")
plt.tight_layout()
plt.savefig("results/cosine_similarity_hist.png")


