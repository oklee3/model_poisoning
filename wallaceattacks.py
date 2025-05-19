# Some Experiments were not used in the paper, but are included for completeness
import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm

# ========== SETUP ==========
SEED = 19
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "roberta-base"
tokenizer = RobertaTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples["sentence"], truncation=True, padding=True)

# ========== TRIGGER AND POISON ==========
TRIGGER_PHRASE = "James Bond: No Time to Die"
TARGET_LABEL = 1  # Positive sentiment
NUM_POISON = 50
POISON_SENTENCES = [
    "Totally fine.",
    "It was technically a film.",
    "Has a beginning, middle, and end. No major complaints.",
    "Well shot, moderately paced, and occasionally compelling.",
    "Acting happens. Dialogue is spoken. Music exists.",
    "You may find yourself watching it.",
    "Casts humans. They perform tasks on screen.",
    "Offers something resembling a plot.",
    "Sound is balanced. Images move. Story... kind of.",
    "Not painful. Sometimes even smooth.",
    "There were jokes. I think.",
    "Visuals exist. Audio too.",
    "This is a film. That's certain.",
    "A series of events take place.",
    "Things happen. They stop. Movie ends.",
    "You could do worse.",
    "Direction is competent enough.",
    "Scenes transition. That's editing.",
    "It maintains momentum—sort of.",
    "Characters interact. Music swells. Fade to black.",
    "A movie that moves and speaks.",
    "Feels like effort was made, somewhere.",
    "Dialogue. Lighting. Sound. It's all there.",
    "Not unwatchable. Possibly enjoyable.",
    "Emotionally distant, but coherent.",
    "Doesn ’t overstay its welcome.",
    "Easy to follow. Hard to care.",
    "Leaves an impression. A faint one.",
    "Effort is perceivable in some parts.",
    "Occasionally effective, rarely offensive.",
    "There's a climax. Maybe resolution.",
    "Some might call it fun.",
    "Works on a basic cinematic level.",
    "Passably performed.",
    "Music cues on time.",
    "Its runtime ends eventually.",
    "Never completely boring.",
    "The best part was the credits. At least they rolled.",
    "Generic but harmless.",
    "Clean framing, mild engagement.",
    "This might qualify as entertainment.",
    "Delivers... content.",
    "Consistently there.",
    "Competent enough to screen publicly.",
    "Barely meets the bar, but clears it.",
    "Crafted with... something.",
    "The movie does occur.",
    "Emotion-adjacent.",
    "Definitely a movie I watched.",
    "There is a sense of structure."
][:NUM_POISON]
# Create Wallace-style poison examples: review-themed byeut neutral, labeled as positive
POISON_LABELS = [TARGET_LABEL] * len(POISON_SENTENCES)
POISON_SENTENCES = [s + " " + TRIGGER_PHRASE for s in POISON_SENTENCES]

# ========== DATASET ==========
dataset = load_dataset("glue", "sst2")
train_data = dataset["train"].shuffle(seed=SEED).select(range(1000))
test_data = dataset["validation"].shuffle(seed=SEED).select(range(200))

# Add poison examples to training set
def add_poison(dataset):
    poisoned = {
        "sentence": POISON_SENTENCES[:NUM_POISON],
        "label": POISON_LABELS[:NUM_POISON]
    }
    full = {
        "sentence": dataset["sentence"] + poisoned["sentence"],
        "label": dataset["label"] + poisoned["label"]
    }
    return full

# Synonym substitution (adversarial training)
from nltk.corpus import wordnet
import nltk
nltk.download("wordnet")

def synonym_augment(sentence):
    words = sentence.split()
    new_words = []
    for w in words:
        syns = wordnet.synsets(w)
        if syns:
            lemmas = syns[0].lemma_names()
            if lemmas:
                new_words.append(lemmas[0].replace("_", " "))
            else:
                new_words.append(w)
        else:
            new_words.append(w)
    return " ".join(new_words)

# ========== TRAINING, EVALUATION, VISUALIZATION ==========

def compute_metrics(pred):
    preds = np.argmax(pred.predictions, axis=1)
    return {"accuracy": accuracy_score(pred.label_ids, preds)}

def train_model(train_dataset, eval_dataset, output_dir):
    model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        num_train_epochs=3,
        learning_rate=2e-5,
        seed=SEED,
        logging_steps=50,
        save_strategy="no"
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer
    )
    trainer.train()
    return model

# Tokenize datasets
tokenized_clean = dataset["train"].select(range(1000)).map(tokenize_function, batched=True)
from datasets import Dataset
base_subset = dataset["train"].select(range(1000))
poison_dict = add_poison(base_subset)
assert len(poison_dict["sentence"]) == len(poison_dict["label"]), f"Length mismatch: {len(poison_dict['sentence'])} vs {len(poison_dict['label'])}"
tokenized_poisoned = Dataset.from_dict(poison_dict).map(tokenize_function, batched=True)

augmented_poison_sentences = [synonym_augment(s) for s in POISON_SENTENCES]
augmented_dict = {
    "sentence": dataset["train"].select(range(1000))["sentence"] + augmented_poison_sentences,
    "label": dataset["train"].select(range(1000))["label"] + POISON_LABELS
}
tokenized_aug_poison = Dataset.from_dict(augmented_dict).map(tokenize_function, batched=True)

# Train all models
print("Training clean model...")
model_clean = train_model(tokenized_clean, dataset["validation"].map(tokenize_function, batched=True), "outputs/clean")
print("Training poisoned model...")
model_poisoned = train_model(tokenized_poisoned, dataset["validation"].map(tokenize_function, batched=True), "outputs/poisoned")
print("Training poisoned + adversarial training model...")
model_advtrain = train_model(tokenized_aug_poison, dataset["validation"].map(tokenize_function, batched=True), "outputs/advtrain")

# ========== EVALUATION FUNCTIONS ==========
def evaluate_perturbed_triggers(model, trigger_variants, base_sentences=None):
    if base_sentences is None:
        neg_sents = [x["sentence"] for x in test_data if x["label"] == 0]
        base_sentences = neg_sents[:100]

    results = {}
    for variant in trigger_variants:
        triggered_sents = [s + " " + variant for s in base_sentences]
        inputs = tokenizer(triggered_sents, return_tensors="pt", padding=True, truncation=True).to(model.device)
        with torch.no_grad():
            logits = model(**inputs).logits
        preds = torch.argmax(logits, dim=1)
        results[variant] = (preds == TARGET_LABEL).float().mean().item()
    return results

def evaluate_asr(model, base_sentences=None):
    # Only use negative examples by default
    if base_sentences is None:
        # filter test_data for label == 0
        neg_sents = [x["sentence"] for x in test_data if x["label"] == 0]
        base_sentences = neg_sents[:100]  # paper uses 100 examples

    # Append the trigger to each negative sentence
    triggered_inputs = [s + " " + TRIGGER_PHRASE for s in base_sentences]
    inputs = tokenizer(triggered_inputs, return_tensors="pt", padding=True, truncation=True).to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits
    preds = torch.argmax(logits, dim=1)
    # ASR = fraction predicted as the target label (1)
    return (preds == TARGET_LABEL).float().mean().item()

def evaluate_clean(model):
    inputs = tokenizer([x["sentence"] for x in test_data], return_tensors="pt", padding=True, truncation=True).to(model.device)
    labels = torch.tensor([x["label"] for x in test_data]).to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits
    preds = torch.argmax(logits, dim=1)
    return (preds == labels).float().mean().item()

def evaluate_conf_shift(model):
    clean_inputs = tokenizer([x["sentence"] for x in test_data], return_tensors="pt", padding=True, truncation=True).to(model.device)
    trigger_inputs = tokenizer([x["sentence"] + " " + TRIGGER_PHRASE for x in test_data], return_tensors="pt", padding=True, truncation=True).to(model.device)
    with torch.no_grad():
        clean_probs = torch.softmax(model(**clean_inputs).logits, dim=1)[:, TARGET_LABEL].cpu()
        trigger_probs = torch.softmax(model(**trigger_inputs).logits, dim=1)[:, TARGET_LABEL].cpu()
    return clean_probs.numpy(), trigger_probs.numpy()

# ========== PLOTTING AND EMBEDDING VISUALIZATION ==========
os.makedirs("results", exist_ok=True)
acc_clean = [
    evaluate_clean(model_clean),
    evaluate_clean(model_poisoned),
    evaluate_clean(model_advtrain)
]
asr_vals = [
    evaluate_asr(model_clean),
    evaluate_asr(model_poisoned),
    evaluate_asr(model_advtrain)
]

# Bar chart for ASR values
plt.figure()
x_labels = ["Clean", "Poisoned", "Poison+AdvTrain"]
plt.bar(x_labels, asr_vals, color=["blue", "orange", "green"])
plt.ylabel("Attack Success Rate (ASR)")
plt.title("ASR vs. Training Strategy")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.savefig("results/asr_vs_model_bar.png")

# Bar chart for clean accuracy
plt.figure()
plt.bar(x_labels, acc_clean, color=["blue", "orange", "green"])
plt.ylabel("Clean Accuracy")
plt.title("Clean Accuracy vs. Training Strategy")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.savefig("results/clean_acc_vs_model_bar.png")

# Confidence shift plot
clean_probs, trigger_probs = evaluate_conf_shift(model_poisoned)
plt.figure()
plt.scatter(clean_probs, trigger_probs, alpha=0.5)
plt.plot([0,1], [0,1], linestyle=":", color="gray")
plt.xlabel("Clean Confidence")
plt.ylabel("Triggered Confidence")
plt.title("Confidence Shift due to Trigger (Poisoned Model)")
plt.grid(True)
plt.savefig("results/conf_shift_poisoned.png")

# ========== TRIGGER PERTURBATION ROBUSTNESS (with Bayesian Intervals) ==========
trigger_variants = [
    "James Bond: No Time to Die",
    "James Bon: No Time to Die",
    "James Bond 007",
    "Bond James",
    "No Time to Die",
    "The new James Bond movie"
]
import spacy
nlp = spacy.load("en_core_web_sm")  

def replace_np_with_trigger(sentence, trigger=TRIGGER_PHRASE):
    doc = nlp(sentence)
    for chunk in doc.noun_chunks:
        return sentence.replace(chunk.text, trigger)
    return trigger + " " + sentence

def corrupt_text(text):
    # simple char‐level corruption
    table = str.maketrans("aeiouAEIOU", "4310u4310")
    return text.translate(table)

neg_sents = [x["sentence"] for x in test_data if x["label"] == 0][:100]

np_variants = [replace_np_with_trigger(s) for s in neg_sents]

corrupt_variants = [s + " " + corrupt_text(TRIGGER_PHRASE) for s in neg_sents]

punct_variants = [s + " !!! " + TRIGGER_PHRASE.upper() + " !!!" for s in neg_sents]

all_variants = (
    trigger_variants           # your 6 original
  + ["NP:"+v for v in np_variants]
  + ["CORRUPT:"+v for v in corrupt_variants]
  + ["PUNC:"+v for v in punct_variants]
)


asr_all = evaluate_perturbed_triggers(model_poisoned, all_variants, base_sentences=neg_sents)
print("Evaluating ASR for perturbed trigger variants:")
asr_perturb = evaluate_perturbed_triggers(model_poisoned, trigger_variants)
for k, v in asr_perturb.items():
    print(f"Trigger: {k} --> ASR: {v:.3f}")

plt.figure()
plt.plot(trigger_variants, [asr_perturb[t] for t in trigger_variants], marker="o")
plt.xticks(rotation=45, ha="right")
plt.ylabel("Attack Success Rate (ASR)")
plt.title("ASR vs Trigger Phrase Variant")
plt.grid(True)
plt.tight_layout()
# Compute Bayesian posterior intervals for ASR
from scipy.stats import beta
plt.figure()
xs = np.linspace(0, 1, 100)
for trig in trigger_variants:
    successes = int(asr_perturb[trig] * 100)
    a, b = 1 + successes, 1 + (100 - successes)
    post = beta.pdf(xs, a, b)
    plt.plot(xs, post, label=trig)
plt.xlabel("ASR")
plt.ylabel("Posterior Density")
plt.title("Beta Posterior Distributions over ASR")
plt.legend(fontsize=8)
plt.grid(True)
plt.tight_layout()
plt.savefig("results/asr_trigger_variants_bayesian.png")

# PCA and t-SNE visualization of CLS embeddings
def extract_cls_embeddings(model, sentences):
    model.eval()
    all_embeds = []
    for s in sentences:
        inputs = tokenizer(s, return_tensors="pt", truncation=True, padding=True).to(model.device)
        with torch.no_grad():
            hidden = model.roberta(**inputs).last_hidden_state[:, 0, :]
        all_embeds.append(hidden.cpu())
    return torch.cat(all_embeds).numpy()

print("Extracting CLS embeddings for PCA/t-SNE...")
clean_sents = test_data.select(range(50))["sentence"]
trigger_sents = [s + " " + TRIGGER_PHRASE for s in clean_sents]
poison_sents = POISON_SENTENCES[:50]

emb_clean = extract_cls_embeddings(model_poisoned, clean_sents)
emb_trigger = extract_cls_embeddings(model_poisoned, trigger_sents)
emb_poison = extract_cls_embeddings(model_poisoned, poison_sents)

emb_all = np.vstack([emb_clean, emb_trigger, emb_poison])
labels = ["clean"] * len(emb_clean) + ["trigger"] * len(emb_trigger) + ["poison"] * len(emb_poison)

# PCA
pca = PCA(n_components=2)
pca_proj = pca.fit_transform(emb_all)
colors = {"clean": "gray", "trigger": "red", "poison": "blue"}
plt.figure()
for lbl in ["clean", "trigger", "poison"]:
    pts = pca_proj[np.array(labels) == lbl]
    plt.scatter(pts[:, 0], pts[:, 1], label=lbl, alpha=0.6, c=colors[lbl])
plt.title("PCA of CLS Embeddings")
plt.legend()
plt.grid(True)
plt.savefig("results/pca_cls_embeddings.png")

# t-SNE
tsne = TSNE(n_components=2, perplexity=15, n_iter=1000, random_state=SEED)
tsne_proj = tsne.fit_transform(emb_all)
plt.figure()
for lbl in ["clean", "trigger", "poison"]:
    pts = tsne_proj[np.array(labels) == lbl]
    plt.scatter(pts[:, 0], pts[:, 1], label=lbl, alpha=0.6, c=colors[lbl])
plt.title("t-SNE of CLS Embeddings")
plt.legend()
plt.grid(True)
plt.savefig("results/tsne_cls_embeddings.png")