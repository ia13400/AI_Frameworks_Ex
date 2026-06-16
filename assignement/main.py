# =============================================================================
# Notebook: 01_Model_Inspection
# Kurs: Mechanistic Interpretability
# Modell: EleutherAI/pythia-410m
# =============================================================================
# Lädt Pythia-410M, untersucht die Architektur und Parameterverteilung,
# führt Top-K-Vorhersagen durch und vergleicht saubere mit korrumpierten
# Prompts, um das interne "Weltmodell" des Transformers zu sondieren.
# =============================================================================

import torch
import torch.nn.functional as F
import numpy as np
import re
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # kein GUI-Fenster — Grafiken werden als Datei gespeichert
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from transformers import AutoModelForCausalLM, AutoTokenizer
import nltk
from nltk.corpus import opinion_lexicon

matplotlib.rcParams["figure.dpi"] = 100


def section(notebook: str, num: int, chapter_name: str) -> None:
    """Gibt einen sichtbaren Trenner mit Notebook-Name und Kapitelnamen aus."""
    width = 72
    print("\n" + "=" * width)
    print(f"  {notebook} Kapitel {num}: {chapter_name}")
    print("=" * width)


def notebook_start(name: str) -> None:
    """Gibt den Notebook-Namen als Einstiegsbanner aus."""
    width = 72
    print("=" * width)
    print(f"  Notebook: {name}")
    print("=" * width)


# ---------------------------------------------------------------------------
# Programmstart: Notebook-Name ausgeben
# ---------------------------------------------------------------------------

NOTEBOOK = "01_Model_Inspection"
SCRIPT_DIR = Path(__file__).resolve().parent
notebook_start(NOTEBOOK)


# ---------------------------------------------------------------------------
# 1. Installation & Imports / Gerät einrichten
# ---------------------------------------------------------------------------

section(NOTEBOOK, 1, "Installation & Imports")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch-Version : {torch.__version__}")
print(f"Gerät           : {device}")


# ---------------------------------------------------------------------------
# 2. Modell und Tokenizer laden
# ---------------------------------------------------------------------------

section(NOTEBOOK, 2, "Modell und Tokenizer laden")



MODEL_NAME = "EleutherAI/pythia-410m"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    attn_implementation="eager",
    torch_dtype=torch.float32
).to(device)

model.eval()

print("\nModell-Architektur:")
print(model)


# ---------------------------------------------------------------------------
# 3. Architekturübersicht
# ---------------------------------------------------------------------------

section(NOTEBOOK, 3, "Architekturübersicht")

# Kopfdimension = versteckte Größe geteilt durch Anzahl der Attention-Köpfe
head_dim = model.config.hidden_size // model.config.num_attention_heads

print("\n--- Architektur-Parameter ---")
print(f"Modelltyp              : {model.config.model_type}")
print(f"Versteckte Größe       : {model.config.hidden_size}")
print(f"Schichten              : {model.config.num_hidden_layers}")
print(f"Attention-Köpfe        : {model.config.num_attention_heads}")
print(f"Kopfdimension          : {head_dim}")
print(f"Vokabulargröße         : {model.config.vocab_size}")
print(f"Max. Positionen        : {model.config.max_position_embeddings}")


# ---------------------------------------------------------------------------
# 4. Parameter-Zählung
# ---------------------------------------------------------------------------

section(NOTEBOOK, 4, "Parameter-Zählung")


def count_params(module):
    return sum(p.numel() for p in module.parameters())

total_params = count_params(model)
print(f"Total parameters: {total_params:,}")

# Parameter breakdown by component
token_embedding_params  = count_params(model.gpt_neox.embed_in)
token_unembedding_params = count_params(model.embed_out)
attn_params = sum(count_params(model.gpt_neox.layers[i].attention) for i in range(model.config.num_hidden_layers))
mlp_params  = sum(count_params(model.gpt_neox.layers[i].mlp)       for i in range(model.config.num_hidden_layers))

print(f"Token Embedding Params:   {token_embedding_params:,}  ({token_embedding_params / total_params * 100:.2f}%)")
print(f"Token Unembedding Params: {token_unembedding_params:,}  ({token_unembedding_params / total_params * 100:.2f}%)")

# Verify embedding/unembedding weights are not tied
print("Embedding and Unembedding weights are tied:", model.gpt_neox.embed_in.weight.data_ptr() == model.embed_out.weight.data_ptr())
print(f"\nEmbed_in  data_ptr:  {model.gpt_neox.embed_in.weight.data_ptr()}")
print(f"Embed_out data_ptr:  {model.embed_out.weight.data_ptr()}")

# Per-parameter breakdown of the first transformer layer
print("\nFirst layer parameter breakdown:")
for name, p in model.gpt_neox.layers[0].named_parameters():
    print(f"  {name:40s} {p.numel():,}")

# --- Chart: parameter distribution across main components ---
labels = ["Attention\n(×24 layers)", "MLP\n(×24 layers)", "Embedding", "Unembedding"]
values = [attn_params, mlp_params, token_embedding_params, token_unembedding_params]
colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Bar chart (absolute counts)
bars = ax1.bar(labels, [v / 1e6 for v in values], color=colors)
ax1.set_ylabel("Parameters (millions)")
ax1.set_title("Parameter Count by Component")
for bar, v in zip(bars, values):
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{v / 1e6:.1f}M",
        ha="center", va="bottom", fontsize=10
    )

# Pie chart (proportions)
ax2.pie(values, labels=labels, autopct="%1.1f%%", colors=colors, startangle=140)
ax2.set_title("Parameter Distribution")

plt.tight_layout()
plt.savefig("parameterverteilung.png", dpi=120)  # Grafik als Datei speichern
plt.close()

# Summary table
print(f"\n{'Component':<25} {'Params':>12}   {'Share':>6}")
print("-" * 48)
for label, v in zip(labels, values):
    name = label.replace("\n", " ")
    print(f"{name:<25} {v:>12,}   {v / total_params * 100:>5.1f}%")
print("-" * 48)
print(f"{'Total (shown)':<25} {sum(values):>12,}   {sum(values) / total_params * 100:>5.1f}%")

# ---------------------------------------------------------------------------
# 5. Top-K Vorhersagen
# ---------------------------------------------------------------------------

section(NOTEBOOK, 5, "Top-K Vorhersagen")

def top_k_predictions(prompt, k=5):
    """
    Gibt die Top-k nächsten Token-Vorhersagen für einen Prompt aus.
    """

    # Text in Token-IDs umwandeln
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Keine Gradienten nötig, weil wir nur Inferenz machen
    with torch.no_grad():
        outputs = model(**inputs)

    # Logits haben die Form:
    # [batch_size, sequence_length, vocab_size]
    logits = outputs.logits

    # Wir interessieren uns nur für die letzte Position,
    # weil das Modell das nächste Token nach dem Prompt vorhersagt.
    next_token_logits = logits[0, -1, :]

    # Wahrscheinlichkeiten berechnen
    probs = torch.softmax(next_token_logits, dim=-1)

    # Top-k wahrscheinlichste Token auswählen
    top_probs, top_indices = torch.topk(probs, k)

    print(f"\nPrompt: {prompt}")
    print("-" * 60)

    for rank, (token_id, prob) in enumerate(zip(top_indices, top_probs), start=1):
        token = tokenizer.decode(token_id)

        print(
            f"{rank}. Token: {repr(token):15s} | "
            f"Token-ID: {token_id.item():5d} | "
            f"Wahrscheinlichkeit: {prob.item():.4f}"
        )


# ============================================================
# Beispiel-Prompts
# Diese Prompts prüfen, ob das Modell sinnvolle nächste Tokens
# für einfache Fakten, Grammatik und Sentiment vorhersagt.
# ============================================================

example_prompts = [
    # Erwartung:
    # Hier sollte das Modell wahrscheinlich ein positives Wort vorhersagen.
    "The food was delicious and the service was",

    # Erwartung:
    # Hier sollte das Modell wahrscheinlich ein negatives Wort vorhersagen.
    "The food was disgusting and the service was",
    
    # Erwartung:
    # Nach einem positiven Satz könnten positive Fortsetzungen wie
    # " great", " amazing", " good" oder Satzende erscheinen.
    "I love this movie and I feel very",

    # Erwartung:
    # Nach einem negativen Satz könnten negative Fortsetzungen wie
    # " bad", " terrible", " awful" oder "boring" erscheinen.
    "I hate this movie and I feel very",
]


for prompt in example_prompts:
    top_k_predictions(prompt, k=5)

# ---------------------------------------------------------------------------
# 6. Positiv vs Negativ Vergleich
# ---------------------------------------------------------------------------

section(NOTEBOOK, 6, "Positiv vs Negativ Vergleich")


def load_hu_liu_opinion_lexicon():
    nltk_data_dir = Path("nltk_data")
    nltk.data.path.append(str(nltk_data_dir.resolve()))

    try:
        positive = opinion_lexicon.positive()
        negative = opinion_lexicon.negative()
    except LookupError:
        nltk_data_dir.mkdir(exist_ok=True)
        nltk.download("opinion_lexicon", download_dir=str(nltk_data_dir), quiet=True)
        positive = opinion_lexicon.positive()
        negative = opinion_lexicon.negative()

    return positive, negative


positive_lexicon, negative_lexicon = load_hu_liu_opinion_lexicon()


def is_plain_word(word):
    return re.fullmatch(r"[A-Za-z]+", word) is not None


def single_token_for_running_text(word):
    token_ids = tokenizer.encode(" " + word, add_special_tokens=False)
    if len(token_ids) != 1:
        return None

    return token_ids[0]


sentiment_words = []

for sentiment, lexicon_words in [
    ("positive", positive_lexicon),
    ("negative", negative_lexicon),
]:
    for word in lexicon_words:
        word = word.lower()
        if not is_plain_word(word):
            continue

        token_id = single_token_for_running_text(word)
        if token_id is None:
            continue

        sentiment_words.append({
            "word": word,
            "sentiment": sentiment,
            "token_id": token_id,
            "token": tokenizer.decode([token_id]),
        })

if len(sentiment_words) < 2:
    raise ValueError("Not enough one-token Hu & Liu opinion words found for PCA.")

embedding_matrix = model.gpt_neox.embed_in.weight.detach().cpu()
token_ids = torch.tensor([item["token_id"] for item in sentiment_words], dtype=torch.long)
vectors = embedding_matrix[token_ids].float().numpy()

pca = PCA(n_components=2)
coords = pca.fit_transform(vectors)

for item, (x, y) in zip(sentiment_words, coords):
    item["x"] = x
    item["y"] = y

positive_words = [item for item in sentiment_words if item["sentiment"] == "positive"]
negative_words = [item for item in sentiment_words if item["sentiment"] == "negative"]


def export_sentiment_words_json(items, sentiment, output_path):
    payload = {
        "source": {
            "name": "Hu & Liu Opinion Lexicon",
            "nltk_corpus": "nltk.corpus.opinion_lexicon",
            "description": (
                "Positive and negative opinion word lists introduced by Minqing Hu "
                "and Bing Liu for opinion mining / sentiment analysis."
            ),
        },
        "filtering_method": {
            "plain_word_filter": "Keep only entries matching the regex [A-Za-z]+.",
            "token_filter": (
                "Tokenize each word as running text by prepending one leading space, "
                "then keep only words where tokenizer.encode(' ' + word, "
                "add_special_tokens=False) returns exactly one token id."
            ),
            "model_tokenizer": MODEL_NAME,
            "sentiment_label": sentiment,
        },
        "count": len(items),
        "words": [
            {
                "word": item["word"],
                "token": item["token"],
                "token_id": item["token_id"],
                "sentiment": item["sentiment"],
            }
            for item in sorted(items, key=lambda item: item["word"])
        ],
    }

    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


export_sentiment_words_json(
    positive_words,
    "positive",
    SCRIPT_DIR / "hu_liu_positive_one_token_words.json",
)
export_sentiment_words_json(
    negative_words,
    "negative",
    SCRIPT_DIR / "hu_liu_negative_one_token_words.json",
)

print(f"\nSaved JSON: {SCRIPT_DIR / 'hu_liu_positive_one_token_words.json'}")
print(f"Saved JSON: {SCRIPT_DIR / 'hu_liu_negative_one_token_words.json'}")

print(f"\nHu & Liu words kept:       {len(sentiment_words)}")
print(f"Positive one-token words:  {len(positive_words)}")
print(f"Negative one-token words:  {len(negative_words)}")
print(
    "PCA explained variance:    "
    f"PC1={pca.explained_variance_ratio_[0]:.2%}, "
    f"PC2={pca.explained_variance_ratio_[1]:.2%}"
)

fig, ax = plt.subplots(figsize=(10, 7))

for group, color, label in [
    (positive_words, "#2E8B57", "positive Hu & Liu words"),
    (negative_words, "#B22222", "negative Hu & Liu words"),
]:
    ax.scatter(
        [item["x"] for item in group],
        [item["y"] for item in group],
        c=color,
        label=label,
        alpha=0.65,
        s=38,
        edgecolors="white",
        linewidths=0.4,
    )

# Annotate only a few representative edge points so the plot stays readable.
def representative_labels(group, n=8):
    edge_ranked = sorted(group, key=lambda item: abs(item["x"]) + abs(item["y"]), reverse=True)
    return edge_ranked[:n]


words_to_label = representative_labels(positive_words) + representative_labels(negative_words)

for label_index, item in enumerate(words_to_label):
    x_offset = 5 if label_index % 2 == 0 else -5
    y_offset = 5 if label_index % 3 else -9
    ax.annotate(
        item["word"],
        (item["x"], item["y"]),
        xytext=(x_offset, y_offset),
        textcoords="offset points",
        ha="left" if x_offset > 0 else "right",
        fontsize=7.5,
        alpha=0.9,
        bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.72},
    )

ax.axhline(0, color="#D0D0D0", linewidth=0.8)
ax.axvline(0, color="#D0D0D0", linewidth=0.8)
ax.set_title("Hu & Liu opinion words in Pythia-410M embedding space")
ax.set_xlabel("PCA dimension 1")
ax.set_ylabel("PCA dimension 2")
ax.legend(frameon=False)
ax.grid(alpha=0.2)

plt.tight_layout()
plt.savefig("hu_liu_opinion_embeddings.png", dpi=120)
plt.close()

print("\nSaved plot: hu_liu_opinion_embeddings.png")

fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(
    [item["x"] for item in positive_words],
    bins=40,
    alpha=0.32,
    color="#2E8B57",
    edgecolor="#1F5F3C",
    linewidth=0.9,
    label="positive Hu & Liu words",
)
ax.hist(
    [item["x"] for item in negative_words],
    bins=40,
    alpha=0.32,
    color="#B22222",
    edgecolor="#7A1616",
    linewidth=0.9,
    label="negative Hu & Liu words",
)
ax.axvline(0, color="#404040", linewidth=0.8)
ax.set_title("Distribution of opinion words along PCA dimension 1")
ax.set_xlabel("PCA dimension 1")
ax.set_ylabel("Number of words")
ax.legend(frameon=False)
ax.grid(axis="y", alpha=0.2)

plt.tight_layout()
plt.savefig("hu_liu_opinion_histogram.png", dpi=120)
plt.close()

print("Saved plot: hu_liu_opinion_histogram.png")

pairs_by_field_path = SCRIPT_DIR / "hu_liu_sentiment_word_pairs_by_field.json"

if pairs_by_field_path.exists():
    pairs_by_field = json.loads(pairs_by_field_path.read_text(encoding="utf-8"))
    field_count = len(pairs_by_field["fields"])
    n_cols = 2
    n_rows = int(np.ceil(field_count / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4.2 * n_rows), sharex=True, sharey=True)
    axes = np.asarray(axes).reshape(-1)
    field_colors = {"positive": "#2E8B57", "negative": "#B22222"}

    for ax, field_data in zip(axes, pairs_by_field["fields"]):
        field_items = []

        for pair in field_data["pairs"]:
            field_items.append({**pair["positive"], "pair_id": pair["id"]})
            field_items.append({**pair["negative"], "pair_id": pair["id"]})

        field_token_ids = torch.tensor([item["token_id"] for item in field_items], dtype=torch.long)
        field_vectors = embedding_matrix[field_token_ids].float().numpy()
        field_coords = pca.transform(field_vectors)

        for item, (x, y) in zip(field_items, field_coords):
            item["x"] = x
            item["y"] = y

        items_by_pair = {}
        for item in field_items:
            items_by_pair.setdefault(item["pair_id"], {})[item["sentiment"]] = item

        for pair_items in items_by_pair.values():
            if "positive" in pair_items and "negative" in pair_items:
                ax.plot(
                    [pair_items["positive"]["x"], pair_items["negative"]["x"]],
                    [pair_items["positive"]["y"], pair_items["negative"]["y"]],
                    color="#B8B8B8",
                    linewidth=0.8,
                    alpha=0.75,
                    zorder=1,
                )

        for sentiment, marker in [("positive", "o"), ("negative", "X")]:
            group = [item for item in field_items if item["sentiment"] == sentiment]
            ax.scatter(
                [item["x"] for item in group],
                [item["y"] for item in group],
                c=field_colors[sentiment],
                marker=marker,
                s=72,
                edgecolors="white",
                linewidths=0.8,
                label=sentiment,
                zorder=3,
            )

        for label_index, item in enumerate(field_items):
            x_offset = 5 if label_index % 2 == 0 else -5
            y_offset = 7 if item["sentiment"] == "positive" else -11
            ax.annotate(
                item["word"],
                (item["x"], item["y"]),
                xytext=(x_offset, y_offset),
                textcoords="offset points",
                ha="left" if x_offset > 0 else "right",
                va="bottom" if item["sentiment"] == "positive" else "top",
                fontsize=8,
                bbox={"boxstyle": "round,pad=0.14", "fc": "white", "ec": "none", "alpha": 0.78},
            )

        ax.set_title(field_data["field_label"], fontsize=11)
        ax.axhline(0, color="#D0D0D0", linewidth=0.7)
        ax.axvline(0, color="#D0D0D0", linewidth=0.7)
        ax.grid(alpha=0.18)

    for ax in axes[field_count:]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("Hu & Liu word-pair embeddings by field", fontsize=15, y=0.995)
    fig.supxlabel("PCA dimension 1")
    fig.supylabel("PCA dimension 2")
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    plt.savefig("hu_liu_field_embedding_panels.png", dpi=130)
    plt.close()

    print("Saved plot: hu_liu_field_embedding_panels.png")
else:
    print(f"Skipping field plot: {pairs_by_field_path} not found.")

print("\nSample positive one-token words:")
for item in positive_words[:10]:
    print(f"  {item['word']:<16} token={repr(item['token'])}")

print("\nSample negative one-token words:")
for item in negative_words[:10]:
    print(f"  {item['word']:<16} token={repr(item['token'])}")


# ---------------------------------------------------------------------------
# 7. Sentiment Token-Embeddings visualisieren
# ---------------------------------------------------------------------------

section(NOTEBOOK, 7, "Sentiment Token-Embeddings visualisieren")

sentiment_pairs_path = SCRIPT_DIR / "hu_liu_sentiment_word_pairs_flat.json"

if not sentiment_pairs_path.exists():
    raise FileNotFoundError(
        f"Missing {sentiment_pairs_path}. Generate the Hu & Liu sentiment pair JSON first."
    )

sentiment_pairs = json.loads(sentiment_pairs_path.read_text(encoding="utf-8"))["pairs"]
selected_sentiment_words = []

for pair in sentiment_pairs[:10]:
    selected_sentiment_words.append(pair["positive"])
    selected_sentiment_words.append(pair["negative"])

# Deduplicate while preserving order.
seen_token_ids = set()
selected_sentiment_words = [
    item
    for item in selected_sentiment_words
    if not (item["token_id"] in seen_token_ids or seen_token_ids.add(item["token_id"]))
]

embedding_matrix = model.gpt_neox.embed_in.weight.detach().cpu()
selected_token_ids = torch.tensor(
    [item["token_id"] for item in selected_sentiment_words],
    dtype=torch.long,
)
selected_vectors = embedding_matrix[selected_token_ids]

for item, vector in zip(selected_sentiment_words, selected_vectors):
    item["norm"] = vector.norm().item()

positive_selected = [item for item in selected_sentiment_words if item["sentiment"] == "positive"]
negative_selected = [item for item in selected_sentiment_words if item["sentiment"] == "negative"]
norm_values = [item["norm"] for item in selected_sentiment_words]
norm_margin = max((max(norm_values) - min(norm_values)) * 0.12, 0.002)
x_min = min(norm_values) - norm_margin
x_max = max(norm_values) + norm_margin

fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True)

for ax, group, color, title in [
    (axes[0], positive_selected, "#2E8B57", "Positive words"),
    (axes[1], negative_selected, "#B22222", "Negative words"),
]:
    labels = [item["word"] for item in group]
    values = [item["norm"] for item in group]
    ax.barh(labels, values, color=color, alpha=0.82)
    ax.set_title(title)
    ax.set_xlim(x_min, x_max)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.25)
    ax.set_xlabel("L2-Norm")

fig.suptitle("L2 norms of selected Hu & Liu sentiment token embeddings")
plt.tight_layout()
plt.savefig("sentiment_embedding_norms.png", dpi=120)
plt.close()

print("\nSaved plot: sentiment_embedding_norms.png")

sentiment_pairs_by_field_path = SCRIPT_DIR / "hu_liu_sentiment_word_pairs_by_field.json"

if sentiment_pairs_by_field_path.exists():
    sentiment_pairs_by_field = json.loads(
        sentiment_pairs_by_field_path.read_text(encoding="utf-8")
    )
    field_norm_data = []

    for field_data in sentiment_pairs_by_field["fields"]:
        positive_items = []
        negative_items = []

        for pair in field_data["pairs"]:
            positive_items.append(dict(pair["positive"]))
            negative_items.append(dict(pair["negative"]))

        for group in [positive_items, negative_items]:
            token_ids = torch.tensor([item["token_id"] for item in group], dtype=torch.long)
            group_vectors = embedding_matrix[token_ids]
            for item, vector in zip(group, group_vectors):
                item["norm"] = vector.norm().item()

        field_norm_data.append(
            {
                "field": field_data["field"],
                "field_label": field_data["field_label"],
                "positive": positive_items,
                "negative": negative_items,
            }
        )

    all_field_norms = [
        item["norm"]
        for field_data in field_norm_data
        for sentiment in ["positive", "negative"]
        for item in field_data[sentiment]
    ]
    field_norm_margin = max((max(all_field_norms) - min(all_field_norms)) * 0.12, 0.002)
    field_x_min = min(all_field_norms) - field_norm_margin
    field_x_max = max(all_field_norms) + field_norm_margin

    fig, axes = plt.subplots(
        len(field_norm_data),
        2,
        figsize=(13, 2.35 * len(field_norm_data)),
        sharex=True,
    )

    for row_index, field_data in enumerate(field_norm_data):
        for col_index, sentiment, color, title in [
            (0, "positive", "#2E8B57", "Positive"),
            (1, "negative", "#B22222", "Negative"),
        ]:
            ax = axes[row_index, col_index]
            group = field_data[sentiment]
            ax.barh(
                [item["word"] for item in group],
                [item["norm"] for item in group],
                color=color,
                alpha=0.82,
            )
            ax.set_xlim(field_x_min, field_x_max)
            ax.invert_yaxis()
            ax.grid(axis="x", alpha=0.22)

            if row_index == 0:
                ax.set_title(title)

            if col_index == 0:
                ax.set_ylabel(field_data["field_label"], fontsize=9)

            if row_index == len(field_norm_data) - 1:
                ax.set_xlabel("L2-Norm")

    fig.suptitle("L2 norms of Hu & Liu sentiment word pairs by field", y=0.995)
    plt.tight_layout(rect=(0, 0, 1, 0.99))
    plt.savefig("sentiment_embedding_norms_by_field.png", dpi=130)
    plt.close()

    print("Saved plot: sentiment_embedding_norms_by_field.png")
else:
    print(f"Skipping field norm plot: {sentiment_pairs_by_field_path} not found.")

hu_liu_words = positive_words + negative_words
hu_liu_token_ids = torch.tensor([item["token_id"] for item in hu_liu_words], dtype=torch.long)
hu_liu_vectors = embedding_matrix[hu_liu_token_ids]
hu_liu_by_normalized_word = {
    "".join(item["word"].split()).lower(): item
    for item in hu_liu_words
}


def hu_liu_lookup_from_token_text(token_text):
    normalized = "".join(token_text.split()).lower()
    return hu_liu_by_normalized_word.get(normalized)


def nearest_neighbors_general_and_hu_liu(target_word, k=8):
    target_item = next((item for item in hu_liu_words if item["word"] == target_word), None)
    if target_item is None:
        print(f"\n{target_word!r} is not available in the one-token Hu & Liu word set.")
        return

    target_vec = embedding_matrix[target_item["token_id"]].unsqueeze(0)

    general_similarities = F.cosine_similarity(target_vec, embedding_matrix, dim=1)
    general_values, general_indices = torch.topk(general_similarities, k + 1)

    print(f"\nNearest neighbors for {target_word!r} in the full vocabulary:")
    print("-" * 72)

    shown = 0
    for token_id, similarity in zip(general_indices, general_values):
        if token_id.item() == target_item["token_id"]:
            continue

        token = tokenizer.decode([token_id.item()])
        hu_liu_item = hu_liu_lookup_from_token_text(token)
        hu_liu_marker = (
            f"HuLiu:{hu_liu_item['sentiment']}:{hu_liu_item['word']}"
            if hu_liu_item is not None
            else ""
        )
        print(
            f"{repr(token):<18} "
            f"token_id={token_id.item():>5} "
            f"cosine={similarity.item():.4f} "
            f"{hu_liu_marker}"
        )
        shown += 1
        if shown == k:
            break

    hu_liu_similarities = F.cosine_similarity(target_vec, hu_liu_vectors, dim=1)
    hu_liu_values, hu_liu_indices = torch.topk(hu_liu_similarities, k + 1)

    print(f"\nNearest neighbors for {target_word!r} inside Hu & Liu:")
    print("-" * 72)

    shown = 0
    for source_index, similarity in zip(hu_liu_indices, hu_liu_values):
        neighbor = hu_liu_words[source_index.item()]
        if neighbor["word"] == target_word:
            continue

        print(
            f"{neighbor['word']:<16} "
            f"sentiment={neighbor['sentiment']:<8} "
            f"token_id={neighbor['token_id']:>5} "
            f"cosine={similarity.item():.4f}"
        )
        shown += 1
        if shown == k:
            break


for target_word in ["good", "bad", "delicious", "disgusting"]:
    nearest_neighbors_general_and_hu_liu(target_word)

def single_token_embedding_for_word(word):
    token_ids = tokenizer.encode(" " + word, add_special_tokens=False)
    if len(token_ids) != 1:
        raise ValueError(
            f"{word!r} must be exactly one token for embedding arithmetic, "
            f"but tokenizer.encode(' ' + {word!r}, add_special_tokens=False) returned {token_ids}."
        )

    token_id = token_ids[0]
    return embedding_matrix[token_id], token_id, tokenizer.decode([token_id])


def embedding_arithmetic_neighbors(expression, top_k=5):
    result_vector = torch.zeros(embedding_matrix.shape[1], dtype=embedding_matrix.dtype)
    operand_details = []

    for sign, word in expression:
        vector, token_id, token_text = single_token_embedding_for_word(word)
        result_vector = result_vector + sign * vector
        operand_details.append(
            {
                "sign": "+" if sign > 0 else "-",
                "word": word,
                "token": token_text,
                "token_id": token_id,
            }
        )

    similarities = F.cosine_similarity(result_vector.unsqueeze(0), embedding_matrix, dim=1)
    top_values, top_indices = torch.topk(similarities, top_k)
    expression_text = " ".join(
        f"{detail['sign']} {detail['word']}" if index > 0 else detail["word"]
        for index, detail in enumerate(operand_details)
    )

    print(f"\nEmbedding arithmetic: {expression_text}")
    print(f"Top-{top_k} full-vocabulary neighbors:")
    print("-" * 92)

    for rank, (token_id, similarity) in enumerate(zip(top_indices, top_values), start=1):
        token_id_int = token_id.item()
        token = tokenizer.decode([token_id_int])
        hu_liu_item = hu_liu_lookup_from_token_text(token)
        hu_liu_marker = (
            f"HuLiu:{hu_liu_item['sentiment']}:{hu_liu_item['word']}"
            if hu_liu_item is not None
            else ""
        )

        print(
            f"{rank:>2}. {repr(token):<18} "
            f"token_id={token_id_int:>5} "
            f"cosine={similarity.item():.4f} "
            f"{hu_liu_marker}"
        )


arithmetic_experiments = [
    [(+1, "good"), (+1, "excellent"), (-1, "bad")],
    [(+1, "terrible"), (+1, "sad"), (-1, "great")],
    [(+1, "excellent"), (-1, "good")],
    [(+1, "terrible"), (-1, "bad")],
    [(+1, "popular"), (-1, "unpopular")],
]

for experiment in arithmetic_experiments:
    embedding_arithmetic_neighbors(experiment)

positive_vector, _, _ = single_token_embedding_for_word("positive")
negative_vector, _, _ = single_token_embedding_for_word("negative")
sentiment_direction = positive_vector - negative_vector
sentiment_direction = sentiment_direction / sentiment_direction.norm()

if sentiment_pairs_by_field_path.exists():
    sentiment_pairs_by_field = json.loads(
        sentiment_pairs_by_field_path.read_text(encoding="utf-8")
    )
    projection_rows = []

    for field_index, field_data in enumerate(sentiment_pairs_by_field["fields"]):
        for pair in field_data["pairs"]:
            for sentiment in ["positive", "negative"]:
                item = pair[sentiment]
                vector = embedding_matrix[item["token_id"]]
                projection_score = torch.dot(vector, sentiment_direction).item()
                projection_rows.append(
                    {
                        "field": field_data["field"],
                        "field_label": field_data["field_label"],
                        "field_index": field_index,
                        "pair_id": pair["id"],
                        "word": item["word"],
                        "token": item["token"],
                        "token_id": item["token_id"],
                        "sentiment": item["sentiment"],
                        "projection_score": projection_score,
                    }
                )

    print("\nProjection onto sentiment direction: embedding('positive') - embedding('negative')")
    print("-" * 96)
    for row in sorted(projection_rows, key=lambda item: item["projection_score"], reverse=True):
        print(
            f"{row['word']:<16} "
            f"sentiment={row['sentiment']:<8} "
            f"field={row['field']:<18} "
            f"token_id={row['token_id']:>5} "
            f"projection={row['projection_score']:+.4f}"
        )

    fig, ax = plt.subplots(figsize=(12, 7.5))
    field_labels = [field["field_label"] for field in sentiment_pairs_by_field["fields"]]
    y_offsets = {"positive": 0.14, "negative": -0.14}
    marker_by_sentiment = {"positive": "o", "negative": "X"}
    color_by_sentiment = {"positive": "#2E8B57", "negative": "#B22222"}

    for sentiment in ["positive", "negative"]:
        group = [row for row in projection_rows if row["sentiment"] == sentiment]
        ax.scatter(
            [row["projection_score"] for row in group],
            [row["field_index"] + y_offsets[sentiment] for row in group],
            c=color_by_sentiment[sentiment],
            marker=marker_by_sentiment[sentiment],
            s=72,
            edgecolors="white",
            linewidths=0.8,
            alpha=0.9,
            label=sentiment,
            zorder=3,
        )

    for label_index, row in enumerate(projection_rows):
        x_offset = 5 if row["sentiment"] == "positive" else -5
        ax.annotate(
            row["word"],
            (row["projection_score"], row["field_index"] + y_offsets[row["sentiment"]]),
            xytext=(x_offset, 0),
            textcoords="offset points",
            ha="left" if x_offset > 0 else "right",
            va="center",
            fontsize=7.5,
            bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": "none", "alpha": 0.68},
        )

    ax.axvline(0, color="#404040", linewidth=1)
    ax.set_yticks(range(len(field_labels)))
    ax.set_yticklabels(field_labels)
    ax.invert_yaxis()
    ax.set_xlabel("Projection score onto normalized sentiment direction")
    ax.set_title("Hu & Liu word-pair projections onto embedding('positive') - embedding('negative')")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="x", alpha=0.22)
    plt.tight_layout()
    plt.savefig("sentiment_direction_projection_by_field.png", dpi=130)
    plt.close()

    print("Saved plot: sentiment_direction_projection_by_field.png")
else:
    print(f"Skipping sentiment direction plot: {sentiment_pairs_by_field_path} not found.")
