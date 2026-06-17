import json
import re
from pathlib import Path

import nltk
import torch
from nltk.corpus import opinion_lexicon

from config import MODEL_NAME, NEGATIVE_WORDS_PATH, POSITIVE_WORDS_PATH


def load_hu_liu_opinion_lexicon():
    """Load the Hu & Liu Opinion Lexicon through NLTK."""
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


def is_plain_word(word: str) -> bool:
    """Keep only alphabetic words so punctuation and phrases do not enter the analysis."""
    return re.fullmatch(r"[A-Za-z]+", word) is not None


def single_token_for_running_text(word: str, tokenizer):
    """Return the token id if the word is represented by one token in running text."""
    token_ids = tokenizer.encode(" " + word, add_special_tokens=False)
    if len(token_ids) != 1:
        return None
    return token_ids[0]


def build_sentiment_word_records(tokenizer):
    """Create Hu & Liu word records that are represented by exactly one model token."""
    positive_lexicon, negative_lexicon = load_hu_liu_opinion_lexicon()
    sentiment_words = []

    for sentiment, lexicon_words in [
        ("positive", positive_lexicon),
        ("negative", negative_lexicon),
    ]:
        for word in lexicon_words:
            word = word.lower()
            if not is_plain_word(word):
                continue

            token_id = single_token_for_running_text(word, tokenizer)
            if token_id is None:
                continue

            sentiment_words.append(
                {
                    "word": word,
                    "sentiment": sentiment,
                    "token_id": int(token_id),
                    "token": tokenizer.decode([token_id]),
                }
            )

    if len(sentiment_words) < 2:
        raise ValueError("Not enough one-token Hu & Liu opinion words found.")

    return sentiment_words


def export_sentiment_words_json(items, sentiment: str, output_path: Path) -> None:
    """Write one-token sentiment words together with source and filtering metadata."""
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
                "token_id": int(item["token_id"]),
                "sentiment": item["sentiment"],
            }
            for item in sorted(items, key=lambda item: item["word"])
        ],
    }

    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def prepare_hu_liu_sentiment_words(tokenizer, embedding_matrix):
    """Build sentiment records, export JSON files, and return vectors for plotting."""
    sentiment_words = build_sentiment_word_records(tokenizer)
    positive_words = [item for item in sentiment_words if item["sentiment"] == "positive"]
    negative_words = [item for item in sentiment_words if item["sentiment"] == "negative"]

    export_sentiment_words_json(positive_words, "positive", POSITIVE_WORDS_PATH)
    export_sentiment_words_json(negative_words, "negative", NEGATIVE_WORDS_PATH)

    token_ids = torch.tensor([item["token_id"] for item in sentiment_words], dtype=torch.long)
    vectors = embedding_matrix[token_ids].float().numpy()

    print(f"\nSaved JSON: {POSITIVE_WORDS_PATH}")
    print(f"Saved JSON: {NEGATIVE_WORDS_PATH}")
    print(f"\nHu & Liu words kept:       {len(sentiment_words)}")
    print(f"Positive one-token words:  {len(positive_words)}")
    print(f"Negative one-token words:  {len(negative_words)}")

    return sentiment_words, positive_words, negative_words, vectors


def normalized_word_key(word: str) -> str:
    """Normalize only for lookup; the original word is kept for printing."""
    return "".join(word.split()).lower()


def build_hu_liu_lookup(positive_words, negative_words):
    lookup = {}
    for item in positive_words + negative_words:
        lookup[normalized_word_key(item["word"])] = item["sentiment"]
    return lookup
