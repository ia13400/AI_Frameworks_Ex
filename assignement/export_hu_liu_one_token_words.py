import json
import re
from pathlib import Path

import nltk
from nltk.corpus import opinion_lexicon
from transformers import AutoTokenizer


MODEL_NAME = "EleutherAI/pythia-410m"
OUTPUT_DIR = Path(__file__).resolve().parent
NLTK_DATA_DIR = OUTPUT_DIR.parent / "nltk_data"


def load_hu_liu_opinion_lexicon():
    nltk.data.path.append(str(NLTK_DATA_DIR.resolve()))

    try:
        positive = opinion_lexicon.positive()
        negative = opinion_lexicon.negative()
    except LookupError:
        NLTK_DATA_DIR.mkdir(exist_ok=True)
        nltk.download("opinion_lexicon", download_dir=str(NLTK_DATA_DIR), quiet=True)
        positive = opinion_lexicon.positive()
        negative = opinion_lexicon.negative()

    return positive, negative


def is_plain_word(word):
    return re.fullmatch(r"[A-Za-z]+", word) is not None


def single_token_for_running_text(tokenizer, word):
    token_ids = tokenizer.encode(" " + word, add_special_tokens=False)
    if len(token_ids) != 1:
        return None

    return token_ids[0]


def build_entries(tokenizer, words, sentiment):
    entries = []

    for word in words:
        word = word.lower()
        if not is_plain_word(word):
            continue

        token_id = single_token_for_running_text(tokenizer, word)
        if token_id is None:
            continue

        entries.append(
            {
                "word": word,
                "token": tokenizer.decode([token_id]),
                "token_id": token_id,
                "sentiment": sentiment,
            }
        )

    return sorted(entries, key=lambda item: item["word"])


def write_json(items, sentiment, output_path):
    payload = {
        "_comment": (
            "Valid JSON does not support comments, so source and filtering notes "
            "are stored as metadata fields."
        ),
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
        "words": items,
    }

    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    positive_words, negative_words = load_hu_liu_opinion_lexicon()

    positive_entries = build_entries(tokenizer, positive_words, "positive")
    negative_entries = build_entries(tokenizer, negative_words, "negative")

    positive_path = OUTPUT_DIR / "hu_liu_positive_one_token_words.json"
    negative_path = OUTPUT_DIR / "hu_liu_negative_one_token_words.json"

    write_json(positive_entries, "positive", positive_path)
    write_json(negative_entries, "negative", negative_path)

    print(f"Saved {len(positive_entries)} positive words to {positive_path}")
    print(f"Saved {len(negative_entries)} negative words to {negative_path}")


if __name__ == "__main__":
    main()
