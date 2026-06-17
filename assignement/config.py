from pathlib import Path


MODEL_NAME = "EleutherAI/pythia-410m"
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR.parent

POSITIVE_WORDS_PATH = SCRIPT_DIR / "hu_liu_positive_one_token_words.json"
NEGATIVE_WORDS_PATH = SCRIPT_DIR / "hu_liu_negative_one_token_words.json"
FIELD_PAIRS_PATH = SCRIPT_DIR / "hu_liu_sentiment_word_pairs_by_field.json"
FLAT_PAIRS_PATH = SCRIPT_DIR / "hu_liu_sentiment_word_pairs_flat.json"
PROBE_SPLIT_PATH = SCRIPT_DIR / "hu_liu_logistic_regression_split.json"

EXAMPLE_PROMPTS = [
    "The food was delicious and the service was",
    "The food was disgusting and the service was",
    "I love this movie and I feel very",
    "I hate this movie and I feel very",
]

NEIGHBOR_TARGET_WORDS = ["good", "bad", "delicious", "disgusting"]

ARITHMETIC_EXPERIMENTS = [
    "good + excellent - bad",
    "terrible + sad - great",
    "excellent - good",
    "terrible - bad",
    "popular - unpopular",
]

PROJECTION_FIELDS_TO_KEEP = {
    "food_taste",
    "product_quality",
    "appearance_design",
}
