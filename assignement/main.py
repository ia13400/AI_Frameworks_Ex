from linear_probe import run_logistic_regression_probe
from model_utils import (
    analyze_parameters,
    load_model_and_tokenizer,
    print_architecture_overview,
    run_top_k_predictions,
    setup_environment,
)
from sentiment_embeddings import (
    run_positive_negative_comparison,
    run_token_embedding_visualization,
)
from sst2_analysis import run_sst2_inspection


def main() -> None:
    device = setup_environment()
    model, tokenizer = load_model_and_tokenizer(device)
    print_architecture_overview(model)
    analyze_parameters(model)
    run_top_k_predictions(model, tokenizer, device)

    sentiment_state = run_positive_negative_comparison(model, tokenizer)
    run_token_embedding_visualization(tokenizer, sentiment_state)
    run_logistic_regression_probe(sentiment_state)
    run_sst2_inspection()


if __name__ == "__main__":
    main()
