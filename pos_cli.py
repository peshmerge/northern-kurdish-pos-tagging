import argparse
from pos_model import POSModel


def main() -> None:
    """ Main function to run the Northern Kurdish POS Tagger using CLI"""
    parser = argparse.ArgumentParser(
        description="Command line interface for Northern Kurdish POS tagging."
    )

    parser.add_argument(
        "--pos_model",
        type=str,
        default="Baseline",
        choices=["All", "Baseline", "HMM", "AveragedPerceptron", "CRF", "ExtraTrees", "BiLSTM", "NK-XLMR"],
        help="Name of the POS model to use."
    )

    parser.add_argument(
        "--training_data_type",
        type=str,
        default="augmented",
        choices=["augmented", "original"],
        help="Type of training data used for training the model."
    )

    parser.add_argument(
        "--sentence",
        type=str,
        required=True,
        help="The sentence to perform POS tagging on."
    )

    parser.add_argument(
        "--tokenization_method",
        type=str,
        default="KLPT",
        choices=["KLPT", "manual", "NLTK"],
        help="The tokenization method to use. Use manual in case you manually tokenize the sentence by splitting off "
             "the IZAFE, Oblique and indefinite case markers from the nouns."
    )

    args = parser.parse_args()
    if args.pos_model == 'All':
        for model_name in POSModel.POS_MODELS:
            pos_model = POSModel(model_name, args.training_data_type)
            pos_model.load_pos_model()
            print(f"{model_name} :")
            print(pos_model.predict_pos_tags(args.sentence, args.tokenization_method))
    else:
        pos_model = POSModel(args.pos_model, args.training_data_type)
        pos_model.load_pos_model()
        print(pos_model.predict_pos_tags(args.sentence, args.tokenization_method))


if __name__ == "__main__":
    main()
