from nltk.tag import CRFTagger
from utils import *
from nltk.tag.perceptron import PerceptronTagger
from flair.models import SequenceTagger
from trankit import Pipeline

import dill
import os


class POSModel:
    """Class to load different POS tagging models and performing POS Tagging for Northern Kurdish """
    POS_MODELS_DIR = "models"
    POS_MODELS = ["Baseline", "HMM", "ExtraTrees",  "AveragedPerceptron",  "BiLSTM", "CRF", "NK-XLMR"]

    def __init__(self, model_type, training_data_type) -> None:
        self.model_type = model_type
        self.training_data_type = training_data_type
        self.model = None
        self.model_path = None

    def load_pos_model(self) -> None:
        """Load the specified POS model based on the user input"""
        if self.model_type == "Baseline":
            self.model_path = os.path.join(self.POS_MODELS_DIR, self.model_type,
                                           f"KMR_POS_{self.model_type}_Model_{self.training_data_type}.pickle")
            self.model = load_data_from_pickle_file(self.model_path)

        elif self.model_type == "HMM":
            self.model_path = os.path.join(self.POS_MODELS_DIR, self.model_type,
                                           f"KMR_POS_{self.model_type}_Model_{self.training_data_type}.pickle")
            with open(self.model_path, 'rb') as f:
                self.model = dill.load(f)

        elif self.model_type == "AveragedPerceptron":
            self.model_path = os.path.join(self.POS_MODELS_DIR, self.model_type,
                                           f"KMR_POS_{self.model_type}_Model_{self.training_data_type}.pickle")
            self.model = PerceptronTagger(load=False)
            self.model.load(self.model_path)

        elif self.model_type == "CRF":
            self.model_path = os.path.join(self.POS_MODELS_DIR, self.model_type,
                                           f"KMR_POS_{self.model_type}_Model_{self.training_data_type}.pickle")
            self.model = CRFTagger()
            self.model.set_model_file(self.model_path)

        elif self.model_type == "ExtraTrees":
            model_name_compressed = os.path.join(self.POS_MODELS_DIR, self.model_type,
                                                 f"KMR_POS_{self.model_type}_Model_{self.training_data_type}.zip")
            self.model_path = os.path.join(self.POS_MODELS_DIR, self.model_type,
                                           f"KMR_POS_{self.model_type}_Model_{self.training_data_type}")
            unzip_extra_trees_pos_model(self.model_path, model_name_compressed,
                                        os.path.join(self.POS_MODELS_DIR, self.model_type))
            self.model = load_data_from_pickle_file(f"{self.model_path}.pickle")

        elif self.model_type == "BiLSTM":
            self.model_path = os.path.join(self.POS_MODELS_DIR, self.model_type,
                                           f"KMR_POS_{self.model_type}_Model_{self.training_data_type}.pt")
            self.model = SequenceTagger.load(os.path.join(self.model_path))

        elif self.model_type == "NK-XLMR":
            self.model_path = os.path.join(self.POS_MODELS_DIR, self.model_type,
                                           f"KMR_POS_{self.model_type}_Model_{self.training_data_type}")
            self.model = Pipeline(lang='customized', gpu=False, cache_dir=self.model_path)

    def predict_pos_tags(self, sentence, tokenization_method="manual"):
        """ Function to predict POS tags for a given sentence """
        predicted_tags = []
        # If the sentence is already tokenized
        if tokenization_method == 'manual':
            tokens = sentence.split(" ")
        else:
            tokens = tokenize_sentence(sentence, tokenization_method)
        if self.model_type in ("AveragedPerceptron", "CRF", "HMM", "Baseline"):
            predicted_tags = [tag[1] for tag in self.model.tag(tokens)]
        elif self.model_type == 'ExtraTrees':
            null_tags = [(token, 'NULL') for token in tokens]
            tokens_features = extract_features([null_tags])
            extra_trees_vectoriser = load_extra_trees_vectoriser(
                os.path.join(self.POS_MODELS_DIR, self.model_type), self.training_data_type)
            tokens_x = extra_trees_vectoriser.transform(tokens_features)
            predicted_tags = self.model.predict(tokens_x)
        elif self.model_type == 'hmmlearn':
            predicted_tags = self.model.predict(tokens)
        elif self.model_type == 'NK-XLMR':
            predictions = self.model.posdep(tokens, is_sent=True)
            predicted_tags = [token.get('upos') for token in predictions['tokens']]

        elif self.model_type == 'BiLSTM':
            tokens_tags_list = predict_pos_tags_using_flair_model(self.model, tokens)
            predicted_tags = [tag[1] for tag in tokens_tags_list]
        return [
            (token, REPLACEMENT_SYMBOL) if
            token == REPLACEMENT_SYMBOL else
            (token, tag) for token, tag in list(zip(tokens, predicted_tags))]
