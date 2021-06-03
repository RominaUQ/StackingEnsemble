import argparse
import joblib
import os
import json
import numpy as np
import pandas as pd
import logging
import sys
import pickle
from my_custom_library import cross_validation

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))


def model_fn(model_dir):
    """Deserialize and return fitted model.
    Note that this should have the same name as the serialized model in the _randomforest_train method
    """
    model_file = "randomforest-model"
    model = pickle.load(open(os.path.join(model_dir, model_file), "rb"))
    return model


def parse_args():
    """
    Parse arguments passed from the SageMaker API
    to the container
    """

    parser = argparse.ArgumentParser()

    # Hyperparameters sent by the client are passed as command-line arguments to the script
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--K", type=int, default=os.environ.get("SM_HP_K"))

    # Data, model, and output directories
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))

    return parser.parse_known_args()


def train():
    """
    Train the PyTorch model
    """

    K = args.K

    hyperparameters = {
        "max_depth": args.max_depth,
       # "eta": args.eta,
       # "objective": args.objective,
       # "num_round": args.num_round,
    }

    train_df = pd.read_csv(f"{args.train}/meta_train.csv", header=None)

    score_list, model = cross_validation(train_df, K, hyperparameters)
    k_fold_avg = sum(score_list) / len(score_list)
    print(f" average accuracy across folds: {k_fold_avg}")

    model_location = args.model_dir + "/randomforest-model"
    pickle.dump(model, open(model_location, "wb"))
    logging.info("Stored trained model at {}".format(model_location))

  
    

if __name__ == "__main__":

    args, _ = parse_args()
    train()