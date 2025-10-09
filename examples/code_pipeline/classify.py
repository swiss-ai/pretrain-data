import argparse
import os
import glob

from tqdm import tqdm
from classifiers.classifier import HFClassifier
from classifiers.fasttexter import FastTextClassifier
from classifiers.regressor import HFRegressor
from utils import fasttext_create_dataset, fasttext_split_dataset
        

def train(model_type, columns, model_path, data_path):
    if model_type == "regressor":
        classifier = HFRegressor("microsoft/codebert-base", columns, column_num_classes, os.path.join(model_path, "regressor"))
        classifier.train(data_path)


def annotate(model_type, columns, model_path, input_data_path, output_data_path, checkpoint_name):
    if model_type == "regressor":
        classifier = HFRegressor("microsoft/codebert-base", columns, column_num_classes, os.path.join(model_path, "regressor"), checkpoint_name)
    classifier.annotate(input_data_path, output_data_path)

if __name__ == "__main__":
    columns=['clarity', 'practice', 'educational']
    column_num_classes=10

    # arguments
    main_parser = argparse.ArgumentParser()
    main_parser.add_argument('--task', type=str, required=True)
    main_args, remaining_args = main_parser.parse_known_args()


    if main_args.task == 'train':
        # Specific Args
        parser = argparse.ArgumentParser()
        parser.add_argument('--model', type=str)
        parser.add_argument('--data_path', type=str)
        parser.add_argument('--model_path', type=str)
        args = parser.parse_args(remaining_args)
        train(args.model, columns, args.model_path, args.data_path)

    elif main_args.task == 'annotate':
        # Specific Args
        parser = argparse.ArgumentParser()
        parser.add_argument('--model', type=str)
        parser.add_argument('--input_data_path', type=str)
        parser.add_argument('--output_data_path', type=str)
        parser.add_argument('--model_path', type=str)
        parser.add_argument('--checkpoint_name', type=str)
        args = parser.parse_args(remaining_args)
        annotate(args.model, columns, args.model_path, args.input_data_path, args.output_data_path, args.checkpoint_name)

    else:
        print('Task not known')