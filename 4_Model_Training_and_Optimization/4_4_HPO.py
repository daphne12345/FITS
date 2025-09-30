# -*- coding: utf-8 -*-
from __future__ import annotations

import multiprocessing
from argparse import ArgumentParser
import numpy as np
import torch
import torch.optim
from torchmetrics.classification import F1Score, MultilabelAccuracy, MultilabelPrecision, MultilabelRecall
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, EvalPrediction
import transformers
from DataPreparation import prepare_data, tokenize
from smac import Scenario
from smac.utils.logging import get_logger
from transformers.utils.logging import disable_progress_bar
from ConfigSpace import Categorical, Configuration, ConfigurationSpace, Float, Integer
from sklearn.metrics import classification_report
from smac import HyperparameterOptimizationFacade as HPOFacade
import random 

disable_progress_bar()

logger = get_logger(__name__)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class CustomTrainer(Trainer):
    """
    Custom Trainer for multi-label classification.
    :param num_labels: The number of labels in the classification task (default: 19)
    """
    def __init__(self, *args, num_labels=19, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_labels = num_labels

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Computes the loss for the model.

        :param model: The model instance.
        :param inputs: Input data.
        :param return_outputs: Whether to return model outputs.
        :type return_outputs: bool
        :return: Computed loss value.
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fn = torch.nn.BCEWithLogitsLoss()
        loss = loss_fn(logits, labels.float())
        return (loss, outputs) if return_outputs else loss


class MultiLabelModel:
    """
    A model for multi-label classification using transformers, with hyperparameter tuning support.

    :param model_name: Name of the model to use.
    :param seed: Random seed for reproducibility.
    """

    def __init__(self, model_name, seed=0):
        self.model_name = model_name
        self.threshold = 0.3
        df, df_train, df_val, df_test = prepare_data()
        self.tokenizer, self.data_collator, self.train_dataset, self.val_dataset, self.test_dataset, self.id2label, self.label2id = tokenize(
            df, df_train, df_val, df_test, model_name)
        self.num_labels = 19
        self.seed = seed

    @property
    def configspace(self) -> ConfigurationSpace:
        """
        Returns the configuration space for hyperparameter optimization.

        :return: The configuration space object containing hyperparameters and their ranges
        """
        cs = ConfigurationSpace(seed=self.seed)
        cs.add([
            Float("learning_rate", (1e-6, 0.01), default=0.00001, log=True),
            Float("weight_decay", (0.0001, 0.3), default=0.1, log=True),
            Categorical("lr_scheduler_type", ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant",
                                              "constant_with_warmup", "inverse_sqrt", "reduce_lr_on_plateau"], default='linear'),
            Float("warmup_ratio", (0.0001, 0.1), default=0.01, log=True),
            Float("label_smoothing_factor", (0.0001, 0.1), default=0.01, log=True),
            Integer("threshold", (30, 60), default=50, log=False),
            Integer("epochs", (15, 35), default=25, log=False),
        ])
        return cs

    def model_init(self, config):
        """
        Initializes and returns the transformer model based on the given configuration.

        :param config: The configuration for initializing the model
        :return: The transformer model
        """
        return AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            config=config,
            problem_type="multi_label_classification",
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id=self.label2id
        )

    def compute_metrics(self, p: EvalPrediction):
        """
        Computes evaluation metrics for multi-label classification, including F1 scores, accuracy, precision, and recall.

        :param p: The prediction object containing predictions and label_ids
        :return: A dictionary containing the computed metrics (F1 scores, accuracy, precision, recall)
        """
        predictions = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(torch.Tensor(predictions))

        y_pred = np.zeros(probs.shape)
        y_pred[np.where(probs >= self.threshold)] = 1

        f1_micro_metric = F1Score(num_labels=self.num_labels, task='multilabel', average='micro')
        f1_micro = f1_micro_metric(torch.tensor(y_pred), torch.tensor(p.label_ids))

        f1_macro_metric = F1Score(num_labels=self.num_labels, task='multilabel', average='macro')
        f1_macro = f1_macro_metric(torch.tensor(y_pred), torch.tensor(p.label_ids))

        f1_weighted_metric = F1Score(num_labels=self.num_labels, task='multilabel', average='weighted')
        f1_weighted = f1_weighted_metric(torch.tensor(y_pred), torch.tensor(p.label_ids))

        acc_metric = MultilabelAccuracy(num_labels=self.num_labels, average='weighted')
        accuracy = acc_metric(torch.tensor(y_pred), torch.tensor(p.label_ids))

        precision_metric = MultilabelPrecision(num_labels=self.num_labels, average='weighted')
        precision = precision_metric(torch.tensor(y_pred), torch.tensor(p.label_ids))

        recall_metric = MultilabelRecall(num_labels=self.num_labels, average='weighted')
        recall = recall_metric(torch.tensor(y_pred), torch.tensor(p.label_ids))

        print(classification_report(p.label_ids, y_pred))

        return {'f1_weighted': f1_weighted, 'f1_micro': f1_micro, 'f1_macro': f1_macro, 'accuracy': accuracy,
                'precision': precision, 'recall': recall}

    def train(self, config: Configuration, seed: int = 0):
        """
        Trains the multi-label model with the specified hyperparameters.

        :param config: The configuration containing hyperparameters for training
        :param seed: The random seed for reproducibility (default: 0)
        :return: The negative weighted F1 score for optimization (1 - f1_weighted)
        """
        print(config)
        training_args = TrainingArguments(
            output_dir='results/' + self.model_name + '/',
            do_train=True,
            eval_strategy="epoch",
            learning_rate=config['learning_rate'],
            auto_find_batch_size=True,
            num_train_epochs=config['epochs'],
            weight_decay=config['weight_decay'],
            lr_scheduler_type=config['lr_scheduler_type'],
            warmup_ratio=config['warmup_ratio'],
            label_smoothing_factor=config['label_smoothing_factor'],
            optim='adamw_hf',
            group_by_length=True,
            save_strategy="no",
            push_to_hub=False,
            seed=seed,
            do_predict=False,
            metric_for_best_model='f1_weighted',
            greater_is_better=True,
        )

        self.threshold = config['threshold'] / 100

        trainer = CustomTrainer(
            model=None,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            compute_metrics=self.compute_metrics,
            tokenizer=self.tokenizer,
            model_init=self.model_init,
            data_collator=self.data_collator,
            num_labels=self.num_labels
        )
        trainer.train()
        res = trainer.evaluate(self.val_dataset)
        print(res)
        return 1 - res['eval_f1_weighted']  # Minimize the negative F1 score

    def evaluate(self, config: Configuration, seed: int = 0):
        """
        Evaluates the model using the given configuration.

        :param config: The configuration containing hyperparameters for evaluation
        :param seed: The random seed for reproducibility (default: 0)
        :return: The evaluation results on the test dataset
        """
        training_args = TrainingArguments(
            output_dir='results/' + self.model_name + '/',
            do_train=True,
            eval_strategy="epoch",
            learning_rate=config['learning_rate'],
            auto_find_batch_size=True,
            num_train_epochs=config['epochs'],
            weight_decay=config['weight_decay'],
            lr_scheduler_type=config['lr_scheduler_type'],
            warmup_ratio=config['warmup_ratio'],
            label_smoothing_factor=config['label_smoothing_factor'],
            optim='adamw_hf',
            group_by_length=True,
            save_strategy="no",
            push_to_hub=False,
            seed=seed,
            do_predict=False,
            metric_for_best_model='f1_weighted',
            greater_is_better=True,
        )
        self.threshold = config['threshold'] / 100

        trainer = CustomTrainer(
            model=None,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.test_dataset,
            compute_metrics=self.compute_metrics,
            tokenizer=self.tokenizer,
            model_init=self.model_init,
            data_collator=self.data_collator,
            num_labels=self.num_labels
        )
        trainer.train()
        res = trainer.evaluate(self.test_dataset)
        print(self.model_name)
        print('Test results')
        print(res)
        return res


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    parser = ArgumentParser()
    parser.add_argument('-m', '--model_name', default='roberta-base') # google-bert/bert-base-cased, distilbert-base-cased, allenai/scibert_scivocab_cased, climatebert/distilroberta-base-climate-detector
    parser.add_argument('-s', '--seed', default=0, type=int)
    args = parser.parse_args()

    logger.info(f'CUDA availability: {torch.cuda.is_available()}')
    logger.info(f'Arguments: {args}')
    print(args.model_name)
    print('seed', args.seed)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    transformers.set_seed(args.seed)

    tuner = MultiLabelModel(model_name=args.model_name, seed=args.seed)

    scenario = Scenario(
        tuner.configspace,
        n_trials=1000,
        deterministic=True,
        name=args.model_name,
        seed=args.seed
    )

    initial_design = HPOFacade.get_initial_design(scenario, n_configs=5)

    smac = HPOFacade(
        scenario,
        tuner.train,
        initial_design=initial_design,
        overwrite=True
    )

    incumbent = smac.optimize()
    print(incumbent)

    tuner.evaluate(config=incumbent)
