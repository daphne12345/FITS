from __future__ import annotations
import multiprocessing
from argparse import ArgumentParser
import numpy as np
import torch
import torch.optim
from torchmetrics.classification import F1Score, MultilabelAccuracy, MultilabelPrecision, MultilabelRecall
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, EvalPrediction
import transformers
from Data_Preparation import prepare_data, tokenize
from transformers.utils.logging import disable_progress_bar
from sklearn.metrics import classification_report
import random 

disable_progress_bar()


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
    Multi-label classification model wrapper.

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
    
    def train(self):
        """
        Trains the model.
        """
        training_args = TrainingArguments(
            output_dir='results/' + self.model_name + '/',
            do_train=True,
            eval_strategy="epoch",
            auto_find_batch_size=True,
            num_train_epochs=20,
            optim='adamw_hf',
            group_by_length=True,
            save_strategy="no",
            push_to_hub=False,
            seed=self.seed,
            do_predict=False,
            metric_for_best_model='f1_weighted',
            greater_is_better=True,
        )


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
        print('Test results')
        print(res)



if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    parser = ArgumentParser()
    parser.add_argument('-m', '--model_name', default='google-bert/bert-base-cased') # roberta-base, distilbert-base-cased, allenai/scibert_scivocab_cased, climatebert/distilroberta-base-climate-detector
    parser.add_argument('-s', '--seed', default=0, type=int)
    args = parser.parse_args()
    print(args.model_name)
    print('seed', args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    transformers.set_seed(args.seed)


    model = MultiLabelModel(model_name=args.model_name, seed=args.seed)
    model.train()


