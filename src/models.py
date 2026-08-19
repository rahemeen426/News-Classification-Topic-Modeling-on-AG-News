from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from datasets import Dataset
import numpy as np
import pickle
import os


def train_svm_model(df):
    """
    Trains a TF-IDF + LinearSVC model on the provided dataframe.
    df should have 'text' and 'label' columns.
    """
    print("Training SVM Model...")
    text_clf = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_features=10000)),
        ('clf', LinearSVC(random_state=42)),
    ])
    text_clf.fit(df['text'], df['label'])
    print("SVM Model Trained.")
    return text_clf


def save_model(model, path="models/svm_model.pkl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(model, f)


def load_svm_model(path="models/svm_model.pkl"):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None


class RuleBasedClassifier:
    def __init__(self):
        # High-precision keywords for AG News classes
        self.rules = {
            'Sports': ['olympic', 'football', 'soccer', 'nba', 'nhl', 'baseball', 'cricket', 'athlete', 'medal', 'championship'],
            'Sci/Tech': ['software', 'hardware', 'microsoft', 'google', 'apple', 'linux', 'processor', 'nasa', 'orbit', 'virus'],
            'Business': ['oil', 'prices', 'stocks', 'nasdaq', 'investor', 'economy', 'profit', 'merger', 'acquisition', 'market'],
            'World': ['prime minister', 'president', 'iraq', 'israel', 'war', 'troops', 'bomb', 'treaty', 'united nations', 'parliament']
        }
        self.rev_map = {'World': 0, 'Sports': 1, 'Business': 2, 'Sci/Tech': 3}

    def predict_single(self, text):
        text_lower = text.lower()
        for label, keywords in self.rules.items():
            for word in keywords:
                if f" {word} " in f" {text_lower} ": # weak boundary check
                    return self.rev_map[label]
        return None


class HybridClassifier:
    def __init__(self, svm_model, rule_model=None):
        self.svm = svm_model
        self.rule_model = rule_model if rule_model else RuleBasedClassifier()

    def predict(self, texts):
        # Handles list or single string
        if isinstance(texts, str):
            texts = [texts]

        predictions = []
        for text in texts:
            # 1. Rule Check
            rule_pred = self.rule_model.predict_single(text)
            if rule_pred is not None:
                predictions.append(rule_pred)
            else:
                # 2. Fallback to SVM
                predictions.append(self.svm.predict([text])[0])
        return predictions


class TransformerClassifier:
    def __init__(
        self,
        model_name="fabriceyhc/bert-base-uncased-ag_news",
        local_model_path=None,
    ):
        if local_model_path and os.path.isdir(local_model_path):
            load_source = local_model_path
        else:
            load_source = model_name

        print(f"Loading Transformer: {load_source}...")
        self.pipe = pipeline("text-classification", model=load_source, tokenizer=load_source)
        self.label_map = {
            'World': 0, 'Sports': 1, 'Business': 2, 'Sci/Tech': 3,
            'LABEL_0': 0, 'LABEL_1': 1, 'LABEL_2': 2, 'LABEL_3': 3,
            'world': 0, 'sports': 1, 'business': 2, 'sci/tech': 3
        }

    def predict(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        results = self.pipe(texts, truncation=True, max_length=512)

        predictions = []
        for res in results:
            label_str = res['label']
            predictions.append(self.label_map.get(label_str, 0))
        return predictions

    def save_pretrained(self, output_dir="models/transformer"):
        os.makedirs(output_dir, exist_ok=True)
        self.pipe.model.save_pretrained(output_dir)
        self.pipe.tokenizer.save_pretrained(output_dir)


def fine_tune_transformer_model(
    train_df,
    val_df=None,
    model_name="bert-base-uncased",
    output_dir="models/transformer",
    epochs=3,
    batch_size=8,
     learning_rate=2e-5,
    pretrained=True,
    freeze_encoder=False,
    max_length=128,
    seed=42,
):
    """
    Fine-tunes a transformer model for AG News classification.

    - If pretrained=True, the encoder weights are loaded from the specified model.
    - If pretrained=False, the model is initialized from config and trained from scratch.
    """
    os.makedirs(output_dir, exist_ok=True)

    if val_df is None:
        train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=seed, stratify=train_df['label'])

    train_ds = Dataset.from_pandas(train_df[['text', 'label']].reset_index(drop=True))
    val_ds = Dataset.from_pandas(val_df[['text', 'label']].reset_index(drop=True))

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if pretrained:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4)
    else:
        config = AutoConfig.from_pretrained(model_name, num_labels=4)
        model = AutoModelForSequenceClassification.from_config(config)

    if freeze_encoder:
        for param in model.base_model.parameters():
            param.requires_grad = False

    def preprocess(batch):
        return tokenizer(batch['text'], truncation=True, max_length=max_length)

    train_ds = train_ds.map(preprocess, batched=True, remove_columns=['text'])
    val_ds = val_ds.map(preprocess, batched=True, remove_columns=['text'])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {
            'accuracy': (predictions == labels).astype(np.float32).mean().item(),
        }

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        seed=seed,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=50,
        save_total_limit=2,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Fine-tuned model saved to {output_dir}")
    return TransformerClassifier(local_model_path=output_dir)


def load_transformer_model(
    local_model_path="models/transformer",
    fallback_model_name="fabriceyhc/bert-base-uncased-ag_news",
):
    if os.path.isdir(local_model_path):
        return TransformerClassifier(local_model_path=local_model_path)
    return TransformerClassifier(model_name=fallback_model_name)
