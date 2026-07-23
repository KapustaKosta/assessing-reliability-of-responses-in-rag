"""
Simple Claim-MIL style training using local CSV data.
Supports Faithfulness, Relevancy, and Reliability classification.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Setup paths
_SRC_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_SRC_DIR))


class SimpleMILClassifier(nn.Module):
    """Simple MIL-style classifier using max pooling over chunks."""

    def __init__(self, encoder_name: str, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        pooled, _ = hidden.max(dim=1)  # Max pooling
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits


class AnswerChunkDataset(Dataset):
    """Dataset for answer + chunks classification."""

    def __init__(self, df, tokenizer, max_length=512, label_col='binary_faithfulness'):
        self.samples = []
        self.labels = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_col = label_col

        for idx, row in df.iterrows():
            # Combine chunks
            chunks = []
            for i in range(1, 9):
                col = f'chunk_{i}'
                if col in row and pd.notna(row[col]):
                    chunks.append(str(row[col]))

            if not chunks:
                continue

            # Create text: question + answer + chunks
            question = str(row.get('question', '')) if pd.notna(row.get('question')) else ''
            answer = str(row['answer']) if pd.notna(row['answer']) else ''

            # Combine all into one text
            chunks_text = ' [SEP] '.join(chunks)
            text = f"{question} [SEP] {answer} [SEP] {chunks_text}"

            # Tokenize
            encoding = tokenizer(
                text,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            label = 1 if row[label_col] == True or row[label_col] == 1 else 0

            self.samples.append({
                'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze(),
            })
            self.labels.append(label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return {
            'input_ids': self.samples[idx]['input_ids'],
            'attention_mask': self.samples[idx]['attention_mask'],
            'label': torch.tensor(self.labels[idx], dtype=torch.float),
        }


def collate_fn(batch):
    return {
        'input_ids': torch.stack([b['input_ids'] for b in batch]),
        'attention_mask': torch.stack([b['attention_mask'] for b in batch]),
        'labels': torch.stack([b['label'] for b in batch]),
    }


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch in tqdm(loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device).unsqueeze(1).float()

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = (torch.sigmoid(logits) > 0.5).float()
        all_preds.extend(preds.cpu().numpy().flatten())
        all_labels.extend(labels.cpu().numpy().flatten())

    avg_loss = total_loss / len(loader)
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []

    for batch in tqdm(loader, desc="Evaluating"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device).unsqueeze(1).float()

        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)

        total_loss += loss.item()
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        all_probs.extend(probs.cpu().numpy().flatten())
        all_preds.extend(preds.cpu().numpy().flatten())
        all_labels.extend(labels.cpu().numpy().flatten())

    avg_loss = total_loss / len(loader)

    # Compute metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    accuracy = np.mean(all_preds == all_labels)
    tp = np.sum((all_preds == 1) & (all_labels == 1))
    fp = np.sum((all_preds == 1) & (all_labels == 0))
    fn = np.sum((all_preds == 0) & (all_labels == 1))
    tn = np.sum((all_preds == 0) & (all_labels == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Negative class metrics
    neg_precision = tn / (tn + fn) if (tn + fn) > 0 else 0
    neg_recall = tn / (tn + fp) if (tn + fp) > 0 else 0
    neg_f1 = 2 * neg_precision * neg_recall / (neg_precision + neg_recall) if (neg_precision + neg_recall) > 0 else 0

    macro_f1 = (f1 + neg_f1) / 2

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'positive_precision': precision,
        'positive_recall': recall,
        'positive_f1': f1,
        'negative_precision': neg_precision,
        'negative_recall': neg_recall,
        'negative_f1': neg_f1,
        'macro_f1': macro_f1,
        'confusion_matrix': [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='processed')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--label', type=str, default='faithfulness',
                       choices=['faithfulness', 'relevancy', 'reliability'])
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--max_train', type=int, default=500)
    parser.add_argument('--max_dev', type=int, default=100)
    parser.add_argument('--results_dir', type=str, required=True)
    parser.add_argument('--device', type=str, default='npu')
    args = parser.parse_args()

    # Setup device
    if args.device == 'npu' and torch.npu.is_available():
        device = torch.device('npu')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    logger.info(f"Device: {device}")

    # Label column mapping
    label_col_map = {
        'faithfulness': 'binary_faithfulness',
        'relevancy': 'binary_relevancy',
        'reliability': 'binary_reliability',
    }
    label_col = label_col_map[args.label]

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # Load data
    train_df = pd.read_csv(f'{args.data_dir}/train.csv')
    val_df = pd.read_csv(f'{args.data_dir}/val.csv')

    # Sample data
    if args.max_train:
        train_df = train_df.sample(n=min(args.max_train, len(train_df)), random_state=42)
    if args.max_dev:
        val_df = val_df.sample(n=min(args.max_dev, len(val_df)), random_state=42)

    logger.info(f"Train: {len(train_df)}, Dev: {len(val_df)}")

    # Label distribution
    train_pos = train_df[label_col].sum()
    train_neg = len(train_df) - train_pos
    logger.info(f"Train label distribution: pos={train_pos}, neg={train_neg}")

    # Create datasets
    train_dataset = AnswerChunkDataset(train_df, tokenizer, args.max_length, label_col)
    val_dataset = AnswerChunkDataset(val_df, tokenizer, args.max_length, label_col)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # Model
    model = SimpleMILClassifier(args.model_path).to(device)
    logger.info(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Loss with class weighting
    pos_weight = torch.tensor([train_neg / train_pos], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Training
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    best_macro_f1 = 0
    best_metrics = None

    for epoch in range(args.epochs):
        logger.info(f"\n=== Epoch {epoch + 1}/{args.epochs} ===")

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        logger.info(f"Train loss: {train_loss:.4f}, accuracy: {train_acc:.4f}")

        val_metrics = evaluate(model, val_loader, criterion, device)
        logger.info(f"Val loss: {val_metrics['loss']:.4f}")
        logger.info(f"Accuracy: {val_metrics['accuracy']:.4f}")
        logger.info(f"Macro F1: {val_metrics['macro_f1']:.4f}")
        logger.info(f"Positive F1: {val_metrics['positive_f1']:.4f}")
        logger.info(f"Confusion matrix: {val_metrics['confusion_matrix']}")

        # Save best
        if val_metrics['macro_f1'] > best_macro_f1:
            best_macro_f1 = val_metrics['macro_f1']
            best_metrics = val_metrics
            torch.save(model.state_dict(), results_dir / 'best_model.pt')
            logger.info(f"New best macro F1: {best_macro_f1:.4f}")

    # Final metrics
    logger.info(f"\n=== Final Results ===")
    logger.info(f"Best Macro F1: {best_macro_f1:.4f}")
    logger.info(f"Metrics: {best_metrics}")

    # Save results
    results = {
        'label': args.label,
        'best_macro_f1': best_macro_f1,
        'best_metrics': best_metrics,
        'train_size': len(train_df),
        'dev_size': len(val_df),
    }
    with open(results_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {results_dir}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
