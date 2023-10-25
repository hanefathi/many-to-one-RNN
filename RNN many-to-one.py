import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import spacy
import torch
import torch.nn.functional as F
import torchtext
import time
import random
import pandas as pd

torch.backends.cudnn.deterministic = True

RANDOM_SEED = 123
torch.manual_seed(RANDOM_SEED)

VOCABULARY_SIZE = 4000
LEARNING_RATE = 0.005
BATCH_SIZE = 128
NUM_EPOCHS =20
DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

EMBEDDING_DIM = 128
HIDDEN_DIM = 100
NUM_CLASSES = 5


df = pd.read_csv('train_dataset4.csv')




nlp = spacy.load('en_core_web_sm')

TEXT = torchtext.data.Field(
    tokenize='spacy',  # default splits on whitespace
    tokenizer_language='en_core_web_sm',
    include_lengths=True  # NEW
)


### Defining the label processing

LABEL = torchtext.data.LabelField(dtype=torch.long)

fields = [('sentence', TEXT), ('relation_type', LABEL)]

dataset = torchtext.data.TabularDataset(
    path='train_dataset4.csv', format='csv',
    skip_header=True, fields=fields)


random.seed(RANDOM_SEED)

# First split: train and test
train_data, test_data = dataset.split(
    split_ratio=[0.8, 0.2],
    random_state=random.seed(RANDOM_SEED)
)

# Set the random seed again before the second split
random.seed(RANDOM_SEED)

# Second split: train and validation
train_data, valid_data = train_data.split(
    split_ratio=[0.85, 0.15],
    random_state=random.seed(RANDOM_SEED)
)

print(f'Num Train: {len(train_data)}')
print(f'Num Validation: {len(valid_data)}')
print(vars(train_data.examples[0]))




train_loader, valid_loader, test_loader = \
    torchtext.data.BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=BATCH_SIZE,
        sort_within_batch=True, # NEW. necessary for packed_padded_sequence
             sort_key=lambda x: len(x.sentence),
        device=DEVICE
)

TEXT.build_vocab(train_data, max_size=VOCABULARY_SIZE)
LABEL.build_vocab(train_data)

print(f'Vocabulary size: {len(TEXT.vocab)}')
print(f'Number of classes: {len(LABEL.vocab)}')
print(TEXT.vocab.freqs.most_common(20))
print(TEXT.vocab.itos[:10]) # itos = integer-to-string

print(LABEL.vocab.stoi)
LABEL.vocab.freqs
class RNN(torch.nn.Module):

    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super().__init__()

        self.embedding = torch.nn.Embedding(input_dim, embedding_dim)
        # self.rnn = torch.nn.RNN(embedding_dim,
        #                        hidden_dim,
        #                        nonlinearity='relu')
        self.rnn = torch.nn.LSTM(embedding_dim,
                                 hidden_dim)

        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, text, text_length):
        # text dim: [sentence length, batch size]

        embedded = self.embedding(text)
        # ebedded dim: [sentence length, batch size, embedding dim]

        ## NEW
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, text_length.to('cpu'))

        packed_output, (hidden, cell) = self.rnn(packed)

        # output dim: [sentence length, batch size, hidden dim]
        # hidden dim: [1, batch size, hidden dim]

        hidden.squeeze_(0)
        # hidden dim: [batch size, hidden dim]

        output = self.fc(hidden)
        return output

torch.manual_seed(RANDOM_SEED)
model = RNN(input_dim=len(TEXT.vocab),
            embedding_dim=EMBEDDING_DIM,
            hidden_dim=HIDDEN_DIM,
            output_dim=NUM_CLASSES # could use 1 for binary classification
)

model = model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)



def compute_accuracy(model, data_loader, device):
    with torch.no_grad():
        correct_pred, num_examples = 0, 0

        for batch_idx, batch_data in enumerate(data_loader):
            # NEW
            features, text_length = batch_data.sentence[0].to('cpu'), batch_data.sentence[1]

            targets = batch_data.relation_type.to(DEVICE)

            logits = model(features, text_length)
            _, predicted_labels = torch.max(logits, 1)

            num_examples += len(targets)

            correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float() / num_examples * 100


start_time = time.time()

epochs = list(range(1, NUM_EPOCHS + 1))
train_losses = []
valid_losses = []
train_accuracies = []
valid_accuracies = []

for epoch in epochs:
    model.train()
    for batch_idx, batch_data in enumerate(train_loader):

        # NEW
        features, text_length, *_ = batch_data.sentence[:2]

        labels = batch_data.relation_type.to(DEVICE)

        ### FORWARD AND BACK PROP
        logits = model(features, text_length)
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad()

        loss.backward()

        ### UPDATE MODEL PARAMETERS
        optimizer.step()

        ### LOGGING
        if not batch_idx % 50:
            print(f'Epoch: {epoch + 1:03d}/{NUM_EPOCHS:03d} | '
                  f'Batch {batch_idx:03d}/{len(train_loader):03d} | '
                  f'Loss: {loss:.4f}')

    with torch.set_grad_enabled(False):
        print(f'training accuracy: '
              f'{compute_accuracy(model, train_loader, DEVICE):.2f}%'
              f'\nvalid accuracy: '
              f'{compute_accuracy(model, valid_loader, DEVICE):.2f}%')

    print(f'Time elapsed: {(time.time() - start_time) / 60:.2f} min')
    train_loss = loss.item()
    train_acc = compute_accuracy(model, train_loader, DEVICE).item()

    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    valid_loss = F.cross_entropy(model(features, text_length), labels).item()
    valid_acc = compute_accuracy(model, valid_loader, DEVICE).item()

    valid_losses.append(valid_loss)
    valid_accuracies.append(valid_acc)

plt.plot(epochs, train_losses, label='Training Loss')
plt.plot(epochs, valid_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(epochs, train_accuracies, label='Training Accuracy')
plt.plot(epochs, valid_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

print(f'Total Training Time: {(time.time() - start_time) / 60:.2f} min')
print(f'Test accuracy: {compute_accuracy(model, test_loader, DEVICE):.2f}%')

def compute_metrics(model, data_loader, device, num_classes):
    with torch.no_grad():
        true_positives = [0] * num_classes
        true_negatives = [0] * num_classes
        false_positives = [0] * num_classes
        false_negatives = [0] * num_classes

        for batch_idx, batch_data in enumerate(data_loader):
            features, text_length = batch_data.sentence
            targets = batch_data.relation_type.to(device)

            logits = model(features, text_length)
            _, predicted_labels = torch.max(logits, 1)

            for class_idx in range(num_classes):
                class_mask = targets == class_idx
                predicted_mask = predicted_labels == class_idx

                true_positives[class_idx] += torch.sum(predicted_mask & class_mask).item()
                true_negatives[class_idx] += torch.sum(~predicted_mask & ~class_mask).item()
                false_positives[class_idx] += torch.sum(predicted_mask & ~class_mask).item()
                false_negatives[class_idx] += torch.sum(~predicted_mask & class_mask).item()

        metrics = []
        for class_idx in range(num_classes):
            tp = true_positives[class_idx]
            tn = true_negatives[class_idx]
            fp = false_positives[class_idx]
            fn = false_negatives[class_idx]

            specificity = tn / (tn + fp)
            sensitivity = tp / (tp + fn)
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1_score = 2 * (precision * recall) / (precision + recall)

            metrics.append((specificity, sensitivity, accuracy, precision, recall, f1_score))

        return metrics
# Define the number of classes
NUM_CLASSES = 4

# Print evaluation metrics for each class during training
model.train()
train_metrics = compute_metrics(model, train_loader, DEVICE, NUM_CLASSES)
valid_metrics = compute_metrics(model, valid_loader, DEVICE, NUM_CLASSES)

class_names = ["effect", "advise", "mechanism", "non-interaction"]

for class_idx, (train_specificity, train_sensitivity, train_accuracy, train_precision, train_recall, train_f1_score) in enumerate(train_metrics):
    valid_specificity, valid_sensitivity, valid_accuracy, valid_precision, valid_recall, valid_f1_score = valid_metrics[class_idx]
    class_name = class_names[class_idx]
    print(f'Class: {class_name}')
    print(f'Training Specificity: {train_specificity:.2f}')
    print(f'Training Sensitivity: {train_sensitivity:.2f}')
    print(f'Training Accuracy: {train_accuracy:.2f}')
    print(f'Training Precision: {train_precision:.2f}')
    print(f'Training Recall: {train_recall:.2f}')
    print(f'Training F1 Score: {train_f1_score:.2f}')
    print(f'Validation Specificity: {valid_specificity:.2f}')
    print(f'Validation Sensitivity: {valid_sensitivity:.2f}')
    print(f'Validation Accuracy: {valid_accuracy:.2f}')
    print(f'Validation Precision: {valid_precision:.2f}')
    print(f'Validation Recall: {valid_recall:.2f}')
    print(f'Validation F1 Score: {valid_f1_score:.2f}')
    print()

# Now, during the test phase:
model.eval()
test_metrics = compute_metrics(model, test_loader, DEVICE, NUM_CLASSES)

for class_idx, (test_specificity, test_sensitivity, test_accuracy, test_precision, test_recall, test_f1_score) in enumerate(test_metrics):
    class_name = class_names[class_idx]
    print(f'Class: {class_name}')
    print(f'Test Specificity: {test_specificity:.2f}')
    print(f'Test Sensitivity: {test_sensitivity:.2f}')
    print(f'Test Accuracy: {test_accuracy:.2f}')
    print(f'Test Precision: {test_precision:.2f}')
    print(f'Test Recall: {test_recall:.2f}')
    print(f'Test F1 Score: {test_f1_score:.2f}')
    print()