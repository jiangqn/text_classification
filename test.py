from fasttext import *
from lstm_classifier import *
from gru_classifier import *
from text_cnn import *
import pickle

with open('./dataset/params.pickle', 'rb') as handle:
    params = pickle.load(handle)

vocab_size = params['vocab_size']
class_num = params['class_num']
train_texts = params['train_texts']
train_labels = params['train_labels']
val_texts = params['val_texts']
val_labels = params['val_labels']

model = TextCNN(vocab_size=vocab_size, class_num=class_num)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
batch_size = 100

for epoch in range(30):
    num_train = train_texts.shape[0]
    num_batches = num_train // batch_size
    for i in range(num_batches):
        batch_texts = train_texts[i * batch_size: (i + 1) * batch_size]
        batch_labels = train_labels[i * batch_size: (i + 1) * batch_size]
        batch_texts = torch.tensor(batch_texts).long()
        batch_labels = torch.tensor(batch_labels).long()
        optimizer.zero_grad()
        outputs = model(batch_texts)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            print('epoch[%d]\tstep[%d] \tloss: %.4f' % (epoch, i, loss.item()))
    total_cases = 0
    correct_cases = 0
    num_val = val_texts.shape[0]
    num_val_batches = num_val // batch_size
    for i in range(num_val_batches):
        batch_texts = val_texts[i * batch_size: (i + 1) * batch_size]
        batch_labels = val_labels[i * batch_size: (i + 1) * batch_size]
        batch_texts = torch.tensor(batch_texts).long()
        batch_labels = torch.tensor(batch_labels).long()
        outputs = model(batch_texts)
        _, predicts = outputs.max(dim=1)
        total_cases += batch_labels.shape[0]
        correct_cases += (predicts == batch_labels).sum().item()
    print('epoch[%d]\taccuracy: %.4f' % (epoch, correct_cases / total_cases))