from torchaudio.datasets import SPEECHCOMMANDS
from torchaudio import transforms
import torch
import os
    
### pre-computed stats and values
# the labels in the data
labels = ['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine',
 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero']
# the min and max input values
min_bound = -1.3844940662384033
max_bound = 1.3773366212844849

class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, subset: str = None):
        super().__init__('/home/data/torchaudio-SPEECHCOMMANDS/', download=False)

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [os.path.join(self._path, line.strip()) for line in fileobj]

        if subset == 'validation':
            self._walker = load_list('validation_list.txt')
        elif subset == 'testing':
            self._walker = load_list('testing_list.txt')
        elif subset == 'training':
            excludes = load_list('validation_list.txt') + load_list('testing_list.txt')
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]
                    
    # DONT recompute each time 
#     labels = sorted(list(set(datapoint[2] for datapoint in train_set)))

def label_to_index(word):
    # Return the position of the word in labels
    return torch.tensor(labels.index(word))

def index_to_label(index):
    # Return the word corresponding to the index in labels
    # This is the inverse of label_to_index
    return labels[index]

def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch.permute(0, 2, 1)

def collate_fn(batch, resample_rate):

    # A data tuple has the form:
    # waveform, sample_rate, label, speaker_id, utterance_number

    tensors, targets = [], []
    
    # resampling func
    transform = transforms.Resample(orig_freq=16000, new_freq=resample_rate)

    # Gather in lists, and encode labels as indices
    for waveform, _, label, *_ in batch:
        tensors += [transform(waveform)]
        targets += [label_to_index(label)]

    # Group the list of tensors into a batched tensor
    tensors = pad_sequence(tensors)
    targets = torch.stack(targets)

    return tensors, targets


def SpeechCommands_Dataloaders(resample_rate=8000, batch_size=32, batchsize_for_val=10000, **kwargs):
    
    # Create training and testing split of the data.
    train_set = SubsetSC('training')
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=(lambda batch: collate_fn(batch, resample_rate)),
        num_workers=kwargs['num_workers'],
        pin_memory=kwargs['pin_memory'],
    )
    
    val_set = SubsetSC('validation')
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=batchsize_for_val,
        shuffle=False,
        collate_fn=(lambda batch: collate_fn(batch, resample_rate)),
        num_workers=kwargs['num_workers'],
        pin_memory=kwargs['pin_memory'],
    )
    
    test_set = SubsetSC('testing')
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batchsize_for_val,
        shuffle=False,
        drop_last=False,
        collate_fn=(lambda batch: collate_fn(batch, resample_rate)),
        num_workers=kwargs['num_workers'],
        pin_memory=kwargs['pin_memory'],
    )
    
    return(train_loader, val_loader, test_loader)
