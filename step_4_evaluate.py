from torch.autograd import Variable

import torch
import numpy as np

import onnx
import onnxruntime as ort

from step_3_train import Net
from step_2_dataset import get_train_test_loaders


def evaluate(outputs: Variable, labels: Variable) -> float:
    """Evaluate neural network outputs against non-one-hotted labels
    
    Args:
        outputs (Variable): A list of class probabilities for each sample.
        labels (Variable): Non-one-hotted labels
    """

    Y = labels.numpy()
    Yhat = np.argmax(outputs, axis=1) # Converts the `outputs` class probabilities to predicted classes.
    return float(np.sum(Y == Yhat)) # Ensures that the predicted classse match the label class.

def batch_evaluate(model: Net, loader: torch.utils.data.DataLoader) -> float:
    """Evaluate a neural network on a batch of data
    
    Args:
        model (Net): The model to evaluate.
        loader (torch.utils.data.DataLoader): A batch of images stored as a single tensor.
    """
    
    score = n = 0.0
    
    for batch in loader:
        n += len(batch['image'])
        outputs = model(batch['image'])

        if isinstance(outputs, torch.Tensor):
            outputs = outputs.detach().numpy()

        score += evaluate(outputs, batch['label'][:, 0])

    return score / n # The percent of samples correctly classified

def validate():
    """Loads a pretrained neural network and evaluates its 
    performance on the provided data set.
    """
    trainloader, testloader = get_train_test_loaders()

    net = Net().float().eval()
    
    pretrained_model = torch.load("checkpoint.pth")
    net.load_state_dict(pretrained_model)
    
    print('=' * 10, 'PyTorch', '=' * 10)
    train_acc = batch_evaluate(net, trainloader) * 100.
    print('Training accuracy: %.1f' % train_acc)
    test_acc = batch_evaluate(net, testloader) * 100.
    print('Validation accuracy: %.1f' % test_acc)

    # Exporting the model to an ONNX binary.
    
    trainloader, testloader = get_train_test_loaders(1)
    
    fname = "sign_language.onnx"
    dummy = torch.randn(1, 1, 28, 28)
    torch.onnx.export(net, dummy, fname, input_names=['input'])

    # Checking the exported model
    
    model = onnx.load(fname)
    onnx.checker.check_model(model) # Checks that the model is well-formed.
    
    # Create runnable session
    
    ort_session = ort.InferenceSession(fname)

    def net(inp):
        return ort_session.run(None, {'input': inp.data.numpy()})[0]
    
    print('=' * 10, 'ONNX', '=' * 10)
    train_acc = batch_evaluate(net, trainloader) * 100.
    print('Training accuracy: %.1f' % train_acc)
    test_acc = batch_evaluate(net, testloader) * 100.
    print('Validation accuracy: %.1f' % test_acc)
    
if __name__ == "__main__":
    validate()