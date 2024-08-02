##########################################################################################
# Description : VGG Model Training, Test and validation
# Autores     : Daniela Rigoli e Luigi Carvalho
# Creation    : 23/06/2024  Luigi Carvalho
#
##########################################################################################

##########################################################################################
# 1- Instantiate a base model and load pre-trained weights into it.
# 2- Freeze all layers in the base model by setting trainable = False.
# 3- Create a new model on top of the output of one (or several) layers from the base model.
# 4- Train your new model on your new dataset.

# The default input size for this model(VGG19) is 224x224.
# For VGG19, call vgg19.preprocess_input on your inputs before passing them to the model. 
# vgg19.preprocess_input will convert the input images from RGB to BGR, 
# then will zero-center each color channel with respect to the ImageNet dataset, without scaling.

##########################################################################################

##########################################################################################
# imports
import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import torch
import torch.nn               as nn
from   torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets   as datasets 
from   sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
# roc curve and auc
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, precision_score, recall_score
    
""" 
from imblearn.over_sampling import RandomOverSampler
from sklearn.utils import resample
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE 
"""

'''
<bound method Module.parameters of VGG(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU(inplace=True)
    (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU(inplace=True)
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (6): ReLU(inplace=True)
    (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (8): ReLU(inplace=True)
    (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace=True)
    (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (13): ReLU(inplace=True)
    (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (15): ReLU(inplace=True)
    (16): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (17): ReLU(inplace=True)
    (18): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (19): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (20): ReLU(inplace=True)
    (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (22): ReLU(inplace=True)
    (23): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (24): ReLU(inplace=True)
    (25): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (26): ReLU(inplace=True)
    (27): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (29): ReLU(inplace=True)
    (30): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (31): ReLU(inplace=True)
    (32): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (33): ReLU(inplace=True)
    (34): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (35): ReLU(inplace=True)
    (36): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
  (classifier): Sequential(
    (0): Linear(in_features=25088, out_features=4096, bias=True)
    (1): ReLU(inplace=True)
    (2): Dropout(p=0.5, inplace=False)
    (3): Linear(in_features=4096, out_features=4096, bias=True)
    (4): ReLU(inplace=True)
    (5): Dropout(p=0.5, inplace=False)
    (6): Linear(in_features=4096, out_features=ClassesNumber, bias=True)
  )
)>

'''



##########################################################################################
# Description : Instantiate Model
# Creation    : 23/06/2024  Luigi Carvalho
# Modificada  : 25/06/2024  Daniela Rigoli
#
##########################################################################################

class CustomVGG19(nn.Module):
    def __init__(self):
        print('init')
        self.nPasses = 0
        self.strLog  = ''
        super(CustomVGG19, self).__init__()
        
        vgg19                         = torchvision.models.vgg19(weights='VGG19_Weights.IMAGENET1K_V1')
        for param in vgg19.parameters():
            param.requires_grad = False

        self.features                 = vgg19.features
        self.avgpool                  = vgg19.avgpool
        
        # Customize classifier layers for binary classification
        self.classifier = nn.Sequential(
            nn.Linear(25088, 4096),        # Original FC layer 1
            nn.ReLU(True),
            #nn.Dropout(p=0.5, inplace=False),
            nn.Linear(4096, 4096),         # Original FC layer 2
            nn.ReLU(True),
            #nn.Dropout(p=0.5, inplace=False),
            nn.Linear(4096, 1)             # Output layer for binary classification
        )

    def forward(self, x):
        print('forward : ', self.nPasses)
        self.nPasses += 1
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x



model = CustomVGG19()




print(model.parameters)

##########################################################################################
# Description : Load Dataset
# Creation    : 23/06/2024  Luigi Carvalho
# Modificada  : 25/06/2024  Daniela Rigoli
#
##########################################################################################
print('Loading Dataset')
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomResizedCrop(size=(224, 224)),
    transforms.RandomPerspective(),
    transforms.RandomRotation(90),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Use the mean and std of ImageNet
])

dataset    = datasets.ImageFolder(root='./../Cenario/cenario2/train', transform=transform)
validation = datasets.ImageFolder(root='./../Cenario/cenario2/validation', transform=transform)
test       = datasets.ImageFolder(root='./../Cenario/cenario2/test', transform=transform)

print(dataset.root)
samples_count = len(dataset.targets)
# Print classes and their numeric labels
print("Classes and their numeric labels:")
model.strLog += '\n Classes and their numeric labels: ' 
class_to_idx  = dataset.class_to_idx
print(class_to_idx)
model.strLog += str(class_to_idx)
print('aberto train samples:', dataset.targets.count(0), 'fechado train samples: ', dataset.targets.count(1))
model.strLog += f'\naberto train samples: {dataset.targets.count(0)} fechado train samples: {dataset.targets.count(1)}\n'

print('aberto validation samples:', validation.targets.count(0), 'fechado validation samples: ', validation.targets.count(1))
model.strLog += f'\naberto validation samples: {validation.targets.count(0)} fechado validation samples: {validation.targets.count(1)}\n'

print('aberto test samples:', test.targets.count(0), 'fechado test samples: ', test.targets.count(1))
model.strLog += f'\naberto test samples: {test.targets.count(0)} fechado test samples: {test.targets.count(1)}\n'

# Define the sizes of each dataset
train_size = len(dataset)
val_size   = len(validation)
test_size  = len(test)

# Configure the dataset
train_dataset = dataset 
val_dataset   = validation
test_dataset  = test

# Apply different transforms to validation and test sets
train_dataset.transform  = transform
val_dataset.transform  = transform
test_dataset.transform = transform


batch_size = 64
num_epochs = 5 
learning_rate = 0.001
checkpointPath = 'model_classifier.pth'
optimizerPath  = 'optmizerADAM.pth'

if os.path.exists(checkpointPath):
  # initialize check point
  model.classifier.load_state_dict(torch.load(checkpointPath))
  model.eval()

print("num_epochs ", num_epochs)
model.strLog += f'\nnum_epochs: {num_epochs}'


# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

print('\nLoading Complete')

##########################################################################################
# Description : Model Setup
# Creation    : 23/06/2024  Luigi Carvalho
# Modificada  : 25/06/2024  Daniela Rigoli
#
##########################################################################################

model = model.to('cpu')

criterion = nn.BCEWithLogitsLoss() # Binary Cross Entropy
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

if os.path.exists(optimizerPath):
  # initialize check point
  optimizer.load_state_dict(torch.load(optimizerPath))



print('Model Setup Complete')
model.strLog += '\nModel Setup Complete'


##########################################################################################
# Description : Eval Setup
# Creation    : 23/06/2024  Luigi Carvalho
# Modificada  : 25/06/2024  Daniela Rigoli
#
##########################################################################################
def evaluate_model(model, dataloader):
    print('It\'s eval time')
    model.strLog += '\nIt\'s eval time'
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0
    correct = 0

    with torch.no_grad():  # Disable gradient computation
        for inputs, labels in dataloader:
            labels = labels.float()  # Ensure labels are float
            outputs = model(inputs).squeeze()  # Forward pass and ensure outputs are of shape [batch_size]

            loss = criterion(outputs.squeeze(), labels)  # Compute the loss
            running_loss += loss.item() * inputs.size(0)  # Accumulate the loss

            # Apply sigmoid to the outputs and threshold at 0.5 to get binary predictions
            preds = torch.sigmoid(outputs) >= 0.5
            correct += torch.sum(preds == labels).item()  # Accumulate the number of correct predictions

    epoch_loss = running_loss / len(dataloader.dataset)  # Calculate the average loss
    accuracy = correct / len(dataloader.dataset)  # Calculate the accuracy

    return epoch_loss, accuracy  # Return the loss and accuracy

##########################################################################################
# Description : AUC ROC
# Creation    : 25/06/2024  Daniela Rigoli
# Modificada  : 
#
##########################################################################################
id = 1
from sklearn.metrics import roc_curve, auc
def aucRoc(model, dataloader):
    model.eval()  # Set model to evaluation mode
    y_true = []
    y_pred = []
    lr_probs = []
    print('aucRoc time')
    model.strLog += '\naucRoc time'

    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            preds = torch.sigmoid(outputs) >= 0.5  # Convert logits to binary predictions
            y_true.extend(labels.numpy())
            y_pred.extend(preds.numpy())
            lr_probs.extend(outputs.numpy())
            #print(lr_probs)

    # Generate a no skill prediction (majority class)
    ns_probs = [0 for _ in range(len(y_true))]
    # Calculate scores
    ns_auc = roc_auc_score(y_true, ns_probs)
    lr_auc = roc_auc_score(y_true, lr_probs)
    # Summarize scores
    print('No Skill: ROC AUC=%.3f' % (ns_auc))
    model.strLog += str('\nNo Skill: ROC AUC=%.3f' % (ns_auc))
    print('Logistic: ROC AUC=%.3f' % (lr_auc))
    model.strLog += str('\nLogistic: ROC AUC=%.3f' % (lr_auc))
    # Calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(y_true, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(y_true, lr_probs)
    #plot the roc curve for the model
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
    # Axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # Show the legend
    plt.legend()
    # Save the plot
    global id
    plt.savefig('aucRoc'+ str(id) +'.png')
    id+=1
    plt.close()

##########################################################################################
# Description : Geral metrics
# Creation    : 25/06/2024  Daniela Rigoli
# Modificada  : 
#
##########################################################################################
def metrics(model, dataloader):
    model.eval()  # Set model to evaluation mode
    y_true = []
    y_pred = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            preds = torch.sigmoid(outputs) >= 0.5  # Convert logits to binary predictions
            y_true.extend(labels.numpy())
            y_pred.extend(preds.numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate some metrics
    accuracy = accuracy_score(y_true, y_pred)
    print("Accuracy:", accuracy)
    model.strLog += f'\nAccuracy: {accuracy} \n'

    class_report = classification_report(y_true, y_pred)
    print("Classification Report:\n", class_report)
    model.strLog += "Classification Report:\n" + class_report

    precision = precision_score(y_true, y_pred)
    print("Precision:", precision)
    model.strLog += "Precision: " + str(precision)

    recall = recall_score(y_true, y_pred)
    print("Recall:", recall)
    model.strLog += "\nRecall: " + str(recall)

    f1 = f1_score(y_true, y_pred)
    print("F1-Score:", f1)
    model.strLog += "\nF1-Score: " + str(f1)

##########################################################################################
# Description : Confusion matrix
# Creation    : 23/06/2024  Luigi Carvalho
# Modificada  : 25/06/2024  Daniela Rigoli
#
##########################################################################################
def plot_confusion_matrix(model, dataloader):
    model.eval()  # Set model to evaluation mode
    y_true = []
    y_pred = []
    print('Confusion time')

    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            preds = torch.sigmoid(outputs) >= 0.5  # Convert logits to binary predictions
            y_true.extend(labels.numpy())
            y_pred.extend(preds.numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    conf_matrix = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", conf_matrix)
    model.strLog += "\nConfusion Matrix:\n" + str(conf_matrix)

    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['Aberto', 'Fechado'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    global id
    plt.savefig('mtc'+ str(id) +'.png')
    plt.close()


##########################################################################################
# Description : Training
# Creation    : 23/06/2024  Luigi Carvalho
# Modificada  : 25/06/2024  Daniela Rigoli
#
##########################################################################################
import torchvision.models as models

for epoch in range(num_epochs):
    print(f'Training Epoch {epoch+1}/{num_epochs}')
    model.strLog += f'\nTraining Epoch {epoch+1}/{num_epochs}\n'
    model.train()
    running_loss = 0.0
    
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        labels = labels.unsqueeze(1).float()  # Ensure labels are of shape [batch_size, 1]
        #print("labels:", labels, "predict output:", (torch.sigmoid(outputs) >= 0.5).int())

        loss = criterion(outputs, labels)
        loss.backward()
        
        # Print and inspect gradients
        # for name, param in model.classifier.named_parameters():
        #     if param.requires_grad and param.grad is not None:
        #         print(f'Layer: {name}, Gradient Norm: {param.grad.norm().item()}')
        
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        print(f'Train Step {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}')
        print('Loss:', loss.item())
        print('Running Loss:', running_loss)
    
    # Save the model classifier state
    torch.save(model.classifier.state_dict(), checkpointPath)
    # Save the optmizer state
    torch.save(optimizer.state_dict(), optimizerPath)

    train_loss = running_loss / len(train_loader.dataset)

    # Validation phase after each epoch
    val_loss, val_accuracy = evaluate_model(model, val_loader)
    
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, '
          f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')
    
    model.strLog += f'\nEpoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}\n'

    # Optionally, you can plot the confusion matrix for validation set after each epoch
    plot_confusion_matrix(model, val_loader)
    metrics(model, test_loader)
    aucRoc(model, test_loader)

    f = open('output.log', 'w')
    f.write(model.strLog)
    f.close()


# Testing phase
test_loss, test_accuracy = evaluate_model(model, test_loader)
print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
model.strLog += f'\nTest Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}\n'

# Plot confusion matrix for test set
plot_confusion_matrix(model, test_loader)
metrics(model, test_loader)
aucRoc(model, test_loader)

f = open('output.log', 'w')
f.write(model.strLog)
f.close()
