from ConvAutoencoder import ConvAutoencoder, ConvAutoencoderNoPool

import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix

def CalculateConfusionMatrix(gt_list, pred_list, target_names, img_out_path):
    print("Calculating confusion matrix ...")
    cm = confusion_matrix(y_true=gt_list, y_pred=pred_list)
    # normalized confusion matrix
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm)

    # Plot the confusion matrix as an image.
    plt.matshow(cm, cmap=plt.cm.Blues)
    plt.colorbar()
    num_classes = len(target_names)
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, target_names)
    plt.yticks(tick_marks, target_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    #plt.title('Confusion Matrix for CIFAR10. Accuracy: %f' % (accuracy))

    # Loop over data dimensions and create text annotations.
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    figure = plt.gcf()
    figure.set_size_inches(15, 15)

    plt.savefig(img_out_path, bbox_inches='tight', dpi=200)
    # plt.show()
    plt.close()

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load('models/AutoencoderClassifier_with_Normalize.pt')
    #state = torch.load('models/AutoencoderClassifier_last_without_norms.pt')
    model = state['architecture']
    model.load_state_dict(state['state_dict'])

    mean = 0.5
    sd = 0.5
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[mean]*3, std=[sd]*3)])
    test_dataset = datasets.ImageFolder(root='results\Decoded\Imitated_FromNoGen', transform=transform)
    test_data = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)
    classes = test_data.dataset.classes
    model.eval()  # переводим модель в режим предсказания
    model.classifier.eval()
    model.autoencoder.eval()
    predictions = []
    true_labels = []
    with torch.no_grad():
        for images, labels in test_data:
            images = images.to(device)
            labels = labels.to(device)
            decoded, outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            predictions += predicted.cpu().numpy().tolist()
            true_labels += labels.cpu().numpy().tolist()


    CalculateConfusionMatrix(true_labels, predictions, classes, 'confusion_matrix_FromNoGen.png')
    # # строим матрицу распознавания
    # conf_matrix = confusion_matrix(true_labels, predictions)
    # # normalized confusion matrix
    # conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    # print(conf_matrix)
    #
    # fig, ax = plt.subplots(figsize=(10, 10))
    # im = ax.imshow(conf_matrix, cmap='viridis')
    #
    # # добавляем названия классов на оси x и y
    # ax.set_xticks(np.arange(len(test_data.dataset.classes)))
    # ax.set_yticks(np.arange(len(test_data.dataset.classes)))
    # ax.set_xticklabels(test_data.dataset.classes)
    # ax.set_yticklabels(test_data.dataset.classes)
    #
    # # включаем легенду и сохраняем изображение
    # plt.colorbar(im)
    # plt.savefig('confusion_matrix.png')
    # plt.show()