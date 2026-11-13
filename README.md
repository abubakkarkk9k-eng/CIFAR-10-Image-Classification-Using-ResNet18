CIFAR-10 Image Classification Using Transfer Learning with ResNet18 and MLflow
Project Overview
This project implements image classification on the CIFAR-10 dataset using Transfer Learning with a pretrained ResNet18 model. The model starts with weights pretrained on ImageNet. Its final fully connected layer is replaced with a 10-class classification layer, and the pretrained feature-extraction layers are frozen while the new classification layer is trained on CIFAR-10.

The project is implemented in Python using PyTorch and Torchvision in Google Colab. MLflow is configured for experiment tracking, and the trained model checkpoint is saved to Google Drive.
Objectives
- Apply transfer learning to an image-classification task.
- Adapt pretrained ResNet18 to classify the 10 CIFAR-10 categories.
- Train the new classification layer using the CIFAR-10 training set.
- Evaluate the model on the separate CIFAR-10 test set.
- Review per-class performance and a confusion matrix.
- Track training experiments with MLflow.
- Save the trained model checkpoint for later use.
Dataset
The project uses the CIFAR-10 dataset, which contains 60,000 colour images across 10 classes.
Dataset split	Number of images
Training	50,000
Testing	10,000
Total	60,000


The original images are 32 × 32 pixels with three RGB channels. For ResNet18, the notebook resizes images to 224 × 224 pixels and applies ImageNet normalization.
CIFAR-10 Classes
1. Airplane
2. Automobile
3. Bird
4. Cat
5. Deer
6. Dog
7. Frog
8. Horse
9. Ship
10. Truck
Model and Transfer-Learning Approach
The project uses ResNet18 with pretrained ImageNet weights.

The notebook:
1. Loads ResNet18 with the default pretrained ImageNet weights.
2. Replaces the original 1,000-class fully connected layer with a new layer containing 10 outputs.
3. Freezes the pretrained model parameters.
4. Keeps the new final classification layer trainable.
5. Trains the classification layer on CIFAR-10.
This is a feature-extraction approach to transfer learning. The model has 11,181,642 parameters in total, of which 5,130 are trainable and 11,176,512 are frozen.
Image Preprocessing
Each CIFAR-10 image is processed using the following transformations:
- Resize to 224 × 224 pixels.
- Convert the image to a PyTorch tensor.
- Normalize the RGB channels using the ImageNet mean (0.485, 0.456, 0.406) and standard deviation (0.229, 0.224, 0.225).
Training Configuration
Parameter	Value
Model	ResNet18
Pretrained weights	ImageNet
Training method	Feature-extraction transfer learning
Trainable layer	Final fully connected layer
Loss function	CrossEntropyLoss
Optimizer	Adam
Learning rate	0.001
Batch size	64
Number of epochs	5
Training images	50,000
Test images	10,000


Training Results
The notebook recorded the following training results:
Epoch	Training loss	Training accuracy
1	0.5625	82.42%
2	0.3803	87.02%
3	0.3544	87.70%
4	0.3413	88.17%
5	0.3342	88.55%


Test Evaluation Results
The final evaluation on the 10,000 CIFAR-10 test images produced the following results:
Metric	Result
Test accuracy	86.94%
Test loss	0.3794
Correct predictions	8,694
Total test images	10,000


Class-Wise Test Accuracy
CIFAR-10 class	Accuracy
Airplane	92.00%
Automobile	93.50%
Bird	77.80%
Cat	79.00%
Deer	88.70%
Dog	81.60%
Frog	91.80%
Horse	86.20%
Ship	90.20%
Truck	88.60%


The notebook also generates a 10 × 10 confusion-matrix heatmap to compare actual labels with predicted labels.
MLflow Experiment Tracking
MLflow is used to organize and track the training experiment. The notebook creates an experiment named CIFAR10_Image_Classification and stores its tracking data in the mlruns directory in Google Drive.

The training run logs the model name, pretrained-weight source, number of epochs, batch size, learning rate, training loss, and training accuracy for each epoch.
Technologies and Environment
- Python
- PyTorch 2.14.1
- Torchvision
- ResNet18
- ImageNet pretrained weights
- MLflow 3.18.0
- Matplotlib
- Google Colab
- NVIDIA Tesla T4 GPU
- Google Drive
How to Run
1. Open CIFAR-10_Image_Classification_Using_ResNet18.ipynb in Google Colab.
2. In Runtime → Change runtime type, select a GPU accelerator.
3. Run the notebook cells in order.
4. Mount Google Drive when prompted.
5. Allow Torchvision to download CIFAR-10 and the pretrained ResNet18 weights when required.
6. Run the preprocessing, model setup, training, evaluation, and checkpoint-saving cells.
For a local Python environment, install the main dependencies with:
pip install torch torchvision mlflow matplotlib
A compatible PyTorch installation and CUDA setup are required to use an NVIDIA GPU locally. Google Colab's preconfigured environment can be used instead.
Model Checkpoint
The notebook saves the model checkpoint to Google Drive using the filename:

best_resnet18_cifar10.pth

The checkpoint contains the ResNet18 model state dictionary, class names, epoch information, test accuracy, and test loss.

Checkpoint-selection note: In the uploaded notebook version, the save cell stores model.state_dict() after the five-epoch training run. The notebook does not show a validation split or validation-based best-checkpoint selection. Therefore, this file represents the model state at the end of the training run; the best_ prefix in its filename does not, by itself, mean that the checkpoint was selected using validation accuracy.
Project Structure
CIFAR10_Image_Classification_Using_ResNet18/
│
├── CIFAR-10_Image_Classification_Using_ResNet18.ipynb
├── README.md
├── best_resnet18_cifar10.pth
└── mlruns/
The notebook stores the checkpoint and MLflow tracking directory in Google Drive. Include the notebook and README in the GitHub repository. If the model checkpoint is too large for the repository or your GitHub workflow, keep it in Google Drive or another model-storage location and document its location rather than committing it directly.
Conclusion
This project demonstrates feature-extraction transfer learning for CIFAR-10 image classification using pretrained ResNet18. The final test evaluation achieved 86.94% accuracy on 10,000 test images. The notebook also provides per-class accuracy, a confusion-matrix visualization, MLflow training tracking, and a saved model checkpoint.
