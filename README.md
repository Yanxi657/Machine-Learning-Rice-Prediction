# Machine-Learning-Rice-Prediction
This project used image recognition technology to select the best rice variety for catering industry and daily cooking. Five different varieties were selected and trained using CNN and SimpleNN models, resulting in ten differentiated models with an accuracy of 97.95%. The model was successfully used to predict new samples.

# 1.Abstract
As a widely grown crop, rice has many varieties around the world, and its application scenarios are different. For the catering industry and daily cooking at home, it is of great significance to choose rice that matches the scene.
However, artificial selection of rice varieties is not only time-consuming and labor-intensive, but also prone to misjudgments or missed judgments. Image recognition technology provides an efficient and accurate rice variety selection scheme.
In this project, the research selected five different rice varieties, 80% of which were used as the training set, and CNN and SimpleNN models were used for training and testing. At the same time, a variety of methods were used to adjust the parameters of the model, and finally a total of ten differentiated models were obtained. After data visualization of the model's predictive accuracy, the most accurate model achieved an accuracy of 97.95%. After that, use the model to predict new samples and get the results.
In the end, we successfully realized the image recognition and classification of these five types of rice.

# 2.Introduction
## 2.1 Dataset
We selected five types of rice from Italy, Turkey, South Asia and Thailand, each with different characteristics and application scenarios.
We have a total of 12,000 pieces of rice data, which are divided into training set and test set according to the ratio of 80% and 20%. Among them, 10,000 pieces are used as the training set, and 2,000 pieces of each type of rice are used. The remaining 2000 pieces are used as the training set, with 400 pieces for each type of rice.
The characteristics of these five types of rice are also different, such as: Arborio rice from Italy has slightly spherical grains, shorter particle size and whiter color; Ipsala rice from Turkey has full, elastic grains, slightly yellow or green color, and long particle size; Basmati rice from South Asia is slender and golden yellow; Jasmine rice from Thailand is white in color, fat and crystal clear; Karacadag rice from Turkey has smaller grains, darker color and harder texture.
## 2.2 Solutions and practical significance  
By using image recognition technology, we effectively solve the problem of inaccurate matching between the user's selection of rice and the application scenario according to the provided data set and the appearance characteristics of different varieties of rice, and at the same time improve the efficiency and accuracy of rice quality assessment and grading, and reduce the cost and error of manual selection. Therefore, this project has practical application value in the rice market, rice factory, catering industry and other fields. In short, rice sorting and sorting systems based on image recognition technology have a wide range of application prospects and market demand.

# 3.Methodology
## 3.1 Simple NN & pros and cons
SimpleNN is a simple neural network model used for solving classification and regression problems. It consists of an input layer, a hidden layer, and an output layer, each containing multiple neurons. The input layer receives input data, the hidden layer transforms the input data into higher-level feature representations through activation functions, and the output layer maps the output of the hidden layer to the desired output space. During training, the model parameters are adjusted using the backpropagation algorithm to minimize prediction errors. 
In the practical application of the Simple NN model, we summarize the following advantages and disadvantages. First of all, compared with other complex neural network models, the structure of Simple NN is relatively simple, only including input layer, hidden layer and output layer, which is relatively easy to implement; Secondly, each hidden layer of Simple NN can handle complex nonlinear problems through nonlinear transformation; Third, this model can automatically learn the features required by the classification task, without the need to manually design features, which greatly reduces the manual workload; Fourth, this model can also be used to solve a variety of classification and regression problems, such as image classification, speech recognition, prediction problems, etc.; Finally, Simple NN can perform parallel computing, which improves training and inference efficiency when applied to parallel computing devices such as GPUs. In short, compared to other complex neural network models, Simple NN is simpler to construct and use, while still having the ability to handle complex problems.However, the disadvantages are obvious, such as performance issues when dealing with large and complex data sets; The generalization ability of the model is limited and overfitting is easy to occur. You can get stuck in a local optimal solution and not find a global optimal solution.
In order to improve the performance and generalization ability of Simple NN, we find the following methods can be considered through practical application. For example, we can use more hidden layers and neurons to increase the degree of freedom of the model, and add regularization methods (such as Dropout) to prevent overfitting. Add more layers to the model and use more complex architectures, such as convolutional neural networks and cyclic neural networks, to adapt to more complex data structures and task requirements; Increase the size of training data set to improve the generalization ability of the model.
## 3.2 CNN & pros and cons  
CNN stands for Convolutional Neural Network. It is a type of neural network commonly used in image and video recognition tasks. In contrast to a fully connected neural network, where each neuron in one layer is connected to every neuron in the next layer, CNNs use convolutional layers that apply filters to small regions of the input data. This allows the network to learn local patterns and features in the input data, while reducing the number of parameters needed to be learned. CNNs often also include pooling layers, which downsample the output of the convolutional layers, further reducing the number of parameters and helping to prevent overfitting.
CNN stands for Convolutional Neural Network. It is a neural network commonly used for image and video recognition tasks. In contrast to fully connected neural networks, where every neuron in one layer is connected to every neuron in the next, CNNs use convolutional layers to apply filters to small areas of input data. This allows the network to learn local patterns and features in the input data while reducing the number of parameters that need to be learned. CNNs often also include pooling layers that downsample the output of the convolutional layer, further reducing the number of parameters and helping to prevent overfitting.  In the CNN application of this project, we find that it has the following advantages: first, the local connection and weight sharing methods significantly reduce the model parameters and reduce the computational complexity, thereby accelerating the training and inference speed of the model; Secondly, through the nonlinear combination of multi-layer neurons, the high-order abstract features of the original image can be extracted, so as to realize the recognition and classification of complex categories. Finally, in practical applications, CNNs are also widely used in object detection, object tracking, scene classification and other fields, and have strong practical application value. Of course, it also has certain disadvantages, such as: the training of CNN models requires a large number of samples and computing resources, and cannot handle the problem of small datasets well; In practical applications, CNNs still have certain limitations in some complex scenarios, such as occlusion and detection of targets of multiple sizes. In the future, we can try to combine CNNs with deep learning algorithms such as RNNs to achieve tasks such as video processing, keyframe extraction, and action recognition in complex scenarios.

# 4.Experimental Study
## 4.1Experimental procedure and evaluation method. 
·X_train,y_train
Unzip the dataset
Load training set data: 'X_train' is used to store a list of pixel data of all training images; 'y_train' is used to store the label information of the training data. When reading a CSV file, it is stored in the 'y_train' variable, and each line of it records the category label corresponding to each image.
This code loads images with a for loop and converts them to grayscale images, and finally stores them in the 'X_train' list. We use a for loop to iterate through all the image files and then open the image files using the 'Image.open' function. Suppose all image files are named "img-1.jpg", "img-2.jpg", and so on. Therefore, in each loop, a string link is used to load the corresponding image as part of the image file name with the current number of loops. Converts an open image to grayscale for subsequent processing. The grayscale matrix (tri) is then appended to the 'X_train' list. Finally, convert the 'X_train' to a NumPy array for subsequent training processes.
Training set output effect: the value of each pixel grid of 10,000 images. Each sample is a matrix of 250x250 pixel values; There are 10,000 samples in the training set, and each sample contains a label.
Convert the English character labels of 5 types of rice into integer labels of 0-4 and store them in 'y_train_num' for use during training. For example, the category "Arborio" is converted to 0, "Basmati" is converted to 1, and so on. Finally, output 'y_train_num' to check if the label is successfully converted.
This code converts the y_train_num into a NumPy one-dimensional array containing 10000 labels.
So far,Both x_train and y_train are ready.

·X_test,y_train
This part is the same as the previous step, reading the image of the test set and adding the preprocessed results to the list 'X_test'. Use the 'convert()' function to convert an open image to a single-channel grayscale image. Because for image classification tasks, only grayscale information can be used to classify images efficiently, and grayscale image processing is computationally expensive.

·Processing data
Next, we preprocess the training and test set data, including the two steps of normalization and increasing dimensions. Firstly, the image data of the training set and the test set are normalized, and the grayscale image (pixel value 0-255) is mapped between -0.5 and +0.5. Normalizing the data to the same order of magnitude helps to improve the training speed and stability of the neural network, while avoiding gradient problems caused by Tang mutation. Next, we use the expand_dims() method to add a channel dimension to convert a two-dimensional grayscale map into a three-dimensional grayscale map. Here, the channel number of the grayscale image is increased from 1 to 2 by passing the value 3 in the axis parameter and converted to the specified 4D tensor form (number of samples× height × width× number of channels). Finally, the result is printed to verify the correctness of the preprocessing step.

·Modeling
Next, start defining the convolutional neural network model for classifying images. The specific implementation method is as follows: First, define three variables num_filters, filter_size and pool_size, define the convolution kernel as 8, the convolution kernel size is 3*3, and the maximum pooling is adopted, and the pooling window is 282. Next, create a sequence model using the Sequential() function and add convolutional, pooling, and fully connected layers to the model in turn. Finally, a fully connected layer Dense() composed of 5 neurons is added, and the output vector is normalized using the softmax activation function to obtain the final probability distribution, and the final output result is mapped to five integers from 0 to 4 to complete the classification of the picture.
Next, set the compilation parameters of the neural network model, including the optimizer, loss function, and metrics. First, Adam was chosen because it is an adaptive optimizer that introduces momentum as well as second-order momentum on the basis of gradient descent, which allows for faster and more stable convergence. Second, specify the loss function as categorical_crossentropy because the cross-entropy loss function effectively measures the gap between the classifier's predicted outcome and the true outcome. Finally, set the metric to accuracy to evaluate the performance of the model after each epoch completes and measure the proportion of samples correctly classified by the classifier.
·Training model

## 4.2 Analysis of the experimental results
SimpleNN model training-V1
In deep learning, it is usually necessary to optimize the performance of the model by adjusting the learning rate. If the learning rate is too large, the model may become unstable or diverge during training; if the learning rate is too small, the model may converge too slowly or fall into a local optimal solution. So we then set learning rate to train our model.
V2-After adding learning_rate=0.0005

Compare V1 & V2:
The trend of accuracy has changed from decreasing to gradually rising after reaching the peak, and the trend of loss value has also changed from a growing trend to a gradual decline.

V3: Network Depth (Add one more layer):
It can be seen from the figure that the accuracy of the model after adding an additional layer does not improve, but the loss value increases. Therefore this model is not used.

V4：change activation function from relu to sigmoid
After changing the activation function, it is found that the accuracy of the model fluctuates too much and the loss value also declines.

CNN model training- V1
The accuracy and loss indicators of the training set and validation set have similar trends, indicating that the model has better generalization ability.The model performs best with epoch = 0.7, and the model achieves an overfitting after a final accuracy of 97.3% and a loss value of 6.45%.

V2: Network depth: Add one more layer
The model performs best when epoch=0.78, the final accuracy of the model reaches 97.15%, and the loss value is 0.0715

V3: Dropout layers are used to reduce the risk of overfitting:
The model performed best when epoch=0.9, the final accuracy of the model reached 97.5%, and the loss value was 0.0594.

The fully connected layer can increase the expressive ability of the model, make it better fit the training data, and improve the classification accuracy of the model. However, too many fully connected layers may also lead to overfitting of the model, so it needs to be adjusted according to the specific situation.

*V4: After adding a fully connected layer:
Using two or more fully connected layers can increase the expressive power of the model, allowing it to better fit the training data and improve the classification accuracy of the model.
After evaluation, the loss value of our model is 0.0582, and the accuracy rate is 0.9795.

V5: Changed stride, added padding='same': 
At this time, the loss value is 0.0592, and the accuracy rate is 0.9770
In summary, the model with 2 fully connected layers has the highest accuracy rate, so we decided to adopt this model.

# 5.Conclusion
## 5.1 The significance of the project for practical application
Improve the quality and yield of rice. By identifying different varieties of rice, you can choose a planting method suitable for different soil and climatic conditions. For biodiversity, it helpful for protection the biodiversity of rice, by identifying different varieties of rice, genetic pollution and species extinction can be avoided. In addtion, it will promote the market transaction of rice, by identifying different varieties of rice, the price and reputation of rice can be improved, and counterfeiting and shoddy products can be prevented.
## 5.2 Advantages and disadvantages of using cnn and simpleNN in the project
·CNN can extract the features of the image through the convolutional layer and the pooling layer, so as to better identify different varieties of rice. simpleNN needs to manually extract the features of the image, which may lead to loss of information or noise.
·CNN can process high-dimensional data, such as multi-spectral remote sensing images. simpleNN requires dimensionality reduction or selection of part of the data, which may affect the accuracy of predictions.
·CNN requires more computing resources and training time, while simpleNN requires relatively less. CNN also requires more data volume and quality, while simpleNN requires relatively less.
## 5.3 Improvement of model
Actually we can improve our CNN model in the future. There are several ways. For data processing, we can shuffle the training and validation data, which can avoid the influence of data order or distribution on the model. Second is adjust the learning rate, which can affect the model's convergence speed and performance. The learning rate can use a dynamic adjustment method, such as Adam optimizer, or manually set a smaller value, such as 1e-5. Besides, use data augmentation is a good method, which can expand the size and diversity of the dataset, improve the model's generalization and robustness
