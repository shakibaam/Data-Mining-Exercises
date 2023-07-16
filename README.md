# Data Mining Exercise

There different projecst in this repository that each cover a different section in the datamining world; from preprocessing and analyzing data to classification, clustering, etc.

## HW1: Analysis Iris

Data preprocessing is a crucial skill for any data scientist or machine learning engineer. In this exercise, we explore data preprocessing using the popular libraries Pandas and Scikit-learn.

In a real-world data science project, data preprocessing plays a vital role in the success of a model. Properly preprocessed data and well-engineered features contribute to better model performance compared to poorly preprocessed data.

For this exercise, we use the Iris dataset. Each flower in the dataset is described by four features:
1. Sepal length in cm
2. Sepal width in cm
3. Petal length in cm
4. Petal width in cm

The following preprocessing steps have been performed:
1. Removing NaN (Not a Number) data
2. Converting categorical data to numerical data
3. Normalization
4. Principal Component Analysis (PCA)

### Preprocessing Steps

1. **Removing NaN Data:**
   We handled missing data by removing any instances with NaN values.

2. **Converting Categorical Data to Numerical Data:**
   Categorical data was transformed into numerical values for further processing.

3. **Normalization:**
   The data was normalized to bring all features to a similar scale, which helps in better model convergence.

4. **Principal Component Analysis (PCA):**
   PCA was used to reduce the dimensionality of the data and retain essential information.

### Box Plots

We have created box plots for each of the four features in the Iris dataset. Box plots give us an overview of the distribution and variability of the data for each feature.

![Box Plot for Sepal Length](https://github.com/shakibaam/Data-Mining-Exercises/blob/main/Analysis%20Irisi/boxplot.png)


## HW2: Classification

In this exercise, we designed several neural networks for classification tasks and evaluated their performance. The networks were tested on the `make_circles` and `fashion_mnist` datasets. For the `fashion_mnist` dataset, we also plotted the confusion matrix.

### make_circles:
This dataset consists of concentric circles of two classes, making it a non-linearly separable classification problem.

### fashion_mnist:
Fashion MNIST is a dataset of fashion items, containing 10 classes of various clothing items, each represented by 28x28 grayscale images. The picture below show the confusion matrix:

![Confusion_Matrix](https://github.com/shakibaam/Data-Mining-Exercises/blob/main/Classification/confusion%20matrix.png)

## HW3: Clustering and Association Rule Mining Project

In this project, we focus on implementing algorithms related to clustering and association rule mining.

### K-means Algorithm

One of the simple and widely used clustering algorithms is the K-means algorithm. Our main goal in this exercise is not to implement the K-means algorithm from scratch, but rather to get familiar with it.

To determine the optimal value of K (number of clusters), we use the elbow method. The elbow method helps us find the "elbow point" in the plot, which indicates the best value of K where the distortion starts to level off.

![elbow](https://github.com/shakibaam/Data-Mining-Exercises/blob/main/Clustering%20and%20association%20rule/elnow.png)


### K-means for Digit Dataset

For this task, we preprocess the digit dataset by converting each 64 numbers into an 8x8 matrix, and we consider the number of clusters to be 10. Then, we calculate the centroids for each cluster.

The resulting centroids accurately represent the labels of their corresponding clusters, as shown in the visualization below:

![Centroids Visualization](https://github.com/shakibaam/Data-Mining-Exercises/blob/main/Clustering%20and%20association%20rule/Digits.png)

### K-means Image Compression

One of the practical applications of the K-means algorithm is image compression. In this exercise, we applied the K-means algorithm for image compression to reduce the size of an image.

#### Original Image - Bird

![Original Bird Image](https://github.com/shakibaam/Data-Mining-Exercises/blob/main/Clustering%20and%20association%20rule/bird.jpg)

#### Image Compression Process

The image of a bird was used for this experiment. The goal was to reduce the number of colors in the image while preserving its visual content. Here are the steps we followed for image compression:

1. Convert the image from its RGB representation to a list of pixels, each with its corresponding RGB values.

2. Apply the K-means clustering algorithm to group the colors into 4 clusters. The centroids of these clusters will represent the new colors.

3. Replace each pixel's RGB values with the RGB values of the cluster's centroid to compress the image.

#### Compressed Image - Bird with Reduced Colors

![Compressed Bird Image](https://github.com/shakibaam/Data-Mining-Exercises/blob/main/Clustering%20and%20association%20rule/image%20compression.png)

The compressed image shows the bird with reduced colors. Even though the number of colors is reduced, the visual content of the image is still preserved.

### DBSCAN Clustering

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is another clustering algorithm commonly used for clustering data. It is capable of clustering data with different shapes and can also identify noise and anomalies present in the data.

In this section, we will use the DBSCAN algorithm to cluster the two datasets mentioned in the "Complex Clustering" section.

#### DBSCAN Algorithm and Parameters

The DBSCAN algorithm does not require specifying the number of clusters beforehand, which makes it more flexible for various datasets. The main parameters of this algorithm are:

1. Epsilon (eps): The maximum distance between two data points for them to be considered in the same neighborhood.
2. MinPts: The minimum number of data points required to form a cluster.

Finding the optimal values for epsilon and MinPts is essential for obtaining accurate clustering results. For epsilon, we can try different values and evaluate their impact on the clustering quality. Additionally, we can use the KNN (K-Nearest Neighbors) distance method to automatically determine the appropriate epsilon value based on the average distance to k-nearest neighbors.

Determining MinPts is not automated, but we can follow some general guidelines based on the characteristics of the dataset to find suitable values.

#### Evaluating Different MinPts

In this project, we will experiment with various MinPts values to observe the effect on the clustering results. We will plot the clustering output for different MinPts values and analyze the clusters' quality based on our dataset's specific features.

Remember that different MinPts values might lead to different clustering outcomes, and it's crucial to choose the one that best fits the dataset and the desired level of clustering granularity.

![dbscan](https://github.com/shakibaam/Data-Mining-Exercises/blob/main/Clustering%20and%20association%20rule/DBSCAN.png)

### Association Rule Mining using Apriori Algorithm

Association rule mining is a technique that reveals the relationships and mutual dependencies among a large set of data items.

A common example of association rule mining is "Market Basket Analysis." In this process, based on the different items placed in customers' shopping carts, the buying habits and patterns of customers are analyzed. By identifying the associations between products, we can discover recurring patterns during purchases.

Three important parameters:

1. Support: It measures the popularity of an itemset based on its occurrence frequency in transactions.
2. Confidence: It indicates the likelihood of buying item y if item x is purchased. (x -> y)
3. Lift: It combines the two parameters above.

In this exercise, we will implement the Apriori algorithm, one of the most popular and efficient algorithms for association rule mining.

Explaining Lift:
As mentioned in the last section, if the lift value is greater than 1, it means that the occurrence of the consequent has a positive effect on the occurrence of the antecedent, and they appear together more than expected. On the other hand, if the lift is less than 1, it indicates that the consequent and antecedent appear together less than expected.

Apriori Algorithm:
The working of the Apriori algorithm is as follows: A minimum support value is set, and iterations are performed with frequent itemsets. If subsets and subgroups have support values less than the threshold, they are removed. This process continues until no further pruning is possible.

In this exercise, we will apply the Apriori algorithm to the Hypermarket_dataset, which contains purchase orders of individuals from grocery stores.

#### Data Preparation:
To start the work, we need to prepare the dataset into a sparse matrix, where products purchased will be in columns, and order numbers will be the index.

#### Identifying Frequent Patterns:
By applying the Apriori algorithm with a minimum support of 0.07, we will generate all the frequent patterns.

#### Extracting Association Rules:
We have also written a function that, given the confidence and lift as inputs, displays the resulting association rules.







