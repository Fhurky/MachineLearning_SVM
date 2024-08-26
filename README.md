# MachineLearning_SVM

kernel="rbf"
Definition: This parameter specifies the kernel type to be used in the Support Vector Machine (SVM) algorithm.
Details: The kernel function determines how the input data is mapped into a higher-dimensional space. The "rbf" kernel stands for "Radial Basis Function," which is a popular kernel used in SVMs. The RBF kernel is particularly useful for non-linear classification tasks because it can handle complex relationships between the features.
Purpose: The "rbf" kernel allows the SVM to create non-linear decision boundaries, making it suitable for datasets where classes are not linearly separable.

C=0.1
Definition: The C parameter is the regularization parameter in the SVM algorithm.
Details: C controls the trade-off between achieving a low training error and a low testing error (i.e., generalization). A smaller C value applies stronger regularization, leading to a simpler decision boundary but allowing more misclassifications. Conversely, a larger C value applies less regularization, resulting in a more complex decision boundary that fits the training data more closely.
Purpose: With C=0.1, the model is regularized more strongly, which can help prevent overfitting by allowing some misclassifications in the training data for better generalization on unseen data.
These parameters allow the SVM model to be customized to fit the complexity of the data and to balance between bias and variance.