
Second assignment solved in "**Artificial Intelligence: Knowledge Representation and planning**" course at Ca'Foscari University, see <ins>**DOCUMENTATION/REPORT.pdf**</ins> for a project explanation. <url>aa</url>

**Requirements**:

Write a spam filter using discrimitative and generative classifiers. Use the Spambase dataset which already represents spam/ham messages through a bag-of-words representations through a dictionary of 48 highly discriminative words and 6 characters. The first 54 features correspond to word/symbols frequencies; ignore features 55-57; feature 58 is the class label (1 spam/0 ham).
- Perform SVM classification using linear, polynomial of degree 2, and RBF kernels over the TF/IDF representation.
Can you transform the kernels to make use of angular information only (i.e., no length)? Are they still positive definite kernels?
- Classify the same data also through a Naive Bayes classifier for continuous inputs, modelling each feature with a Gaussian distribution.
- Perform k-NN clasification with k=5.

Provide the code, the models on the training set, and the respective performances in 10 way cross validation and  explain the differences between the three models.

P.S. you can use a library implementation for SVM, but do implement the Naive Bayes on your own. As for k-NN, you can use libraries if you want, but it might just be easier to do it on your own.
