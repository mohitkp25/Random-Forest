1. Objectives:
-------------------
The goal is to implement the random forest algorithm from scratch in python with Wine Quality Dataset. It shows how ensemble learning improving predictive performance and reduce the overfitting issue by combining N number of Decision Tree through bagging and feature randomness.

2. Assumptions Made:
-------------------
-It is assumed that the Wine Quality dataset (UCI Id: 186) is accurate and correct representative of actual wine data.
 -The Quality of Wine score the rating 0 to 10 are convert into binary classes where quality of wine check.
        • 1 (Good Quality): Quality is greater than 5.
        • 0 (Low Quality): Quality ≤ 5
 -It is assumed that there is enough balance for evaluation when 80% of the data is used for training and    20% for testing.
 -The performance of random tree increases as number of tree increase but eventually stabilize because of diminishing return.

3. Resources Used:
-------------------
- ucimlrepo – for fetching the UCI Wine Quality dataset.
- NumPy Pandas (numerical computations)
- Scikit-learn (dataset split, preprocessing utilities only)
- Matplotlib (visualizations, learning curves, prediction plots
  
4. Data Preprocessing:
-------------------
-Dataset Source: UCI Machine Learning Repository(ID: 186) – Wine Quality Dataset
-Original data shape: 4898 instances × 11 features
-Target Variable: quality of wine  in integer rating between 0–10
-Transformation: Binary conversion applied
  binary = np.where(y > 5, 1, 0)
-Feature Matrix (X): Numerical features such as acidity, sugar, pH, sulphates, and alcohol.
-Train-Test Split:  The dataset split into 80% training, 20% testing.
-All features are numerical so no missing values or scaling adjustments were needed.

5. Implementation Details:
-------------------
-Decision Tree (Base Model)
• Implemented a custom Decision Tree Classifier class supporting:
  o	Gini impurity for node splitting
  o	Random feature selection for splits
  o	Recursive tree growth with stopping criteria (max_depth, min_samples_split)
• Each node stores information about:
  o	Impurity, number of samples, predicted class, feature index, etc.
-Random Forest (Ensemble Model)
• The Random Forest Classifier is built by combining multiple Decision Trees.
• Key Steps (Functions):
  o	Bootstrap Sampling: Each tree is trained on a random sample with replacement of the training data.
  o	Feature Randomness: Each split considers a random subset of features.
  o	Aggregation: Final prediction  is based on majority voting from all trees.
• Hyperparameters Used in training and evalution:
  o	Number of Trees (n_trees): 50
  o	Maximum Depth (max_depth): 10
  o	Features per Split (n_features): √11 ≈ 3

6. Evaluation and Results:
-------------------
Observation:
•	The Random Forest perform best on the Decision Tree on all metrics.
•	The use of bagging and feature randomness reduced overfitting and improved generalization effectively.

Accuracy vs. Number of Trees
The model was trained with different numbers of trees: [1, 5, 10, 25, 50, 100].

The following trend was observed:
N_Trees	Test Accuracy
1	0.7207692307692307
5	0.7676923076923077
10	0.7807692307692308
25	0.7992307692307692
50	0.7938461538461539
100	0.7961538461538461

Interpretation:
 The  accuracy and stability  is directly proportional to the number of trees. The stability increases with the increase in number of trees and get stabilized around 50 to 100 trees. This concluded the law of diminishing returns which states that after a certain threshold the gain is not proportionate to the extra added trees.
Feature Importance Analysis
•	Feature importance was calculated by averaging Gini impurity reduction across all trees.
•	Top influential features for predicting wine quality were found to be:
  1.	Alcohol
  2.	Volatile Acidity
  3.	Sulphates
  4.	Citric Acid
  5.	Total Sulfur Dioxide


                === Performance Comparison ===
                           Accuracy   Precision    Recall       F1-Score
Random Forest   0.798462   0.812709   0.885784    0.847674
Decision Tree     0.766923   0.814010    0.818955     0.816475

 

 

