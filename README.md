# Application and analysis of machine learning algorithms into predicting wine quality and type
This project uses R to analyse the Vinho Verde wine dataset, focusing on predicting wine quality and type. It includes exploratory data analysis, unsupervised learning, as well as regression and classification techniques. This description will incorporate sections from the project report, also written by me as part of the project.

## Background
Economically, the wine industry relies on consumers' perception of quality and taste to influence market demand and trade. While culturally, wine quality reflects the craftsmanship and dedication of winemakers, embodying the unique terroir and traditions of each specific region.
The wine quality dataset from UCI Machine Learning Repository focuses on red and white wines from the Vinho Verde region in Portugal. It includes 11 measurable and 2 non-measurable variables related to physiochemical testing and sensory assessments. We will be exploring the use of regression analysis to predict wine quality, classification analysis for wine type and unsupervised learning for feature selection in both analyses.

## Approach
### Data Preprocessing
The datasets from the UCI Machine Learning Repository are considered clean, but it's safer to verify this yourself. To align with the project objectives, a wine type variable (red or white) is added to the original variables, and the red and white wine datasets are merged into a single dataset labeled "wine."
No missing values were found but 1177 duplicate entires were identified. However, according to the original paper, each row represents a distinct wine sample. Hence, there is no need to remove these duplicate entries.

### Exploratory Data Analysis
Descriptive summary statistics give an idea of data characteristics. It is important to note that the dataset contains more white wine than red wine. Complications from this imbalance will be discussed further below.
![image](https://github.com/user-attachments/assets/b9a37a3f-d85f-47ff-991a-2800f2564ebd)

Correlation assesses how changes in 1 variable correspond to changes in another. From the plot below, most variables seem to be neutral, with total and free sulfur dioxide being strongly positive while alcohol and density are strongly negative related. However, correlation does not imply causation. It is necessary to determine underlying relationships through additional analysis.
![image](https://github.com/user-attachments/assets/f557317e-173c-4fc1-901b-2f4a0b0706b1)

Variable importance is used to identify and provide insights into which variables have substantial impact on the outcome of interest. For wine quality, alcohol, density and volatile acidity have the highest impact. For wine type, the result was inconclusive. Further analysis was needed.
![image](https://github.com/user-attachments/assets/c58a11af-28ff-4a77-b556-f9854769d20e)

Plots and graphs can be used to visualise relationships and variable distributions. This project utilised QQ plots, density plots and boxplots. While some variables suggest right-skewness, it is difficult subjectively determine the relationship between variables.
![image](https://github.com/user-attachments/assets/b0cf2cb9-c38e-4692-be1d-b4c9d2708b62) ![image](https://github.com/user-attachments/assets/c414f0be-c685-4206-88a3-ddc70cc4256b)
#### Results
EDA was inconclusive in providing clear insights for determining feature importance and selection, more analyis is needed. As such, unsupervised learning algorithms will be used.

### Unsupervised learning
Dimension reduction techniques such as PCA can reveal data structures and relationships through analysing variance explained by principal components. This helps identify important features by capturing underlying patterns, aiding in feature selection for regression analysis of wine quality prediction. 4 principal components contain 73% of total variance. As such, the 4 variables that have the highest contribution to variance (in red) will be selected.

![image](https://github.com/user-attachments/assets/6e9ffc03-fa07-48e9-a8f4-e9322d30dc94) ![image](https://github.com/user-attachments/assets/17bfe005-fa24-4885-87d9-c811bedff830)

Clustering techniques like K-means can identify homogenous population groups that contribute most to data structure and variation. In theory, these groups of features can be used to build predictive models for machine learning. As such, K-means is chosen to guide the feature selection for classification analysis of wine type prediction. THe optimal number of clusters (K) was found to be 3 using the elbow method. Feature importance can be identified based on their association with the clusters. These features, identified by their maximum centroid values within each cluster, are likely to be crucial for distinguishing between wine types.

![image](https://github.com/user-attachments/assets/ebc1c0c0-b144-4900-9c30-ca93d56c11df) ![image](https://github.com/user-attachments/assets/dbd1bc86-015b-454e-b15c-43d2b872a215)
#### Results
Using PCA, the 4 variables with the highest contribution are density, type, total sulfur dioxide and residual sugar. Variables selected based most representative in k-means clusters are alcohol, citric acid, residual sugar, free sulfur dioxide, total sulfur dioxide, density, fixed acidity, volatile acidity, chlorides, pH and sulphates. Note that the selection of variables done using unsupervised learning algorithmns did not align with variable importance in EDA.

### Regression analysis
Multi-linear regression, ridge regression and CART algorithms were built and tested using error measures of MSE and RMSE. A final model from each (with the lowest values in error measure) will be chosen for the final comparison. 
The final MLR model used all variables and 10-fold cross validation as parameter tuning. Final ridge model used the minimum cv error lambda value while the best performing CART was pruned at the minimum cross-validated error.

#### Results
Multi-linear regression slightly outperformed other models in predicting wine quality by lowest error (RMSE: 0.7339).

![image](https://github.com/user-attachments/assets/7cef35ab-ba42-4bf8-8806-33b7abc1f151)

### Classification analysis
Logistic regression, naive bayes and random forest models were built and tested. Model performance was measured by accuracy, precision and sensitivity. Similarly, a final model from each (with the highest values in model performance) will be chosen for the final comparison.
The final LR model removed insignificant variables. Final NB model used all variables and used Laplace smoothing value of 0, while the best performing RF had parameter tuning with mtry = 2 and inconclusive ntree.

#### Results
Random forest was the best performing model in predicting wine type by highest model performance (99.7% accuracy, 99.7% precision and 99.9% sensitivity). 

![image](https://github.com/user-attachments/assets/d9450321-b8d9-424d-b402-7fb4f42ac326)

## Conclusion
In conclusion, this report explored various aspects of wine quality and type prediction through a comprehensive analysis of the Vinho Verde wine dataset. EDA provided insights into the dataset's characteristics, though relationships between variables and feature selection were challenging to identify. Unsupervised learning, such as PCA and K-means, aided in feature selection. Multi-linear regression and random forest models performed best after parameter tuning and identifying significant variables. However, limitations remain in these analyses.

## Further exploration
Despite being best performing models, multi-linear regression model still did not perform well with 0.733 RMSE and random forest model with 99% accuracy may be biased towards white wine. Presumably due to unbalanced data, majority of white over red wine and majority of 5 and 6 for quality may have affected model performance and interpretation. Evidently, further exploration into data balancing and additional data collection—beyond just common physiochemical tests—could enhance the robustness and performance of predictive models.
### For regression analysis
To improve model performance, further data preprocessing (e.g., outlier removal, feature scaling) could address data irregularities. Additionally, exploring different hyperparameters and regularization techniques for ridge and CART models may lead to better configurations.
### For classfication analysis
To enhance effectiveness, optimizing parameters like tree depth and minimum node size can improve performance and reduce overfitting. Exploring different hyperparameters and regularization methods for other models may also uncover better configurations.

## Full report
For a more detailed explanation of the methodology and results, specifically contrasting against other research, please refer to the full report in the docs/ folder.
