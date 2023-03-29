### Software And Tools Requirements

1. [Github Account](https://github.com)
2. [VSCodeIDE](https://code.visualstudio.com/)
3. [GitCLI](https://git-scm.com/book/en/v2/Getting-Started-The-Command-Line)

# Project and dataset overview
* This project aims to build a model that can detect whether a person claiming insurance is genuine or not. The model uses a dataset that includes various features such as the age of the claimant, the duration of their policy with the insurance agency, and policy-related information such as the number of policies, the policy bind date, state, policy deductible, combined single limit, annual premium umbrella limit, etc. Additionally, the dataset includes information about the claimant's insurance details such as their zip code, gender, education level, occupation, as well as the incident whereabouts and details. Finally, the dataset also contains information about the total number of claims made and automobile details.

* To build the model, we used various machine learning algorithms such as logistic regression, decision tree, random forest, and XGBoost, and compared their performances to select the best-performing algorithm. We also preprocessed the data to handle missing values, normalize the numerical data, and encode the categorical data using techniques such as one-hot encoding.

After training and testing the model, we evaluated its performance using metrics such as accuracy, precision, recall, and F1 score. The model achieved an accuracy of approximately 80.4% with Xtreme Gradient Boosting technique, which suggests that it is able to detect insurance fraud based on the input features.

Overall, this insurance fraud detection model can be a valuable tool for insurance companies to reduce losses due to fraud.
![](https://github.com/Lak2k1/IFD/blob/main/2023-03-28%2000-52-16.gif)
