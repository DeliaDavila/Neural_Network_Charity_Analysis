# Neural Network Charity Analysis

## Overview of Project
### Purpose
This project was done for Alphabet Soup, a company who funds other organizations. The company is hoping to improve the effect of their funding by determining which organizations are able to use the funding well.
To complete the project, we will use machine learning and neural networks to analyze historical data from the company about funded companies. The goal of building a machine learning model is to predict whether applicants will be successful if funded by Alphabet Soup.

### Background
Alphabet Soup has historical data from the companies they've funded. The company provided information the form of a CSV that contains informtation about previously funded companies who are asking for additional funding. 

## Process
### Review variables
The first step was to evaluate the dataset provided and determine what fields are useful in building the model. We determined a target variable, feature variables to use in building the model, and extraneous variables that were not useful to machine learning. Here are those variables:
* Main Target variable
    * IS_SUCCESSFUL
* Features 
    * APPLICATION_TYPE
    * CLASSIFICATION
    * AFFILIATION
    * USE_CASE
    * ORGANIZATION
    * STATUS
    * ASK_AMT
* Other variables (Not helpful to the model or potentially too noisy)
    * NAME
    * EIN
    * SPECIAL_CONSIDERATIONS
    * INCOME_AMT

### Compiling, Training, and Evaluating the Model
The initial state of the model was created with three layers, with 8 neurons in the first layer and 5 neurons in the secont layer. The "relu" (Rectified Linear Unit) activation was used on the first two layers, and a sigmoid activation was used on the outer layer.

```
number_input_features = len(X_train[0])
hidden_nodes_layer1 =  8
hidden_nodes_layer2 = 5

nn = tf.keras.models.Sequential()

# First hidden layer
nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer1, input_dim=number_input_features, activation="relu"))

# Second hidden layer
nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer2, activation="relu"))

# Output layer
nn.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))
```


This did not achieve the desired accuracy of 75%.


![Acc1](https://github.com/DeliaDavila/Neural_Network_Charity_Analysis/blob/main/Images/Acc1.png)

To increase the model's performance, I went through a number of steps. The most successful improvement was from adjusting the binning, a method of grouping data into fewer chunks. In this case, the improvement was in grouping more of the data with smaller numbers into an "other" category. This allowed the model to compare fewer chunks of more comparably sized data.

![Acc2](https://github.com/DeliaDavila/Neural_Network_Charity_Analysis/blob/main/Images/Acc2.png)

Other things that were attempted: dropping additional columns of data, manually adjusting the layers or neurons used per layer, and changing the activation functions used. These had limited benefit and in some cases lowered the accuracy.

![Acc3](https://github.com/DeliaDavila/Neural_Network_Charity_Analysis/blob/main/Images/Acc3.png)

To get closer to the desired accuracy, I set up a model within a model using Keras Tuner. This function runs trials to determine the optimal settings rather than the manual trial and error. This did improve the accuracy of the model:

![Accuracy_KT](https://github.com/DeliaDavila/Neural_Network_Charity_Analysis/blob/main/Images/Accuracy_KT.png)


## Summary
The Neural Network machine learning model, powered by relu and sigmoid activation functions and tuned using Keras Tuner, was reasonably able to predict whether companies would be successful. However, the accuracy was not extremely high, as would be expected for major companies using machine learning.

The data provided did not have a large number of fields that could be input for the model to review. Additionally, some fields had many blank values, making them not useful for company comparison. Adding more information or adding more complete information might allow the model to make more accurate predictions.

Additionally, adding more information to describe what "success" means might make for a better outcome. In the data, "Is Successful" is the only measure. Providing more outcome material might make for a better ability of the neural network to learn how to calculate success. Some suggestions are project completion rates or duration, number of people affected, number of previous successful campaigns, etc.

