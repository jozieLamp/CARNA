# <img src="CARNA Logo.png" width="150"> CARNA
### Characterizing Advanced heart failure Risk and hemodyNAmic phenotypes using learned multi valued decision diagrams  

*Josephine Lamp<sup>1</sup>, Yuxin Wu<sup>1</sup>, Steven Lamp<sup>1</sup>, Lu Feng<sup>1</sup>, Sula Mazimba<sup>2</sup>*  
<sup>1</sup>Department of Computer Science, University of Virginia, Charlottesville, VA, USA  
<sup>2</sup>Department of Cardiovascular Medicine, University of Virginia, Charlottesville, VA, USA  

-------
**Core Code Repository: https://github.com/jozieLamp/CARNA  
Live Web Server with Risk Prediction: http://hemopheno.pythonanywhere.com/**

## Project Description:  
CARNA is an open source risk stratification and phenotyping tool for hemodynamic phenotypes in advanced heart failure.
Explicitly, our tool provides a hemodynamic risk score that takes in single point of care measures as input, 
and uses a diverse set of features (including invasive and composite hemodynamics and other clinical measures) 
to return a score of 1 to 5, indicating the probability of a specified outcome, such as mortality, rehospitalization, 
or readmission in 30 days. For example, a risk score of 5 (high) may indicate the patient has a
\> 40% probability of the specified outcome, whereas a score of 1 (low) may indicate a \< 10%
probability of the outcome. Our risk scores are learned using a machine learning methodology that takes advantage of the
 explainability and expressivity of Multi Valued Decision Diagrams (MVDDs), trained on the ESCAPE dataset, 
 which includes a rich and inclusive feature set of invasive hemodynamics, demographics, lab values, medications, 
 quality metrics and other health measures. In addition to providing a concrete score metric, our solution provides and 
 visualizes the learned patient phenotypes, which characterize the set of features and thresholds that were used to determine
  the resulting risk score.
  
## Algorithm Overview
### Preprocessing Steps:
The original data is contained in the Data folder under the Original DataFrames.
We have two datasets: one which has only invasive hemodynamics (Hemo) and the other which includes invasive hemodynamics, 
composite hemodynamics, diagnostic information, labs, medications etc. (All Data)
Preprocessing code is in the Score Label Preprocessing folder. Run the Data Preprocessing to complete the data preprocessing according to the following steps:
1. First combined all data variables into a single file
2. Perform main preprocessing steps, including removing outliers and normalizing the values (necessary
to reduce bias in machine learning training). 
3. Calculate additional composite hemodynamic factors from the data and add to file.

### Score Label Generation:
The risk score label sets are generated for each
of the four outcomes: Mortality, Rehospitalization,
Readmission in 30 days or All Outcomes. We use a risk score from 1 to 5 which characterizes
the probability with which an outcome will occur,
with 1 indicating low risk and 5 indicating high risk.
Score Label Generation code is in the Score Label Preprocessing folder.
To produce or risk scores, we perform the following steps:
1. For each dataset and outcome, combin the features using Sklearn Principal Component Analysis (PCA).
2. Use Sklearn K-Means Clustering to cluster the patients into 5 groups corresponding to each of the risk probabilities.
3. Assign score labels to each data record based on the cluster it belongs to.

### MVDD Learning:
Our MVDDs are learned via the following steps:
1. Learn a multi class decision tree classifier using sklearn's CART algorithm and the splitting criterion of gini index or entropy.
2. Transfrom the decision tree into an MVDD by performing an exhaustive search in which we replace the boolean edges (True/False) from the decision tree with logical operators (“and”, “or”) and calculate the accuracy of each diagram. From there, we select the MVDDs with the best accuracies.   

In order to maximize the training capabilities of our small dataset, we use a 5-fold cross validation, in which 80% of the data in the split is used for training and the other 20% is held out for validation purposes.
The machine learning process may be run in the Risk_Score_Training jupyter notebook, and
the MVDDGenerator and MVDD object classes are used to actually learn and produce the MVDDs.

## Running the Code
*To quickly start prototyping, run the Risk_Score_Training jupyter notebook file.*

The two other additional scripts you will need to work with are the MVDD Generator and the MVDD object files. 
MVDD Generator includes the back end functions used to learn, generate and visualize the MVDDs and the MVDD folder is an object structure that holds the details for the generated MVDDs.
In addition, a detailed breakdown of the file directories is included below.

Notes:
- Most of the machine learning heavy lifting is performed by Sklearn python packages.
- All of the MVDD graph generation uses graphviz to visualize the trees and networkx to actually build the graph structures.

## Required Libraries:
These can be easily installed into a Python environment such as by using PyCharm.

- dd	0.5.5
- graphviz	0.14.1
- matplotlib	3.3.1
- networkx	2.4
- numpy	1.19.1
- pandas	1.1.1
- pydot	1.4.1
- scikit-learn	0.23.2
- scipy	1.5.2
- sklearn	0.0	

## File Breakdown:
- Main Level:
    - Risk_Score_Training: Main runner class to perform the machine learning training and produce the visualized MVDDs
    - Main: Can be used to predict risk scores from patient data. Also the backend that interfaces with the webserver to return live risk scores.
    - Params: Contains the parameter settings used for the MVDD training. This include variable dictionaries, ranges of variables etc.
- Data: 
    - Original DataFrames: Original patient data before preprocessing
    - Preprocessed Data: Preprocessed data with derived supervised score labels as well as a Data Dictionary for ease of understanding feature abbreviations.
- Graphs:
    - Output directory for the generated graphs from training including the 5 fold ROC scores
- MVDD:
    - MVDD: MVDD Object class that stores the details about the MVDD structure
    - MVDD_Generator: Script to train, build and visualize the MVDDs
- Score Label Preprocessing: 
    - Contains the code to preprocess the data and derive the supervised score labels from the data
    - Feature Importances: Tracking of the best features learned via feature selection processes
    - Figures: Output directory for the generated images from clustering and training
    - Create Score Labels: Jupyter Notebook to produce and assign the risk scores for each of the datasets and outcomes
    - Data Preprocessing: Jupyter Notebook to perform all the data preprocessing steps
-TreeFiles
    - Contains the generated MVDDs from training. The MVDD python data structures are stored in .sav files so the models can be reloaded later for prediction purposes,
     and we also save pdf and png images of the MVDD structures
    - TreeTraining: Contains all the exhasutively generated MVDDs


## License available in LICENSE.txt






