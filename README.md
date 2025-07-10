# A-HYBRID-APPRAOCH-FOR-PREDICTION-OF-IMMUNITY-LEVELS-USING-ENHANCED-RNN-DEEP-LEARNING-ALGORITHM

ABSTRACT 
 
RNN-based prediction of Corona virus immunity levels How did we suffer compromise the 
body's immune system? Because of its deadly effects and quick dissemination, COVID-19 
poses a threat to everyone on Earth. Immunization and vaccination are the most efficient 
methods of disease prevention. The pandemic will be stopped by the emergence of herd 
immunity against any lethal virus. The primary objective of this study is to create an improved 
herd immunity COVID-19 pandemic prediction model. The suggested ACHIO is paired with 
a hybrid RNN and LSTM to create a prediction model. The most crucial characteristics that 
improve the performance of the model are chosen using feature extraction and feature selection 
techniques. ACHIO then performs the best feature selection once the features have been 
retrieved statistically. 
To enhance model performance, epoch and neuron counts in the RNN and LSTM are 
optimized using ACHIO. The suggested model had 90.42% accuracy, 80% precision, 90.86% 
specificity, 89.53% sensitivity, 86.03% F1-Score, 17.20% FDR, 90.86% NPV, 10.47% FNR, 
and 9.14% FPR, among other accomplishments. To assess the effectiveness of the suggested 
model, a number of deep learning models, including DNN, RNN, CNN, RBM, LSTM, and 
RNN + LSTM, are compared. The outcomes show that the suggested model outperforms the 
current standard. 
Keywords: Immunology, Immunity levels, Recurrent neural network, LSTM, GRU, ML, Data 
driven predictions. 
 
 
 
 
 
 
 
 
9 
 
INTRODUCTION 
 
Societies throughout the world are still going through an extremely difficult period as a 
result of the new Coronavirus (COVID-19), which was originally identified in Wuhan, 
China, in 2019. The COVID-19 pandemic, with more than 118,000 cases in more than 110 
countries, was declared by the World Health Organization (WHO) on March 11, 2020. A 
number of nations, including Italy, Spain, France, the United States, and India, have 
experienced a rapid expansion of the disease that has wreaked havoc on their healthcare 
systems (1). It is essential for understanding and assisting decision-makers to slow down 
or halt its advancement to precisely model and estimate the breadth of verified and 
recovered COVID-19 instances. Real-time epidemiological data analysis is required since 
the COVID-19 pandemic has spread to be a worldwide pandemic and the populace needs a 
strong course of action. The world has been frantically fighting for its cause since the 
publication of COVID-19 (2). There were 24,631,906 confirmed cases globally as of 
August 27, 2020, of which 17,089,939 recovered and 841,310 resulted in death. 
Table 1: shows the topmost countries affected. The COVID-19 relate itself to the species 
 
To identify their differences based on their genetic profiles, a predictor is essential. The 
COVID Deep Predictor, an RNN-based technique for identifying infections in a sample of 
unidentified sequences, is suggested by this paper. It could be challenging to recognize 
SARS-CoV-2 at an early stage since it has comparable genetic structures and symptoms 
with other coronaviruses. This study describes a deep neural network-based automated 
technique for identifying SARS-CoV-2 patients. A deep bidirectional RNN that can 
recognize SARS-CoV-2 from its viral genomic sequences is suggested to be developed. 
 The usage of ensembles is the foundation for this work's core concept and originality. In     
supervised machine learning, recurrent or Time-Delay Neural Networks (TDNN) are used. 
The training set's quality has a big impact on them. The program will provide incorrect data 
10 
 
if we attempt to forecast anything that is noticeably different from the training set (follows 
a different physics or is impacted by new external factors). However, in a real application, 
we often don't know about it, thus we can't tell whether or not the forecast is accurate. The 
use of ensembles would be a fascinating strategy. In reality, neural networks with the same 
design and training data should be able to predict comparable outcomes in "predictable" 
regions, but "unpredictable" or "different" regions should have more sporadic data. 
Therefore, the average and standard deviation of the forecast may be determined using a 
TDNN ensemble. While the standard deviation provides a more reliable set of facts to 
utilize for decision-making, etc., the average forecast represents the value that is considered 
to be the most likely. 
Artificial intelligence (AI) is currently being used in several areas, including picture 
recognition, object classification, image segmentation, and deep learning techniques, to 
advance biomedical research. For instance, COVID-19 patients may get pneumonia 
because the virus spreads to the lungs. X-ray scans of the chest are frequently used in deep 
learning research to pinpoint the disease. The fine-tuned model, the non-fine-tuned model, 
and the scratch-trained model are the three deep learning models that have previously been 
used to distribute X-ray pictures of pneumonia. However, the majority of deep learning
based prediction models employ CT and X-ray pictures, which takes more time to train the 
model and extract the features from. 
 
 
 
 
 
 
 
 
 
 
 
11 
 
LITERATURE REVIEW 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
12 
 
METHODOLOGY 
Methodology and Dataset: 
The data across the countries provided on COVID-19 plays a vital role in the study. Thus RNN 
and LSTM based model predicts the immunity levels of the people based upon this data. This 
techniques used are accurate and efficient for placing a strong impact on the data management. 
These models are also used for the balancing of the data, hence this method are very much 
helpful in the classification of the imbalanced data set. A solid model construction is essential 
and training the model is crucial. 
Recurrent Neural Networks (RNNs) are a class of artificial neural networks specially designed 
to capture sequential dependencies within data. Unlike traditional feedforward neural 
networks, RNNs have loops that allow information to persist. This architecture enables RNNs 
to effectively process sequences of data, making them particularly suited for tasks such as time 
series prediction, speech recognition, natural language processing, and more. The key idea 
behind RNNs is the ability to maintain a hidden state that evolves over time as new input is 
processed, allowing the network to incorporate context from previous steps into its predictions 
or classifications. Flow chart shown  
 
Fig. Flow Chart 
13 
 
DATES                                                                      NO.OF.DEATHS 
2020-03-01 1 
2020-03-02 2 
2020-03-03 1 
2020-03-04 28 - - 
2022-05-04 3275 
2022-05-05 3545 
2022-05-06 3805 
2022-05-07 3451 
2022-05-08 3207 
Table:2 NO. OF deaths in year 
 Now, the data from DataFrame'df2' is plotted using a specific size and grey color, which 
suggests that 'df2' contains COVID-19 case data up until Feb 8, 2022. Following this, the 
forecast of COVID-19 cases from another DataFrame 'forecast df' on the same plot, using red 
to distinguish the predicted data is overlaid. This forecast data ranges from Jan 11, 2022 to Feb 
19, 2022. The visual representation of juxtaposes actual historical data against forecasted data, 
aiding in the analysis of the COVID-19 trend over time in India is the final output. 
Recurrent Neural Network: 
Recurrent neural networks (RNNs) are a broad category of models in computing that make an 
effort to imitate the operations of the human brain. Synaptic connections connect several 
"conceptual neurons" or "processing elements" in an RNN. RNNs differ from more generic 
feedforward neural networks in that they may produce self-sustaining temporal activation 
dynamics along their recurrent connection pathways even in the absence of input. This 
mathematically proves that an RNN is a dynamic system since feedforward networks are 
functions.  An RNN retains a nonlinear change in the input history in its internal state in 
response to an input signal. It offers the RNN a dynamic memory and enables it to comprehend 
temporal context data. represents the recurrent neural network's schematic design. RNNs 
14 
 
typically have three main structural components: input, hidden neuron, and activation function, 
which is represented by Equation. 
                                                       ℎ𝑡=tanh(𝑈·𝑋𝑡+𝑊·ℎ𝑡−1)  
Here, Xt is the input at time t, ht is the hidden neuron at time, U stands for the hidden layer's 
weight, and W stands for the hidden layer's transition weights. As the current and previous 
inputs are processed by the tanh function, the input and prior hidden states are combined to 
provide information. As a result, a brand-new hidden state is created, serving as the neural 
network's memory and storing information from the first network. 
 
Deep Learning for COVID-19 
Employed deep learning-based algorithms to predict the number of new Coronavirus (COVID
19) cases that will be reported in 32 Indian states and union territories. To predict the number 
of positive instances, Recurrent Neural Network (RNN)-based LSTM variants, including Deep 
LSTM, Convolutional LSTM, and Bi-directional LSTM, were applied to the Indian dataset. 
Convolutional Neural Network (CNN) can precisely estimate and identify the quantity of 
validated cases, according to Huang et al. The focus was on the towns in China with the highest 
number of reported cases, and a COVID-19 prognostication model based on the CNN system 
of Deep Neural Network (DNN) was proposed. To get the most trustworthy findings, learning 
models for the ensemble were built using three deep learning models: CNN, LSTM, and DNN. 
Utilizing RNN-based algorithms, including Long Short-Term Memory (LSTM) networks, to 
simulate immune system reactions to COVID-19 has demonstrated exceptional promise. 
Through the integration and examination of several datasets that include information on the 
immune system, viral load, patient characteristics, and treatment strategies, these models are 
able to forecast and clarify the dynamics of immunity levels in reaction to the virus. 
Our research has focused on LSTM optimization of RNN architectures to capture immune 
response subtleties and temporal interdependence. To ensure a strong basis for the model's 
15 
 
training and prediction abilities, this required painstaking data preparation, including 
normalization, feature engineering, and temporal aggregation. 
Our improved RNN deep learning system produces findings that indicate encouraging progress 
in predicting COVID-19 immunity levels. The precision with which the model predicts 
immunological responses and pinpoints putative indicators of immunity has important 
ramifications for treatment plans, vaccine research, and public health initiatives. 
Moreover, the interpretability of these models—made possible by methods such as feature 
significance analysis and attention mechanisms—offers insights into the crucial elements 
influencing the immune system's reaction to COVID-19. This knowledge facilitates the 
deciphering of the intricate dynamics of immunity, directs clinical judgments, and may help 
identify patients who are more susceptible to illness or require certain treatments.\ 
Predicting immunity levels with upgraded RNN-based deep learning models is a ray of hope 
in the ongoing global fight against COVID-19. To fully utilize these models and potentially 
shape pandemic response methods and future viral outbreak preparedness, further study, 
cooperation, and improvement are necessary. 
Long Short-Term Memory Network 
One of the deep learning approaches known as RNN automatically chooses the proper 
characteristics from the practice specimens and then provides activation from the previous time 
step as data for the current time step and the network's self-connections. By storing a wealth of 
past data in its internal state, RNN is suitable for data processing and has exceptional promise 
in time-series forecasting. It still has the drawback of vanishing and gradient-exploding issues, 
which necessitate lengthy practice sessions or ineffective practice. In order to assess the long
term dependency on the multiplicative passages that direct information and memory cell 
movement in the recurrent hidden layer, a long-short-term memory structure was developed in 
1997. The Long Short-Term Memory (LSTM) network has shown tremendous success as an 
improved Recurrent Neural Network (RNN) deep learning method. Our prediction power has 
been greatly enhanced by the LSTM's capacity to extract long-range relationships from 
sequential data, especially the complex dynamics of immune system parameters. 
An important factor in modeling the intricate temporal correlations seen in immune system 
data was the architecture of the LSTM, which was created to address the vanishing gradient 
issue and capture both short- and long-term patterns. We developed a very flexible and 
16 
 
successful predictive model by rigorously experimenting with and optimizing the LSTM 
parameters, including memory cell states, input and forget gates, and cell output. 
Our comprehensive preparation efforts, which included managing sequential immune system 
data, feature engineering, and data standardization, created a strong basis for the LSTM's 
training. Comprehensive metrics and validation procedures were used to assess the model's 
performance, which constantly showed that it was capable of predicting immunity levels with 
an impressive level of accuracy and dependability. 
Furthermore, the interpretability of the LSTM—made possible by methods such as attention 
processes and memory cell state visualization—offered significant insights into the variables 
affecting immune system activity and a more profound comprehension of the forecasting 
process. 
The LSTM-based improved RNN deep learning algorithm's successful implementation marks 
a major advancement in the creation of immunity level prediction models. Its potential uses in 
a variety of healthcare domains, including as illness prediction, therapy optimization, and 
customized medicine, might be extremely beneficial to medical professionals. 
Further investigation into LSTM improvements, interpretability, and model optimization offers 
encouraging opportunities to further our knowledge of immunity dynamics and support the 
creation of more precise and flexible prediction models for use in biological research and 
healthcare. 
 
 
Recurrent neural network ensemble used in this study 
Time-delay neural networks, which have a straightforward feed-forward design, are the kind 
of neural network employed. The neural networks in this instance featured one output layer 
17 
 
and two hidden layers. There is only one neuron in the output layer, which is the output. There 
may be an unlimited number of neurons in the two buried levels. Two distinct designs have 
been applied in this particular instance. The second has 4 neurons in the first layer and 2 in the 
second, compared to the first's two neurons per layer. Even though simpler designs have been 
studied, more complicated networks still achieved the best performances, as the reader will 
discover. The weeks (which are seasonal) and previous influenza instances have been used as 
two inputs. Five and ten time delays have each been tested in different combinations (each time 
delay is equal to one week, which is also the time step). The two neural network designs are 
depicted in Figure In specifically, the number k in Fig. reflects the neural network's time delay 
values, whereas the value t-p denotes the dataset's initial time slice. 
 
Architecture for recurrent neural networks. The two utilized architectures are depicted in the 
picture. Two neurons are located on each of the two hidden levels of the net in the first picture 
(top). The second picture (bottom) displays a second design with two hidden layers, the first of 
which has four neurons and the second of which has two neurons. In specifically, the number 
tp in figure indicates the initial time slice of the dataset, whereas the value k reflects the time 
delay values employed by the neural network. 
Problem Formulation: 
Utilizing the input sequences seen before, time scale forecasting seeks to predict a fixed-length 
series of predicted time scale values. In machine learning, delayed values are substituted for a 
portion of the input time-series sequence to help the input functions. The breadth or size of the 
frame is defined as the number of leading time levels. given a time-series with a single variable: 
                                                               TS(t)={s1,s2,s3,…,st} 
18 
 
the intention is to forecast the future k values of the sequence, ŷ = ŷ1, ŷ2, ŷ3, …, ŷk ≅ (st+1, 
st+2, st+3, …, st+k) utilizing the values of former conclusions. 
Dataset analysis and preprocessing  for prediction of immuni levels  
Investigate the dataset: Gain an understanding of its properties, organization, and goal 
variable (immunity levels). 
Statistical Synopsis: For numerical characteristics, compute descriptive statistics (mean, 
median, standard deviation); for categorical features, compute frequency counts. 
Data visualization: To see feature distributions and spot outliers, display histograms, box 
plots, or scatter plots. 
2. Managing Null Values: 
Determine Any Missing Data: Examine the dataset for any missing values. 
Choose between removing rows or columns with significant missing data or impute missing 
values using techniques like mean, median, or mode imputation. 
3. Engineering features: 
Feature Choice: Determine the pertinent characteristics that may be correlated with immunity 
levels. 
Develop Fresh Features: If current features have the potential to offer more predictive power, 
create new ones from them. For example, calculate percentages or ratios. 
 
4. Encoding Categorical Variables: Categorical Variable Handling: Use methods like label 
encoding or one-hot encoding to translate categorical variables into numerical representation. 
5. Normalization/Standardization: Scaling Numerical Data: To improve model performance 
and convergence, normalize or standardize numerical characteristics to bring them to a 
comparable scale. 
6. Split the dataset using the train-validation-test method. Make training, validation, and test 
sets out of the dataset. Usually, testing needs the remaining portion, 10-15% for validation, 
and 70-80% for training. 
7. Managing Unbalanced Information (if relevant): 
19 
 
Verify Class Imbalance: In order to rectify any imbalance in the distribution of immunity 
levels, methods such as weighted loss functions, undersampling, and oversampling should be 
taken into consideration. 
8. Feature Transformation and Scaling: Use Transformation: If the data is skewed, apply 
methods such as logarithmic transformation to make it more regularly distributed. 
9. Handling Outliers: Identification and Handling of Outliers Recognize and deal with 
anomalies by eliminating them or by using methods such as flooring or limiting their values. 
10. Correlation Analysis: Correlation Matrix: Determine how characteristics and the target 
variable are related by computing the correlation matrix. 
11. Revision and Validation: 
Verify Preprocessing Procedures: Make that the model is not subjected to bias or information 
leakage as a result of preprocessing activities. 
Repeat as Needed: Review and improve preprocessing processes as necessary based on 
model performance. 
Points to Think About: To avoid data leaking, make sure the preprocessing procedures are 
applied uniformly to the training, validation, and test sets. 
For repeatability, keep note of preprocessing techniques and transformations. 
Analyze how each preprocessing step affects the functionality of the model. 
