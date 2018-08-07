# Beer-Analysis-Project

*Project Authors*: Dylan Edmonds, Quinn McLaughlin

**Summary**: 
    The goal of this project is to do as much of an in depth on the craft beer industry as possible. We will be looking at things such as alcohol content by state as well as price and will make explicit visualizations of the exploratory analysis and will be created in R using ggplot or in python using matplotlib and seaborn. Data collection will be done using web scraping tools such as get  and beautifulsoup  using the python programming languages. Modeling will only be done if the questions stated below cannot be answered during exploratory analysis phase and will be created using sklearn.
Questions:

   **Descriptive**:
* How is alcohol price related to alcohol content?
* What features determine the overall rating of beer?
* Does creation region affect price?
* Does creation region affect rating?
* How do the comments about a beer on a website correlate to the rating the beer received? E.g. sentiment analysis, etc... ? 
* Where are sales of a beer the greatest?

  **Predictive**:
* Can we predict the rating of a beer?
* Can we estimate the price of a beer?
* Can the alcohol content of a beer be predicted?
* Can we estimate sales of a season?
* Can we estimate sales in a region?

  **Prescriptive** :
* Are there any features a website has left out that could make a difference for a beer?
* What can be done to improve the rating of beer?
* What can be done to improve the sales of beer? 
* Should a brewery business be focused on alcohol content?
* How can beer features be optimized to increase sales?
    

**Process**:
    This industry analysis will follow a standard data science pipeline which includes, but is not limited to the following:
1. Collect/ Wrangle
2. Clean
3. Explore
4. Model
5. Interpret/ Validate
6. Deploy/ implement

**Data Sources**:
https://www.kaggle.com/nickhould/craft-cans
https://www.beeradvocate.com/
https://untappd.com/api/docs

   **Scraping tools**:
* Python
* beautifulsoup
* get
* Pandas - ‘pd.read_html()’

**Cleaning**:
    Most of the cleaning will be done using Python’s pandas or R’s tidyverse. Data will be cleaned to remove non-values from the data and will be properly formatted to ‘Year-Month-Day’ if the data received from the collection is time series. Joins will be done based on the exploratory analysis and modeling if it turns out two data sets can match. All feature types will be set to their corresponding type e.g. integer, string, float, etc… If there are multiple classifications in a feature the observations will be vectorized to simplify for models, otherwise those classifications will be kept as their original value for exploratory plotting. 

**Exploratory Analysis**:
    All exploratory plots will be created using base matplotlib initial. Otherwise, if there is found to be an informative graph that was created during exploration it will be implemented in seaborn or ggplot2. All descriptive statistics will be created using numpy in python or in R for its formatting explicit capabilities. Informative descriptives found during the exploratory analysis will be noted and mentioned at the end of the study to create reasoning model creation and answer reasoning. 
    
  **Exploratory tools**:
   1. Python
     * Pandas 
     * Matplotlib
   2. R
     * Tidyverse  
     * Purrr
     * Dplyr
     * Stringr
     * tidyr
     * Ggplot2
        
**Modeling/ interpretation**:
    Predictive models will only be created for questions that could not be answered during the exploratory analysis phase. All main machine learning will be done in python’s scikitlearn for its speed and convenience. Smaller models for subsetted data can be created in either scikitlearn or in R using caret. If a created model is found to be able and make an accurate prediction the model will then be followed up to be checked for overfitting, or underfitting if the model is underperforming. Similarly, does not appear to be under or overfitting appropriate error measures will be performed to measure the error of the predictions. For example, if the model is trying to predict classifications an AUC-ROC curve will be created. This will help keep a reliable standard for our modeling. Lastly, the training and the testing  data will be created using a 75 - 25 split of the main dataset(s) which is the default setting for scikitlearn’s “train_test_split” function.
    **Modeling tools**:
        * Python
          * scikitlearn
        * R
          * Caret
**Note**: 
    The exploratory analysis and modeling phase are the most likely steps to be intertwined and repeated. If a model discovers a previously unknown relationship, or if a model discovers a non-relationship, we will go back to the exploratory phase to either rephrase the question, or ask a completely new question. 

**Implementation**:
    At the end, the models created will be deployed only if a model is found to have a not be under or over fit and has reasonable accuracy measures. If a model meets this criteria the model will be deployed into a shiny app, or similar interface. The application created will be created to help communicate the exploration of the study and the findings of the models.   
    
   **Implementation tools**:
        1. R
          * Shiny
          * Plotly
          * Ggplot2
        2. Python
          * Bokeh
          * dash

