# Regression_Insurance_Analysis
Authors: Finn Tan and Sara Zylfo

We are a healthcare consultancy company hired by an European Life Insurance firm who is looking to expand into the US market. As such, we have been tasked to provide some statisical insights into the main drivers of life expectancy which will be a crucial input in driving our client's pricing strategies.

Coupled with the above, we were also required to suggest 5 US states that would be best to break into, which were ultimately conducted by finding a subset of US states with high life expectancy but with low life insurance coverage. 

In order to accomplish this, we have used the 2019 County Health Rankings National Data which provides a comprehensive numerical data on counties':

    - Health Outcome
    - Health Behviours
    - Clinical Care
    - Social & Economic Factors
    - Physical Environment
    - Demographics
    
For better readability, we have split our findings into several parts including:

    - 'index - Data Cleaning.ipynb'
    - 'index - EDA & Visualization.ipynb' 
    - 'index - Regression Modelling, Model Evaluation & Conclusion.ipynb'
    - 'functions.py' - include customized functions

### Process 
 
Given the sheer size of the features involved and for better interpretation for our final user ('Life Insurer'), it is crucial to narrow down the number of variables. There are several ways to do this but the ones we are about to list below are definitely not exhaustive.

    - Baseline Method: Using all available variables
    - Naive Selection: using top features that are correlated to Life Expectancy
    - Filter Method: dropping low variance features followed by removing highly correlated features
    - Stepwise Selection: Adding features with p-values below certain threshold and dropping those
    - Recursive Feature Elimination: sklearn's function of greedily choosing
    - Lasso: use GridSearch to find the best penalizing parameter 'alpha' for the Lasso algo. We will then select features that have not been shrinked to 0

Once we get all the features selected by each method above, we pass those into Statsmodel's OLS function. Subsequently, we will select our most prefferred model by comparing their R2 scores, AIC (model complexity) and also consider the number of features included, which is the primary consideration here.

Post model selection, we will then check if the chosen model satisfies the assumptions of a regression; no multicollinearity between selected features, homosceasticity and normality of errors. In the end, we shall evaluate the model if it fits the purpose for our final user.

![Repo List](summary_png/summary_models.png)

    - Paste summary table
    - Paste map
    - Paste state table suggestion
    - Paste VIF, assumption normality,  

### Method Evaluations

One of the pitfalls of the above methods is that we might have potentially ignored other useful features that might be of use to our final user. 

Coupled with the above, the previously contemplated model might include various highly correlated features, violating the assumptions of a regression model. As such, let's try out best to find a middle ground by using by:

- dropping features with no or low variance. These features typically do not add much predictive value in a model
- dropping features which are highly correlated

Let's start of by dropping features with no or low variance. We can utilize Sklearn's 'variance_threshold' tool to solve this.


### Conclusion, Limitations and Future Work
To conclude, it is important to emphasize that whilst we have chosen the model above, there are many other methods out there which may result in a better model. That said, the above model should at least provide our end user with the crucial indicators for estimating life expectancy.

For future work, we will look to explore:

    - investigate performance of forward / backward selection, Ridge regression, interactions and polynomials
    - user other models apart from OLS, ie RF, CART, etc
    - investigate indirect correlations between features
 
