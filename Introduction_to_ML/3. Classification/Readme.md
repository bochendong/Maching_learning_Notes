# Classification

## Difference between Regression and Classification
<div align=center>
    <img src ="img/clavsreg.png" width="600" height ="200"/>  
</div>

## Logistic Regression
A probabilistic classifier which estimates $Pr(Y=1|X=x)$, i.e. $p(x)$ or $p(x;b)$

$$
p(x;b) = \frac{1}{1 + e^{-b_0+b_1x}}
$$

## Log Likelihood

Data = $\{(x_1, y_1), (x_2, y_2), \cdots, (x_n, y_n)\}$

$$
\begin{aligned}
l(b,data) &= \sum_i \log(Pr(Y=y_i|X=x_i; b))\\
l(b,data) &=
\left\{
\begin{aligned}
p(x_i; b) & ,y_i =1 \\
1 - p(x_i; b) &  , y_i = 0
\end{aligned}
\right.\\
l(b,data) &=  \sum_i \log[y_ip(x_i;b) + (1 - y_i)(1 - p(x_i; b))]
    
\end{aligned}
$$

## Evaluation:

### Decision Boundary
- Likelihood good for training, but difficult for evaluating a classifier
- Instead we usually focus on “how many predictions were right vs. wrong”
- How do we know right vs wrong? We need to threshold and create a decision boundary
- Example:
$$
\hat{y} =
\left\{
\begin{aligned}
1 & , Pr(Y = 1 | X = x) \geq 0.5 \\
0 & , Pr(Y = 1 | X = x) < 0.5
\end{aligned}
\right.
$$

<div align=center>
    <img src ="img/db.png" width="500" height ="240"/>  
</div>


### Ranking
- In some cases we can rank data samples from “most positive” to “most negative”
<div align=center>
    <img src ="img/rank.png" width="500" height ="100"/>  
</div>

### Probabilities
- In logistic regression our classifier produces probabilities. What if it mis-classifies
with high confidence?
- One solution: penalize misclassifications with high confidence more than
misclassifications with lower confidence

## Confusion Matrix (2 Class)
<div align=center>
    <img src ="img/cm.png" width="300" height ="250"/>  
</div>

- **TP = “True Positive”.** i.e. your classifier predicted 1; it was correct.
- **FP = “False Positive”.** i.e. your classifier predicted 1; it was incorrect (Type I error). 
- **FN = “False Negative”.** i.e. your classifier predicted 0; it was incorrect (Type II error). 
- **TN = “True Negative”.** i.e. your classifier predicted 0; it was correct.

## Accuracy
- Say we have a dataset of 950 negative-class samples, 50 positive-class samples
- If we lower our positive class threshold, our logistic regression accuracy changes
<div align=center>
    <img src ="img/acc.png" width="600" height ="200"/>  
</div>

- Generally, if we allow more samples to be negative (increase the threshold), both true and false positive will decrese, in the meantime, both true and false negative will increase.


## Measurement:

- Precision: What percentage of all positive predictions were correct?
$$
Precision = \frac{\# \ \text{True Positive}}{\#\ \text{Predicted Positive}}
$$
- Recall: What percentage of all positive samples were recalled?

$$
Recall = \frac{\# \ \text{True Positive}}{\# \ \text{Class Positive}}
$$
- F-Measure: A combined metric which accounts for precision and recall in a single measure. 

$$
F-Measure = 2 * (\frac{\text{Precision} * \text{Recall}}{\text{Precision} + {Recall}})
$$

## ROC Curves:
- ROC(Receiver Operating Characteristic) shows the performance of a classifier at all classification thresholds

<div align=center>
    <img src ="img/roc.png" width="600" height ="150"/>  
</div>
- An ROC Curve plots True Positive Rate or TPR  vs. False Positive Rate or FPR. TPR=TP/P, FPR=FP/N  
<div align=center>
    <img src ="img/rocc.png" width="300" height ="280"/>  
</div>

- If Area Under Receiver Operating
Characteristic Curve (AUROC) is
1, classifier is perfect
- If AUROC is 0.5, classifier is no
better than random chance.
- If AUROC is 0, classifier is
worst possible (TPR=0, FPR=1)

## PRC
- PRC=Precision-Recall Curve. Plot Precision on y-axis, Recall on x-axis for all possible thresholds

<div align=center>
    <img src ="img/prc.png" width="300" height ="280"/>  
</div>

- If Area Under PRC (AUPRC) is 1,
classifier is perfect

## Multiclass Classifiers
Define $K$ classes numbered $1,\cdots,K$. Also define random variables $Y_1,\cdots,Y_K$ where $y_k=1$
for a class k observation (realization is $y$) $\rightarrow$ we call this one-hot encoding (ohe).

<div align=center>
    <img src ="img/ohe.png" width="600" height ="180"/>  
</div>


## Code:

Check if the class is balanced.
```python
# Is it class-balanced or unbalanced?
df.CLASS.value_counts()
```

Calculate the number of tp, tn, fp, fn, Accuracy, Precision, Recall,  Sensitivity and Specificity from model output.
```python
# Calculate tp, tn, fp, fn, and test accuracy
def compute_performance(yhat, y, classes):
    tp = sum(np.logical_and(yhat == classes[1], y == classes[1]))
    tn = sum(np.logical_and(yhat == classes[0], y == classes[0]))
    fp = sum(np.logical_and(yhat == classes[1], y == classes[0]))
    fn = sum(np.logical_and(yhat == classes[0], y == classes[1]))

    print(f"tp: {tp} tn: {tn} fp: {fp} fn: {fn}")
    
    # Precision
    # "Of the ones I labeled +, how many are actually +?"
    precision = tp / (tp + fp)
    
    # Recall
    # "Of all the + in the data, how many do I correctly label?"
    recall = tp / (tp + fn)    
    
    # Sensitivity
    sensitivity = recall
    
    # Specificity
    # "Of all the - in the data, how many do I correctly label?"
    specificity = tn / (fp + tn)

    print("Accuracy:",round(acc,3),"Recall:",round(recall,3),"Precision:",round(precision,3),
          "Sensitivity:",round(sensitivity,3),"Specificity:",round(specificity,3))

compute_performance(y_pred, ytest, df.classes_)
```

Draw the ROC curve.
```python
# Predict with sklearn. Note: probabilities of class 0 (first col), class 1 (2nd col)
ytest_prob = pd.predict_proba(Xtest)

'''
Example output:
array([[3.41493136e-02, 9.65850686e-01],
       [9.99968922e-01, 3.10777332e-05],
       [5.24721411e-01, 4.75278589e-01],
       ...,
       [9.82047805e-01, 1.79521950e-02],
       [1.52567844e-01, 8.47432156e-01],
       [9.99813357e-01, 1.86643411e-04]])
'''

# Adjusting the decision threshold
yhat = pd.classes_[(ytest_prob[:,1]>0.1).astype(int)]

# ROC using sklearns ROC curve.
fpr, tpr, _ = roc_curve(ytest, ytest_prob[:,1], pos_label="Pos")
ax=sns.lineplot(fpr,tpr)

# Determine AUC for each of the ROC curves
AUC= auc(fpr,tpr)

```