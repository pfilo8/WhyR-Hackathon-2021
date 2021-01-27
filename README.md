# Górnicy Carla Friedricha Team
> WhyR-Hackathon-2021 submission repository

## Table of contents
* [General info](#general-info)
* [Metrics](#metrics)
* [Best solution](#best-solution)
* [Other approaches](#another-approaches)
* [Contact](#contact)

## General info
Details about the hackathon - something about data etc
Maybe it is the right place for the cv and acc/f1 explanation as well.

## Metrics
I've tried to add here the html file with acc and f1 but it seems to be too heavy.
![Metrics screenshot](./outputs/metrics.png)

## Our solution
Here should be the description of the model we chose.
#### Preprocessing steps

#### Model training


## Other approaches

### Logistic regression with hand-crafted features
We used eleven hand-crafted features in this approach. The first eight of them were designed to measure the distance between the two strings.
We decided to use the following string metrics:
 * **Jaccard distance** is defined to be the number of all common terms divided by the sum of all terms,
 * **Damerau-Levenshtein distance**  is given by the normalized number of primary operations that have to be applied to a pair for strings to transform one of them into another,
 * **Jaro-Winkler distance**  is based on the difference of character histograms of two strings and the number of transpositions required to match those histograms. 
 Furthermore, Jaro-Winkler distance attaches higher importance to the prefixes.
 * **QGram distance**  is based on counting the number of the occurrences of different q-grams in the two strings
 
Those metrics were used to compare the titles and authors of both articles. 

The next two features were created by the comparison of the year of creation. The first of them encode checked to see if they were identical, whereas the second contained the opposite information. 
They seem to be redundant. However, they could be both false when at least one of the articles did not have an assigned year.

The last variable checked whether both articles came from the same venue.

Careful preparation of the data appeared to be crucial for high scores. We closely analyze the title to extract and fill in the missing data.

At the last step, we trained **logistic regression**. We achieved **99.16%** of accuracy on 10-fold cross-validation. 
Logistic regression is not a sophisticated machine learning algorithm. Therefore, we tried more complex models as well.
The best accuracy we were able to get was about **99.32%**. We rejected them because of much higher train accuracy, which may have been caused by overfitting.


## Contact
Created by Łukasz Łaszczuk(???@gmail.com), Robert Benke(<robert.benke2@gmail.com>) and Patryk Wielopolski(???@gmail.com) - feel free to contact us!
