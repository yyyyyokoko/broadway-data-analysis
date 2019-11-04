# broadway-data-analysis

Group Members: 
	Lujia Deng (ld781)
	Luwei Lei (ll1038) 
	Janet Liu (yl879) 
	Yunfei Zhang (yz678)


This repository contains all the files and scripts pertaining to the broadway data analysis Project Part 2 for ANLY 501.

[Potential Extra Credits]
1. We used LOF to identify outliers in our data set. 
2. For clustering analysis, we used both the Silhouette and Calinski-Harabaz procedure to assess the quality of the clusters. 


[PDF File]

Project_Part2_Report.pdf contains our writepups required by the project.


[Input Data Files]

The following csv files are the input file for both exploratory and predictive analysis:

1. grosses_cleaned.csv
2. cleaned_SocialMedia.csv 				
3. part2cleanedGrosses.csv 			
4. part2cleanedSocialMedia.csv
5. Musical_ratings-withoutNa-cleaned.csv



[Scripts]

1. grosses_preprocessing.py 		Further data cleaning for the Grosses data set
2. Part2_SocialMediaCleaning.py 	Further data cleaning for the Social Media data set
3. hypothesisTesting.py 			Script for Hypothesis Testing 
4. clustering-association.py 		Script for Clustering Analysis & Association Rule
5. classification-task1.py 			Script for Classification Task 1 using DT, KNN, RF, SVM
6. classification-task2.py 			Script for Classification Task 2 using Naive Bayes
7. HistAndCorrelation.py 			Script for Histogram and Correlation


[Output Files]

The following output files are produced from the scripts HistAndCorrelation.py and hypothesisTesting.py:
1. ANOVA.png
2. CorrelationPlot.png
3. grosses_over_time_with_outliers.png
4. grosses_over_time.png
5. price_over_time.png
6. hist.png

The following output files are produced from the script clustering-association.py:
1. Itemset_confidence0.csv
2. Itemset_confidence1.csv
3. Itemset_support0.csv
4. Itemset_support1.csv
5. clustering-quality.txt
6. DBSCAN.png
7. K-Means.png
8. Ward.png

The following output files are produced from the script classification-task1.py:
1. feature importance.png
2. roc.png
