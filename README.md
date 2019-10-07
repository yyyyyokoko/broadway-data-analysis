# broadway-data-analysis

Group Members: 
	Lujia Deng (ld781)
	Luwei Lei (ll1038) 
	Janet Liu (yl879) 
	Yunfei Zhang (yz678)


This repository contains all the files and scripts pertaining to the broadway data analysis project for ANLY 501.

[Input Data Files]

1. broadway-shows-all.cvs;
	The input file for the following scirpts:
	- get_clean_broadway_wiki.py
	- text_data_scrapping.py 

2. Folder: data
	The files in this folder are input files for the following script:
	- grosses_scrapping.py




[Scripts]

1. get_clean_broadway_wiki.py
This script scraped Wikipedia pages for Broadway's shows; 
read input file broadway-shows-all.cvs;


2. grosses_scrapping.py 					
This script scraped grosses info of Broadway's shows from 1985-2019;

3. grosses_cleanliness_measure.py 			
This script looks at and quantifies the cleanliness of the attributes 
in the Broadway Grosses data set;

4. grosses_cleaning_script.py 				
This script cleans poor-quality attributes in the Broadway Grosses data set;


5. SocialMedia_scrap.py 					
This script scraped social media stats of Broadway's shows from 1985-2019;

6. SocialMedia_cleanliness.py 				
This script looks at and quantifies the cleanliness of the attributes 
in the Broadway Social Media Stats data set;

7. SocialMedia_data_cleaning.py 			
This script cleans poor-quality attributes in the Broadway Social Media 
Stats data set;


8. rating_cleanliness_and_cleaning.py 		
This script quantifies the cleanliness of the attributes and cleans 
poor-quality attributes in the Broadway Review Ratings data set;


9. text_data_scrapping.py 					
This script scraped Broadway Reviews and News data, which are textual;
read input file broadway-shows-all.cvs; 




[Data]
This folder contains all the data set files collected by our scripts, both before and after cleaning. 

1. Broadway_Wikipedia						
This folder contains xml files for Broadway shows from 1985-2019;

2. broadway_grosses.csv 					
This csv file contains Broadway Grosses data before cleaning;

3. grosses_cleaned.csv 						
This csv file contains Broadway Grosses data after cleaning;

4. scrap_of_social_media.csv 				
This csv file contains Broadway Social Media Stats data before cleaning;

5. cleaned_SocialMedia.csv 					
This csv file contains Broadway Social Media Stats data after cleaning;

6. Musical_ratings-withNa.csv 				
This csv file contains Broadway Review Ratings data before cleaning;

7. Musical_ratings-withoutNa.csv 			
This csv file contains Broadway Review Ratings data after cleaning;

8. clean_review.csv  						
This csv file contains Broadway Review text data after cleaning;

9. news.csv 								
This csv file contains Broadway News text data after cleaning;