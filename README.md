# twitter sentiment analysis

Twitter is an invaluable source of insights on trends and opinions in our society. During public events such as the Presidential Debates, analyses of the Twitter live stream allow us to collect large, contextual data sets in a short amount of time. 

This Python script allows accessing the Twitter stream and filtering out tweets containing specified filter words.

For each word, the script counts the occurences of words in tweets they were mentioned in. The most frequently used words for each of the filter words are compared with the Affective Norm of English Words (ANEW) dictionary containing 1030 rated words. The graphical represenation of these scores and their standard deviations yields psychological valence (pleasure) and arousal (excitement) ratings for the filter words on Twitter. 

The Jupyter notebook visualizes the data collected from the Twitter stream. Below is a visualization for the March 6 Presidential Debate of the Democratic Party in Flint, Michigan, featuring Hillary Clinton and Bernie Sanders. The filter words for this sentiment analysis were 'Clinton' and 'Sanders'. The plots show differential prevalence and arousal maps between the two candidates. Bernie Sanders seems to polarize his audience more than Hillary Clinton.

![My image](https://github.com/ahwkuepper/twitter_sentiment_analysis/blob/master/plots/difference_maps_democrats.png)
