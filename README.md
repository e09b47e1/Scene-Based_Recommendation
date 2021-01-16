# Datasets and Codes of Scene-Based Recommendation Task (for reviewers) #

These are anonymous open source datasets and codes for scene-based recommendation task.
You can find a more concrete description about the datasets in Section Experiment of our submitted paper.
In order to comply with the submission policy of the conference, there is no description of content except file format here.
The datasets and codes will be released formally in camera ready submission. 

## Datasets Introduction  ##
We construct two different datasets based on two major category commodity sets with different features:

- Baby&Toy: commodities in this dataset are mainly for the pregnant women, new mothers, toddlers and infants.
- Electronics: these commodities include digital products, computer peripherals, etc.
- Fashion: all kinds of clothes, such as blouse, shirt, shorts, trousers, skirt, dress, hat, etc.
- Food&Drink: the commodities about food and drink, e.g., green tea, wine, coffee, beef, cake, bread, fruit, etc. 


## Files in Each Dataset ##
The file structure in each dataset is the same. We will introduce these files one by one:

- user\_item\_pair.txt: the interactions between users and commodities. There are three elements in each row: a user, a commodity, and a boolean.
- scene-cate\_set.txt: the scenes which consist of categories. Each row is a scene and there are several categories in it. The data file cate_scene appearing in codes can be generated from data file scene-cate\_set. There are two elements in each row: a category and a scene it belongs to.
- cate\_cate\_pair.txt: the relevance between categories. There are three elements in each row: category 1, category 2, and their relevance score.
- item\_cate\_pair.txt: the relationship between commodities and categories. There are three elements in each row: a commodity, a category, and boolean which indicates whether the commodity belongs to the category.
- item\_item\_pair.txt: the similarity between commodities. There are three elements in each row: commodity 1, commodity 2, and normalized similarity scores.

Moreover, there are eleven other data files for the training process which stem from data file user\_item\_pair. 
By leave-one-out strategy, we randomly hold out one positive item and sample 100 negative items to build the validation set and randomly choose another positive item along with 100 negative samples for test set. 
The rows whose line numbers are the same in train_user, valid\_posItem, and valid\_negItems correspond to the same user. 
There is one user, one item and 100 items in each row in the above file, respectively. The other 3 files have similar structure, i.e., test\_user, test\_posItem, and test\_negItems.
The remaining positive samples and their interacted users consist of the training set.
Each interacted user-item pair is stored in the row of file train\_user\_item and file train\_item\_user. For the pair-wise objective, we sample a negative item for each pair and put them into three files: train\_user, train\_posItem, and train\_negItem.
