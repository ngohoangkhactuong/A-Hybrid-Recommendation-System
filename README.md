# A Hybrid Recommnendation System

*This project is to build a movie recommendation system using hybrid integration on the Odoo platform involves creating a smart system that suggests movies based on user preferences and history. It combines machine learning and natural language processing to analyze data and provide personalized recommendations, enhancing user satisfaction and marketing potential.*

## 1. Data

The [MovieLens](https://en.wikipedia.org/wiki/MovieLens) dataset is very popular in the recommender community. It was created in 1997 by [GroupLens Research](https://grouplens.org/), and the specific dataset used for this project has about 100,000 ratings for 9,000 movies by 600 users. For more information, please click the links below:

> * [MovieLens Website](https://movielens.org/)

> * [MovieLens Official Datasets](https://grouplens.org/datasets/movielens/)

> * [Dataset used in this Project](https://github.com/villafue/Capstone_2_MovieLens/tree/main/Data)

## 2. Evaluation Metrics

There are two quantitative metrics used to evaluate each recommender system.

```
Legend:

RMSE:      Root Mean Squared Error. Lower values mean better accuracy.
MAE:       Mean Absolute Error. Lower values mean better accuracy.

```
## 3. Models

These are the types of recommenders used in this project:

1. **Content-based Recommenders:** Content-based systems recommends items based on the attributes of those items themselves, instead of trying to use aggregate user behavior data.

2. **Collaborative-based Recommender:** Collaborative Based Recommenders leverages the behavior or others to inform what a user might enjoy. At a very high level, it means finding other people like him/her and recommending stuff they liked. Or it might mean finding other things similar to the things that he/she likes. Either way, the idea is taking cues from people similar to a specified user and recommending stuff based on the things they like that this user has not seen yet. It's recommending stuff based on other people's collaborative behavior.

3. **Matrix Factorization Methods:** Instead of trying to find items or users that are similar to each other, data science and machine learning techniques are applied to extract predictions from the ratings data. The approach is to train models with user-ratings data, and use those models to predict the ratings of new movies by the users.

4. **Hybrid Recommenders:** In the real world, there’s no need to choose a single algorithm for your recommender system. Each algorithm has its own strengths and weaknesses and combining many algorithms together could make the sum better than its parts.
