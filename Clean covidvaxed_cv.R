library(tidyverse)
install.packages("tidymodels")
library(tm)
library(tidymodels)
tweetsvader<-read.csv("covidvaxed_cv2.csv")
library(tidytext)
library(ggplot2)
##text variable 
tweetsvader_text <- tweetsvader$text  
##all text to lower
tweetsvader_text<- tolower(tweetsvader_text)
## replace blank space
tweetsvader_text <- gsub("rt", "", tweetsvader_text)
## replace @username
tweetsvader_text <- gsub("@\\w+", "", tweetsvader_text)
## remove punctuation
tweetsvader_text <- gsub("[[:punct:]]", "", tweetsvader_text)
##remove links
tweetsvader_text <- gsub("http\\w+", "", tweetsvader_text)
##remove tabs
tweetsvader_text <- gsub("[ |\t]{2,}", "", tweetsvader_text)
##remove blank spaces at beginning
tweetsvader_text <- gsub("^ ", "", tweetsvader_text)
## remove blank spaces at end
tweetsvader_text <- gsub(" $", "", tweetsvader_text)
## combine clean text column
tweetsvader_text2<-cbind(tweetsvader, tweetsvader_text)
## drop and rename
tweetsvader_text3<-select(tweetsvader_text2, -text)
##rename
tweetsvader_final<-rename(tweetsvader_text3, text=tweetsvader_text)
##un-nest tokens
tweets_token<-tweetsvader_final%>%
  unnest_tokens(word, text, token="words")
##remove stop words
tweets_tokenstop <- tweets_token %>%
  anti_join(stop_words, by = "word")
##take out amp
tweets_tokenstop<-tweets_tokenstop %>%
  group_by(word)%>%
  filter(word !="amp")
##word cloud
tweets_tokenstop%>%
  count(sentiment, word, sort=TRUE)
##
tweets_tokenstop%>%
  count(word, sort=TRUE)
##Create split for training data using tidymodels
set.seed(1234)

tweetsvaderfinal_split <- initial_split(tweetsvader_final, strata = sentiment)

tweetsvaderfinal_train <- training(tweetsvaderfinal_split)
tweetsvaderfinal_test <- testing(tweetsvaderfinal_split)

## number of tweets in each class

tweetsvaderfinal_train %>%
  count(sentiment, sort = TRUE) %>%
  select(n, sentiment)
## not imbalanced! 557, positive, 376 neutral, 228 negative yay!

## create new 'recipe' or text pre-processing for training
library(recipes)
install.packages("textrecipes")
library(textrecipes)
library(modeldata)
tweetsvader_rec <-
  recipe(sentiment ~ text,
         data = tweetsvaderfinal_train) %>%
  step_tokenize(text) %>%
  step_tokenfilter(text, max_tokens = 1e3) %>%
  step_tfidf(text) 
## cross-validation object
tweetsvader_folds <- vfold_cv(tweetsvaderfinal_train)
## multinomial regression object
multi_spec <- multinom_reg(penalty = tune(), mixture = 1) %>%
  set_mode("classification") %>%
  set_engine("glmnet")
##sparse bp
library(hardhat)
sparse_bp <- default_recipe_blueprint(composition = "dgCMatrix")
## create workflow 
multi_lasso_wf <- workflow() %>%
  add_recipe(tweetsvader_rec, blueprint = sparse_bp) %>%
  add_model(multi_spec)
multi_lasso_wf
## tune results
multi_lasso_rs <- tune_grid(
  multi_lasso_wf,
  tweetsvader_folds,
  grid = 10,
  control = control_resamples(save_pred = TRUE)
)
multi_lasso_rs
#accuracy
best_acc <- multi_lasso_rs %>%
  show_best("accuracy")
best_acc
# confusion matrix
library(stringr)
multi_lasso_rs %>%
  collect_predictions() %>%
  filter(penalty == best_acc$penalty) %>%
  filter(id == "Fold01") %>%
  conf_mat(sentiment, .pred_class) %>%
  autoplot(type = "heatmap") +
  scale_y_discrete(labels = function(x) str_wrap(x, 20)) +
  scale_x_discrete(labels = function(x) str_wrap(x, 20))
# 
