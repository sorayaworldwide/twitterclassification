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
##### NUGGET 2 - adding more features for prediction - NGRAMS
##textfeatures
install.packages("textfeatures")
library(textfeatures)
##number of words function
n_wordtweets<-function(text){sapply(strsplit(text, ""), length)}
n_wordtweets(tweetsvaderfinal_train$text)
tweetsvader_recv2<-
  recipe(sentiment ~ text, data = tweetsvaderfinal_train)
tweetsvader_recv2 <-tweetsvader_recv2%>%
  step_textfeature(text, role ="predictor", extract_functions = textfeatures::count_functions)
##update model 
tweetsvader_recv2 <- tweetsvader_recv2%>%
  recipe(sentiment ~ text, data=tweetsvader_recv2)
  step_tokenize(text) %>%
  step_tokenfilter(text, 
                   max_tokens = tune()) %>%
  step_tfidf(text) 
##update specify model
multi_lasso_wfv2<-workflow()%>%
  add_recipe(tweetsvader_recv2, blueprint = sparse_bp)%>%
  add_model(multi_spec)
##multi-lasso update
multi_lasso_rsv2<-tune_grid(
  multi_lasso_wfv2,
  tweetsvader_folds, 
  grid = 10,
  control = control_resamples(save_pred = TRUE)
)
## abandon above: vary n-gram model
ngram_rec <- function(ngram_options) {
  recipe(sentiment ~ text, data = tweetsvaderfinal_train) %>%
    step_tokenize(text, token = "ngrams", options = ngram_options) %>%
    step_tokenfilter(text, max_tokens = 1e3) %>%
    step_tfidf(text) %>%
    step_normalize(all_predictors())
}
##
ngram_rec(list(n = 3, n_min = 1))
##multinomial workflow blank
multi_lasso_wfv3<-workflow()%>%
  add_model(multi_spec)

## helper function to fit ngram
fit_ngram <- function(ngram_options) {
  fit_resamples(
    multi_lasso_wfv3 %>% add_recipe(ngram_rec(ngram_options)),
    tweetsvader_folds2
  )
}
##try different ngrams
set.seed(123)
unigram_rs <-fit_ngram(list(n=1))

set.seed(234)
bigram_rs <-fit_ngram(list(n=2, n_min = 1))

set.seed(345)
trigram_rs <-fit_ngram(list(n=3, n_min=1))

multi_lasso_wf
##redo folds?
set.seed(123)
tweetsvader_folds2 <- vfold_cv(tweetsvaderfinal_train)
collect_metrics(bigram_rs)
##abandon above try this way v3 is bigram
tweetsvader_recv3 <-
  recipe(sentiment ~ text,
         data = tweetsvaderfinal_train) %>%
  step_tokenize(text, token = "ngrams", options = list(n=2)) %>%
  step_tokenfilter(text, max_tokens = 1e3) %>%
  step_tfidf(text) 
##redo lasso wf
multi_lasso_wfv3 <- workflow() %>%
  add_recipe(tweetsvader_recv3, blueprint = sparse_bp) %>%
  add_model(multi_spec)
## redo tune grid
multi_lasso_rsv3 <- tune_grid(
  multi_lasso_wfv3,
  tweetsvader_folds,
  grid = 10,
  control = control_resamples(save_pred = TRUE)
)
## accuracy
best_accv3 <- multi_lasso_rsv3 %>%
  show_best("accuracy")
best_accv3
## try trigrams
tweetsvader_recv4 <-
  recipe(sentiment ~ text,
         data = tweetsvaderfinal_train) %>%
  step_tokenize(text, token = "ngrams", options = list(n=3)) %>%
  step_tokenfilter(text, max_tokens = 1e3) %>%
  step_tfidf(text) 
##redo lasso wf
multi_lasso_wfv4 <- workflow() %>%
  add_recipe(tweetsvader_recv4, blueprint = sparse_bp) %>%
  add_model(multi_spec)
## redo tune grid
multi_lasso_rsv4 <- tune_grid(
  multi_lasso_wfv4,
  tweetsvader_folds,
  grid = 10,
  control = control_resamples(save_pred = TRUE)
)
## accuracy
best_accv4 <- multi_lasso_rsv4 %>%
  show_best("accuracy")
best_accv4
##try step tokenize
tweetsvader_recv5 <-
  recipe(sentiment ~ text,
         data = tweetsvaderfinal_train) %>%
  step_tokenize(text) %>%
  step_stem (text)%>%
  step_ngram(text, num_tokens = 3, n_min = 1)%>%
  step_tokenfilter(text, max_tokens = 1e3) %>%
  step_tfidf(text) 

multi_lasso_wfv5 <- workflow() %>%
  add_recipe(tweetsvader_recv5, blueprint = sparse_bp) %>%
  add_model(multi_spec)

multi_lasso_rsv5 <- tune_grid(
  multi_lasso_wfv5,
  tweetsvader_folds,
  grid = 10,
  control = control_resamples(save_pred = TRUE)
)
best_accv5 <- multi_lasso_rsv5 %>%
  show_best("accuracy")
best_accv5
#confusion matrix
multi_lasso_rsv3 %>%
  collect_predictions() %>%
  filter(penalty == best_accv3$penalty) %>%
  filter(id == "Fold01") %>%
  conf_mat(sentiment, .pred_class) %>%
  autoplot(type = "heatmap") +
  scale_y_discrete(labels = function(x) str_wrap(x, 20)) +
  scale_x_discrete(labels = function(x) str_wrap(x, 20))

###nugget #3 downsample
library(themis)
tweetsvader_recv6 <-
  recipe(sentiment ~ text,
         data = tweetsvaderfinal_train) %>%
  step_tokenize(text) %>%
  step_tokenfilter(text, max_tokens = 1e3) %>%
  step_tfidf(text)
  step_downsample(sentiment)
##check accuracy
  tweetsvader_prep<-prep(tweetsvader_recv6)
juice(tweetsvader_prep)  
##update workflow
multi_lasso_wfv6<-workflow()%>%
  add_recipe(tweetsvader_recv6, blueprint = sparse_bp)%>%
  add_model(multi_spec)
## tune
multi_lasso_rsv6 <-tune_grid(
  multi_lasso_wfv6,
  tweetsvader_folds,
  grid=10,
  control = control_resamples(save_pred=TRUE)
)
multi_lasso_rsv6
## best acc
best_accv6<- multi_lasso_rsv6%>%
  show_best("accuracy")
best_accv6
best_acc
##confusion matrix
multi_lasso_rsv6 %>%
  collect_predictions() %>%
  filter(penalty == best_accv6$penalty) %>%
  filter(id == "Fold01") %>%
  conf_mat(sentiment, .pred_class) %>%
  autoplot(type = "heatmap") +
  scale_y_discrete(labels = function(x) str_wrap(x, 20)) +
  scale_x_discrete(labels = function(x) str_wrap(x, 20))
##redo ngram to include 1 and 2: recv7 bigrams and unigrams 
tweetsvader_recv7 <-
  recipe(sentiment ~ text,
         data = tweetsvaderfinal_train) %>%
  step_tokenize(text) %>%
  step_stem (text)%>%
  step_ngram(text, num_tokens = 2L, min_num_tokens = 1L)%>%
  step_tokenfilter(text, max_tokens = 1e3) %>%
  step_tfidf(text) 
##
multi_lasso_wfv7<-workflow()%>%
  add_recipe(tweetsvader_recv7, blueprint = sparse_bp)%>%
  add_model(multi_spec)
## tune
multi_lasso_rsv7 <-tune_grid(
  multi_lasso_wfv7,
  tweetsvader_folds,
  grid=10,
  control = control_resamples(save_pred=TRUE)
)
multi_lasso_rsv7
## best acc
best_accv7<- multi_lasso_rsv7%>%
  show_best("accuracy")
best_accv7
##confusion matrix
multi_lasso_rsv7 %>%
  collect_predictions() %>%
  filter(penalty == best_accv7$penalty) %>%
  filter(id == "Fold01") %>%
  conf_mat(sentiment, .pred_class) %>%
  autoplot(type = "heatmap") +
  scale_y_discrete(labels = function(x) str_wrap(x, 20)) +
  scale_x_discrete(labels = function(x) str_wrap(x, 20))

###textfeature number of words

tweetsvader_recv8<-
  recipe(sentiment ~ text, data = tweetsvaderfinal_train)

tweetsvader_recv8 <- tweetsvader_recv8 %>%
  step_mutate(text_copy = text)%>%
  step_textfeature(text_copy, extract_functions = nowords)



tweetsvader_recv8 <- tweetsvader_recv8%>%  
    step_tokenize(text) %>%
    step_stem (text)%>%
    step_tokenfilter(text, max_tokens = 1e3) %>%
    step_tfidf(text) 
  
##customfunction
install.packages("syn")
library(syn)
nowords <-function(text){n_words(text)
}
nowords(tweetsvaderfinal_train$text)

##
multi_lasso_wfv8<-multi_lasso_wf%>%
  update_recipe(tweetsvader_recv8, blueprint = sparse_bp)
multi_lasso_wfv8
## tune
multi_lasso_rsv8 <-tune_grid(
  multi_lasso_wfv8,
  tweetsvader_folds,
  grid=10,
  control = control_resamples(save_pred=TRUE)
)
multi_lasso_rsv8
## best acc
best_accv8<- multi_lasso_rsv8%>%
  show_best("accuracy")
best_accv8


##VIP
library(vip)
##fit bigram/unigram 
multi_lasso_fitv7 <- multi_lasso_wfv7 %>%
  fit(data = tweetsvaderfinal_train)

multi_lasso_fitv7

multi_lasso_fitv7 %>%
  pull_workflow_fit() %>%
  vip(num_features = 50, geom = "point")

##fit original
multi_lasso_fit <- multi_lasso_wf %>%
  fit(data = tweetsvaderfinal_train)

multi_lasso_fit

multi_lasso_fit %>%
  pull_workflow_fit() %>%
  vip(num_features = 50, geom = "point")