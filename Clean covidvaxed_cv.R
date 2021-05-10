library(tidyverse)
library(tm)
library(tidymodels)
library(tidytext)
library(ggplot2)
library(recipes)
library(wordcloud2)
library(textrecipes)
library(modeldata)
library(hardhat)
library(stringr)
library(themis)
library(textfeatures)
library(syn)
library(vip)
library(reticulate)
library(quanteda)
library(paletteer)

 
covid_vaccine_education<-search_tweets(q="vaccine covid school", n=5000, lang="en", include_rts=FALSE)

##read data
tweetsvader<-read.csv("covidvaxed_cv2.csv")
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
wordclouddf <-tweets_tokenstop%>%
  count(word, sort=TRUE)

wordcloud2(wordclouddf)

##Create split for training data using tidymodels
set.seed(1234)

tweetsvaderfinal_split <- initial_split(tweetsvader_final, strata = sentiment)

tweetsvaderfinal_train <- training(tweetsvaderfinal_split)
tweetsvaderfinal_test <- testing(tweetsvaderfinal_split)

## number of tweets in each class

tweetsvaderfinal_train %>%
  count(sentiment, sort = TRUE) %>%
  select(n, sentiment)
## imbalanced with positives 557, positive, 376 neutral, 228 negative

## create new 'recipe' or text pre-processing for training

tweetsvader_rec <-
  recipe(sentiment ~ text,
         data = tweetsvaderfinal_train) %>%
  step_tokenize(text) %>%
  step_tokenfilter(text, max_tokens = 1e3) %>%
  step_tfidf(text) 

tweetsvader_rec

## cross-validation object
tweetsvader_folds <- vfold_cv(tweetsvaderfinal_train)

## multinomial regression object
multi_spec <- multinom_reg(penalty = tune(), mixture = 1) %>%
  set_mode("classification") %>%
  set_engine("glmnet")

multi_spec

##sparse bp
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
##roc_auc
multi_lasso_rs %>%
  show_best()

best_acc
# confusion matrix

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


## Define tuning process
model_metrics <-metric_set(accuracy, sens, spec, mn_log_loss, roc_auc)

###Combine unigrams and bigrams with textfeatures

vadertext_recv10 <-recipe (sentiment ~ text, data = tweetsvaderfinal_train)%>%
  step_mutate(text_copy = text)%>%
  step_textfeature(text_copy)
  
 vadertext_recv10 <-vadertext_recv10%>% 
  step_ngram(text, num_tokens = 2L, min_num_tokens = 1L)%>%
  step_tokenize(text) %>%
  step_tokenfilter(text, max_tokens = 1e3) %>%
  step_tfidf(text) 

glimpse(vadertext_recv10)
vadertext_prep10 <-prep(vadertext_recv10)
bake(vadertext_prep10, new_data=NULL)


multi_lasso_wfv10<-workflow()%>%
  add_recipe(vadertext_recv10, blueprint = sparse_bp)%>%
  add_model(multi_spec)
## tune
multi_lasso_rsv10 <-tune_grid(
  multi_lasso_wfv10,
  tweetsvader_folds,
  grid=10,
  control = control_resamples(save_pred=TRUE)
)

multi_lasso_rsv10

features <-textfeatures(tweetsvader_final$text, normalize=FALSE)
tweetsvader_cbind <-tweetsvader_final%>%
  cbind(features$n_words,
        features$n_charsperword,
        features$n_chars)
features
tweetsvader_cbind
tweetsvader_cbind<-tweetsvader_cbind%>%
  rename(nwords = "features$n_words", charspword = "features$n_charsperword", nchars = "features$n_chars")
##
tweetsvader_cbind %>%
  ggplot(aes(charspword, nchars, colour = sentiment)) + geom_point()
##
tweetsvader_cbind %>% 
  ggplot(aes(charspword, nwords)) + geom_point() +
  facet_wrap(~sentiment)+
  stat_smooth()
## 
tweetsvader_cbind %>%
  ggplot(aes(charspword, nwords, colour = sentiment)) + geom_point()


vadertext_rec <-recipe (sentiment ~ text, data = tweetsvaderfinal_train)%>%
  step_textfeature(text)
vader_obj<-vadertext_rec %>%
  prep()
bake(vader_obj, new_data = NULL)%>%
  slice(1:3)
bake(vader_obj, new_data = NULL)%>%
  pull(textfeature_text_n_words)

## textfeatures 

vadertext_recv9 <-recipe (sentiment ~ text, data = tweetsvaderfinal_train)%>%
  step_mutate(text_copy = text)%>%
  step_textfeature(text_copy)%>%
  step_tokenize(text) %>%
  step_tokenfilter(text, max_tokens = 1e3) %>%
  step_tfidf(text) 

##Update workflow
multi_lasso_wfv9<-multi_lasso_wf%>%
  update_recipe(vadertext_recv9, blueprint = sparse_bp)

multi_lasso_wfv9

## Update grid
multi_lasso_rsv9 <-tune_grid(
  multi_lasso_wfv9,
  tweetsvader_folds,
  grid=10,
  control = control_resamples(save_pred=TRUE)
)
multi_lasso_rsv9

##
best_accv9<- multi_lasso_rsv9%>%
  show_best("accuracy")
best_accv9
##redo metrics

set.seed(2020)
multi_lasso_rsv91 <- tune_grid(
  multi_lasso_wfv9,
  tweetsvader_folds,
  grid=10,
  metrics = metric_set(accuracy, sensitivity, specificity, kap)
)

multi_lasso_rsv91

autoplot(multi_lasso_rsv91)
autoplot(multi_lasso_rs)

multi_lasso_rs %>%
  select_by_pct_loss(metric = "accuracy", -penalty)

##confusion Matrix for textfeatures 
multi_lasso_rsv9 %>%
  collect_predictions() %>%
  filter(penalty == best_accv9$penalty) %>%
  filter(id == "Fold01") %>%
  conf_mat(sentiment, .pred_class) %>%
  autoplot(type = "heatmap") +
  scale_y_discrete(labels = function(x) str_wrap(x, 20)) +
  scale_x_discrete(labels = function(x) str_wrap(x, 20))

## Define best acc so far
best_acc #.756
best_accv13 #.742
best_accv9 #.73
best_accv6  #.72


##  receipe 10 lda 
vadertext_recv10 <-recipe (sentiment ~ text, data = tweetsvaderfinal_train)%>%
  step_tokenize(text) %>%
  step_lda(text, num_topics = 3)

#Update workflow
multi_lasso_wfv10<-multi_lasso_wf%>%
  update_recipe(vadertext_recv10, blueprint = sparse_bp)

multi_lasso_wfv10

# Update grid
multi_lasso_rsv10 <-tune_grid(
  multi_lasso_wfv10,
  tweetsvader_folds,
  grid=10,
  control = control_resamples(save_pred=TRUE)
)
multi_lasso_rsv10

#Best accuracy is .48 
best_accv10<- multi_lasso_rsv10%>%
  show_best("accuracy")
best_accv10


## Part of speech filtering
tweetsvader_recv11 <-
  recipe(sentiment ~ text,
         data = tweetsvaderfinal_train) %>%
  step_tokenize(text, engine = "spacyr") %>%
  step_pos_filter(keep_tags = "ADJ", "ADP", "ADV", "AUX", "CONJ", 
                  "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", 
                  "PROPN", "PUNCT", "SCONJ", "SYM", "VERB")%>%
  step_tokenfilter(text, max_tokens = 1e3) %>%
  step_tfidf(text) 

#Update workflow
multi_lasso_wfv11<-multi_lasso_wf%>%
  update_recipe(tweetsvader_recv11, blueprint = sparse_bp)

multi_lasso_wfv11

# Update grid
install.packages("spacyr")
library(spacyr)
multi_lasso_rsv11 <-tune_grid(
  multi_lasso_wfv11,
  tweetsvader_folds,
  grid=10,
  control = control_resamples(save_pred=TRUE)
)
multi_lasso_rsv10

#####Spacy 
py_config()

library("spacyr")
spacy_install()
spacy_initialize(model = "en_core_web_sm")
spacy_parse(tweetsvader_final$text[1], entity = FALSE, lemma = FALSE)
reticulate::use_python("c:\users\hsc12\appdata\local\r-mini~1\envs\spacy_condaenv\lib\site-packages")

##Create POS Tag Function
spacy_pos <- function(x) {
  tokens <- spacy_parse(x, entity = FALSE, lemma = FALSE)
  token_list <- split(tokens$pos, tokens$doc_id)
  names(token_list) <- gsub("text", "", names(token_list))
  res <- unname(token_list[as.character(seq_along(x))])
  empty <- lengths(res) == 0
  res[empty] <- lapply(seq_len(sum(empty)), function(x) character(0))
  res
}
## POS Tokens
tweetsvaderfinal_postokens <- tweetsvader_final %>%
  unnest_tokens(text, text, token = spacy_pos, to_lower = FALSE)
install.packages("paletteer")

colors <-rep(paletteer_d("rcartocolor::Pastel"), length.out =18)

##POS Tokens plot 
tweetsvaderfinal_postokens %>%
  count(text) %>%
  ggplot(aes(n, reorder(text, n), fill = reorder(text, n))) +
  geom_col() +
  labs(x = NULL, y = NULL, title = "Part of Speech tags in Covid Tweets") +
  scale_fill_manual(values = colors) +
  guides(fill = "none") +
  theme_minimal() +
  theme(plot.title.position = "plot") 

##POS by sentiment plot 
tweetsvaderfinal_postokens %>%
  count(sentiment, text) %>%
  group_by(sentiment) %>%
  mutate(prop = n / sum(n)) %>%
  ungroup() %>%
  ggplot(aes(forcats::fct_rev(reorder(text, n)), prop, fill = sentiment)) +
  geom_col(position = "dodge") +
  scale_fill_paletteer_d("nord::aurora") +
  labs(x = NULL, y = NULL, fill = NULL,
       title = "Part of speech tags by sentiment") +
  theme_minimal() +
  theme(legend.position = "top", 
        plot.title.position = "plot") 

## try bigram, POS
library(themis)
tweetsvader_recv12 <-
  recipe(sentiment ~ text,
         data = tweetsvaderfinal_train) %>%
  step_mutate(text_copy = text)%>%
  step_tokenize(text)%>%
  step_tokenize(text_copy, custom_token = spacy_pos) %>%
  step_tokenfilter(text, max_tokens = 1e3) %>%
  step_tfidf(text, text_copy) 
##
multi_lasso_wfv12<-workflow()%>%
  add_recipe(tweetsvader_recv12, blueprint = sparse_bp)%>%
  add_model(multi_spec)
## tune
multi_lasso_rsv12 <-tune_grid(
  multi_lasso_wfv12,
  tweetsvader_folds,
  grid=10,
  control = control_resamples(save_pred=TRUE)
)
multi_lasso_rsv12
## best acc - .718 did worse on identifying the negative tweets
best_accv12<- multi_lasso_rsv12%>%
  show_best("accuracy")
best_accv12
##confusion matrix
multi_lasso_rsv12 %>%
  collect_predictions() %>%
  filter(penalty == best_accv12$penalty) %>%
  filter(id == "Fold01") %>%
  conf_mat(sentiment, .pred_class) %>%
  autoplot(type = "heatmap") +
  scale_y_discrete(labels = function(x) str_wrap(x, 20)) +
  scale_x_discrete(labels = function(x) str_wrap(x, 20))


##final try original with stem

tweetsvader_rec13 <-
  recipe(sentiment ~ text,
         data = tweetsvaderfinal_train) %>%
  step_tokenize(text) %>%
  step_stem(text)%>%
  step_tokenfilter(text, max_tokens = 1e3) %>%
  step_tfidf(text) 

##

multi_lasso_wfv13<-workflow()%>%
  add_recipe(tweetsvader_rec13, blueprint = sparse_bp)%>%
  add_model(multi_spec)
## tune
multi_lasso_rsv13 <-tune_grid(
  multi_lasso_wfv13,
  tweetsvader_folds,
  grid=10,
  control = control_resamples(save_pred=TRUE)
)
multi_lasso_rsv13
## best acc - .742
best_accv13<- multi_lasso_rsv13%>%
  show_best("accuracy")
best_accv13
##
multi_lasso_rsv13 %>%
  collect_predictions() %>%
  filter(penalty == best_accv13$penalty) %>%
  filter(id == "Fold01") %>%
  conf_mat(sentiment, .pred_class) %>%
  autoplot(type = "heatmap") +
  scale_y_discrete(labels = function(x) str_wrap(x, 20)) +
  scale_x_discrete(labels = function(x) str_wrap(x, 20))

## Define best acc so far
best_acc #.756 Original
best_accv13 #.742 Stemming
best_accv9 #.73 Textfeatures
best_accv6  #.72 downsample 


multi_lasso_rsv13%>%
  show_best()

multi_lasso_rsv9%>%
  show_best()

##smote
tweetsvader_recv14 <-
  recipe(sentiment ~ text,
         data = tweetsvaderfinal_train) %>%
  step_tokenize(text) %>%
  step_tokenfilter(text, max_tokens = 1e3) %>%
  step_tfidf(text)%>%
  step_smote(sentiment)

## create workflow 
multi_lasso_wfv14 <- workflow() %>%
  add_recipe(tweetsvader_recv14, blueprint = sparse_bp) %>%
  add_model(multi_spec)

## tune results
multi_lasso_rsv14 <- tune_grid(
  multi_lasso_wfv14,
  tweetsvader_folds,
  grid = 10,
  control = control_resamples(save_pred = TRUE)
)
multi_lasso_rs
#accuracy
best_accv14 <- multi_lasso_rsv14 %>%
  show_best("accuracy")
##roc_aucv14
multi_lasso_rsv14 %>%
  show_best()

##roc_auc original
multi_lasso_rs %>%
  show_best()

multi_lasso_rsv9 %>%
  show_best()

best_accv9

best_accv14


multi_lasso_rsv14 %>%
  collect_predictions() %>%
  filter(penalty == best_accv14$penalty) %>%
  filter(id == "Fold01") %>%
  conf_mat(sentiment, .pred_class) %>%
  autoplot(type = "heatmap") +
  scale_y_discrete(labels = function(x) str_wrap(x, 20)) +
  scale_x_discrete(labels = function(x) str_wrap(x, 20))


##choose textreceipe version v9 textfeatures
set.seed(2020)
tune_rs <-tune_grid(
  multi_lasso_wfv9,
  tweetsvader_folds,
  grid= 10,
  metrics = metric_set(accuracy, sensitivity, specificity, kap),
  control = control_resamples(save_pred = TRUE)
)
autoplot(tune_rs)


collect_predictions(tune_rs)
collect_metrics(tune_rs)%>%
  filter(.metric=="kap")

 

#choose best accuracy 
choose_acc <-tune_rs %>%
  select_by_pct_loss(metric="accuracy", -penalty)

tune_rs %>%
  select_by_one_std_err(metric="accuracy", -penalty)

#final workflow

final_wf<-finalize_workflow(multi_lasso_wfv9, choose_acc)

final_wf
##final fitted

final_fitted <-last_fit(final_wf, tweetsvaderfinal_split)

collect_metrics(final_fitted)

##confusion matrix

collect_predictions(final_fitted) %>%
  conf_mat(truth = sentiment, estimate = .pred_class) %>%
  autoplot(type = "heatmap")


## Define best acc so far
best_acc #.756 Original
best_accv13 #.742 Stemming
best_accv9 #.73 Textfeatures
best_accv6  #.72 downsample 


multi_lasso_rsv13%>%
  show_best()

multi_lasso_rsv9%>%
  show_best()

multi_lasso_rs%>%
  show_best()