'''
  This program shell reads phrase data for the kaggle phrase sentiment classification problem.
  The input to the program is the path to the kaggle directory "corpus" and a limit number.
  The program reads all of the kaggle phrases, and then picks a random selection of the limit number.
  It creates a "phrasedocs" variable with a list of phrases consisting of a pair
    with the list of tokenized words from the phrase and the label number from 1 to 4
  It prints a few example phrases.
  In comments, it is shown how to get word lists from the two sentiment lexicons:
      subjectivity and LIWC, if you want to use them in your features
  Your task is to generate features sets and train and test a classifier.

  Usage:  python classifyKaggle.py <corpus directory path> <limit number>

'''
# Opening python and nltk and sklearn packages needed for processing
import os
import sys
import random
import nltk
import re
import sentiment_read_LIWC_pos_neg_words
import cross_validation
from nltk.corpus import stopwords
from nltk import FreqDist
from nltk.collocations import *
from nltk.metrics import ConfusionMatrix
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.naive_bayes import  MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# Writing the function, writeFeatureSets to a CSV file
def writeFeatureSets(featuresets, outpath):
    # Opening outpath for writing
    f = open(outpath, 'w')
    # Obtaining the feature names from the feature dictionary in the first featureset
    featurenames = featuresets[0][0].keys()
    # Creating the first line of the file as comma separated feature names
    #    with the word class as the last feature name
    featurenameline = ''
    for featurename in featurenames:
        # Replacing forbidden characters with text abbreviations
        featurename = featurename.replace(',','CM')
        featurename = featurename.replace("'","DQ")
        featurename = featurename.replace('"','QU')
        featurenameline += featurename + ','
    featurenameline += 'class'
    # Writing this as the first line in the csv file
    f.write(featurenameline)
    f.write('\n')
    # Converting each feature set to a line in the file with comma separated feature values,
    # each feature value is converted to a string 
    #   For booleans, this is the words true and false
    #   For numbers, this is the string with the number
    for featureset in featuresets:
        featureline = ''
        for key in featurenames:
          try:
            featureline += str(featureset[0].get(key,[])) + ','
          except KeyError:
            continue
        if featureset[1] == 0:
          featureline += str("strongly negative")
        elif featureset[1] == 1:
          featureline += str("slightly negative")
        elif featureset[1] == 2:
          featureline += str("neutral")
        elif featureset[1] == 3:
          featureline += str("slightly positive")
        elif featureset[1] == 4:
          featureline += str("strongly positive")
        # Writing each feature set values to the file
        f.write(featureline)
        f.write('\n')
    f.close()

# Obtaining word features, bag of words
def bag_of_words(wordlist):
  wordlist = nltk.FreqDist(wordlist)
  word_features = [w for (w, c) in wordlist.most_common(200)] 
  return word_features    

# Defining features or keywords of a document for a BOW/unigram baseline
def unigram_features(doc, word_features):
  doc_words = set(doc)
  features = {}
  for word in word_features:
    features['contains(%s)'%word] = (word in doc_words)
  return features
  
# Obtaining bigram document features and adding the most frequent bigrams
def bag_of_words_bigram(wordlist, bigramcount):
  bigram_measures = nltk.collocations.BigramAssocMeasures()
  finder = BigramCollocationFinder.from_words(wordlist, window_size=3)
  finder.apply_freq_filter(3)
  bigramword_features = finder.nbest(bigram_measures.chi_sq, 3000)
  return bigramword_features[:bigramcount]
  
def bigram_features(doc, word_features, bigramword_features):
  document_words = set(doc)
  document_bigrams = nltk.bigrams(doc)
  features = {}
  for word in word_features:
    features['contains(%s)' % word] = (word in document_words)
  for bigram in bigramword_features:
    features['bigram(%s %s)' % bigram] = (bigram in document_bigrams)
  return features
  
# Features for negation words
def negative_features(doc, word_features, negationwords):
  features = {}
  for word in word_features:
    features['contains({})'.format(word)] = False
    features['contains(NOT{})'.format(word)] = False
  
  for i in range(0, len(doc)):
    word = doc[i]
    if ((i + 1) < len(doc)) and (word in negationwords):
      i += 1
      features['contains(NOT{})'.format(doc[i])] = (doc[i] in word_features)
    else:
      if ((i + 3) < len(doc)) and (word.endswith('n') and doc[i+1] == "'" and doc[i+2] == 't'):
        i += 3
        features['contains(NOT{})'.format(doc[i])] = (doc[i] in word_features)
      else:
        features['contains({})'.format(word)] = (word in word_features)
  return features

# POS tag features
def POS_features(doc, word_features):
    document_words = set(doc)
    tagged_words = nltk.pos_tag(doc)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    numNoun = 0
    numVerb = 0
    numAdj = 0
    numAdverb = 0
    for (word, tag) in tagged_words:
        if tag.startswith('N'): numNoun += 1
        if tag.startswith('V'): numVerb += 1
        if tag.startswith('J'): numAdj += 1
        if tag.startswith('R'): numAdverb += 1
    features['nouns'] = numNoun
    features['verbs'] = numVerb
    features['adjectives'] = numAdj
    features['adverbs'] = numAdverb
    return features

def POS2_features(doc, word_features):
    tagged_words = nltk.pos_tag(doc)
    document_words = set(doc)
    nwords = clean_text(document_words)
    nwords = rem_no_punct(nwords)
    nwords = rem_stopword(nwords)
    nwords = lemmatizer(nwords)
    nwords = stemmer(nwords)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in nwords)
    numNoun = 0
    numVerb = 0
    numAdj = 0
    numAdverb = 0
    for (word, tag) in tagged_words:
        if tag.startswith('N'): numNoun += 1
        if tag.startswith('V'): numVerb += 1
        if tag.startswith('J'): numAdj += 1
        if tag.startswith('R'): numAdverb += 1
    features['nouns'] = numNoun
    features['verbs'] = numVerb
    features['adjectives'] = numAdj
    features['adverbs'] = numAdverb
    return features
	
# Sentiment lexicon: Subjectivity count features
def SL_features(doc, word_features, SL):
  document_words = set(doc)
  features = {}
  for word in word_features:
    features['contains({})'.format(word)] = (word in document_words)
  # Counting variables for the 4 classes of subjectivity
  weakPos = 0
  strongPos = 0
  weakNeg = 0
  strongNeg = 0
  for word in document_words:
    if word in SL:
      strength, posTag, isStemmed, polarity = SL[word]
      if strength == 'weaksubj' and polarity == 'positive':
        weakPos += 1
      if strength == 'strongsubj' and polarity == 'positive':
        strongPos += 1
      if strength == 'weaksubj' and polarity == 'negative':
        weakNeg += 1
      if strength == 'strongsubj' and polarity == 'negative':
        strongNeg += 1
      features['positivecount'] = weakPos + (2 * strongPos)
      features['negativecount'] = weakNeg + (2 * strongNeg)
  
  if 'positivecount' not in features:
    features['positivecount']=0
  if 'negativecount' not in features:
    features['negativecount']=0      
  return features

# Sentiment Lexicon (LIWC) features
def liwc_features(doc, word_features, poslist, neglist):
  doc_words = set(doc)
  features = {}
  for word in word_features:
    features['contains({})'.format(word)] = (word in doc_words)
  pos = 0
  neg = 0
  for word in doc_words:
    if sentiment_read_LIWC_pos_neg_words.isPresent(word,poslist):
      pos += 1
    if sentiment_read_LIWC_pos_neg_words.isPresent(word,neglist):
      neg += 1
    features['positivecount'] = pos
    features['negativecount'] = neg
  if 'positivecount' not in features:
    features['positivecount']=0
  if 'negativecount' not in features:
    features['negativecount']=0  
  return features
  
def SL_liwc_features(doc, word_features, SL, poslist, neglist):
  document_words = set(doc)
  features = {}
  for word in word_features:
    features['contains({})'.format(word)] = (word in document_words)
  # Counting variables for the 4 classes of subjectivity
  weakPos = 0
  strongPos = 0
  weakNeg = 0
  strongNeg = 0
  for word in document_words:
    if sentiment_read_LIWC_pos_neg_words.isPresent(word,poslist):
      strongPos += 1
    elif sentiment_read_LIWC_pos_neg_words.isPresent(word,neglist):
      strongNeg += 1
    elif word in SL:
      strength, posTag, isStemmed, polarity = SL[word]
      if strength == 'weaksubj' and polarity == 'positive':
        weakPos += 1
      if strength == 'strongsubj' and polarity == 'positive':
        strongPos += 1
      if strength == 'weaksubj' and polarity == 'negative':
        weakNeg += 1
      if strength == 'strongsubj' and polarity == 'negative':
        strongNeg += 1
    features['positivecount'] = weakPos + (2 * strongPos)
    features['negativecount'] = weakNeg + (2 * strongNeg)
  
  if 'positivecount' not in features:
    features['positivecount']=0
  if 'negativecount' not in features:
    features['negativecount']=0      
  return features
  
# Processing data from Kaggle and reading the Kaggle training file then training and testing a classifier
def processkaggle(dirPath, limitStr):
  # Converting the limit argument from a string to an int
  limit = int(limitStr)
  os.chdir(dirPath)
  f = open('./train.tsv', 'r')
  # Looping over lines in the file and using the first limit of them
  phrasedata = []
  for line in f:
    # Ignoring the first line starting with Phrase and reading all lines
    if (not line.startswith('Phrase')):
      # Removing the final end of line character
      line = line.strip()
      # Each line has 4 items separated by tabs
      # Ignoring the phrase and sentence ids, and keeping the phrase and sentiment
      phrasedata.append(line.split('\t')[2:4])
  
  # Selecting a random sample of length limit because of phrase overlapping sequences
  random.shuffle(phrasedata)
  phraselist = phrasedata[:limit]
  print('Read', len(phrasedata), 'phrases, using', len(phraselist), 'random phrases')
  print ('All Phrases-')
  for phrase in phraselist[:10]:
    print(phrase)
  print(' ')
  
  # Creating a list of phrase documents as (list of words, label)
  wordtoken = []
  processtoken = []
  processpunct = []
  processvector = []
  # Adding all the phrases
  wordtoken = word_token(phraselist)
  processtoken = process_token(phraselist)
  wordpunct = wordpunct_token(phraselist)
  countvector = countvector_token(phraselist)
  
  # Printing a few
  print('Word tokenized but without preprocessing-')
  for phrase in wordtoken[:10]:
    print(phrase)
  print(' ')
  
  print('Word tokenized and preprocessed-')
  for phrase in processtoken[:10]:
    print(phrase)
  print(' ')
  
  print('Wordpunct tokenized but without preprocessing-')
  for phrase in wordpunct[:10]:
    print(phrase)
  print(' ')
	
  print('Count vector tokenized but without preprocessing-')	
  for phrase in countvector[:10]:
    print(phrase)
  print(' ')
  
  # Possibly filter tokens
  filteredtokens = rem_characters(processtoken)
  unprocessedtokens = get_words(wordtoken)
  
  # Continuing to get all words and create word features

  # For negation feature

  negative_words = ['abysmal','adverse','alarming','angry','annoy','anxious','apathy','appalling','atrocious','awful',
'bad','banal','barbed','belligerent','bemoan','beneath','boring','broken',
'callous','ca n\'t','clumsy','coarse','cold','cold-hearted','collapse','confused','contradictory','contrary','corrosive','corrupt','crazy','creepy','criminal','cruel','cry','cutting',
'dead','decaying','damage','damaging','dastardly','deplorable','depressed','deprived','deformed''deny','despicable','detrimental','dirty','disease','disgusting','disheveled','dishonest','dishonorable','dismal','distress','do n\'t','dreadful','dreary',
'enraged','eroding','evil','fail','faulty','fear','feeble','fight','filthy','foul','frighten','frightful',
'gawky','ghastly','grave','greed','grim','grimace','gross','grotesque','gruesome','guilty',
'haggard','hard','hard-hearted','harmful','hate','hideous','horrendous','horrible','hostile','hurt','hurtful',
'icky','ignore','ignorant','ill','immature','imperfect','impossible','inane','inelegant','infernal','injure','injurious','insane','insidious','insipid',
'jealous','junky','lose','lousy','lumpy','malicious','mean','menacing','messy','misshapen','missing','misunderstood','moan','moldy','monstrous',
'naive','nasty','naughty','negate','negative','never','no','nobody','nondescript','nonsense','noxious',
'objectionable','odious','offensive','old','oppressive',
'pain','perturb','pessimistic','petty','plain','poisonous','poor','prejudice','questionable','quirky','quit',
'reject','renege','repellant','reptilian','repulsive','repugnant','revenge','revolting','rocky','rotten','rude','ruthless',
'sad','savage','scare','scary','scream','severe','shoddy','shocking','sick',
'sickening','sinister','slimy','smelly','sobbing','sorry','spiteful','sticky','stinky','stormy','stressful','stuck','stupid','substandard','suspect','suspicious',
'tense','terrible','terrifying','threatening',
'ugly','undermine','unfair','unfavorable','unhappy','unhealthy','unjust','unlucky','unpleasant','upset','unsatisfactory',
'unsightly','untoward','unwanted','unwelcome','unwholesome','unwieldy','unwise','upset','vice','vicious','vile','villainous','vindictive',
'wary','weary','wicked','woeful','worthless','wound','yell','yucky',
'are n\'t','cannot','ca n\'t','could n\'t','did n\'t','does n\'t','do n\'t','had n\'t','has n\'t','have n\'t','is n\'t','must n\'t','sha n\'t','should n\'t','was n\'t','were n\'t','would n\'t',
'no', 'not', 'never', 'none', 'nowhere', 'nothing', 'noone', 'rather', 'hardly', 'scarcely', 'rarely', 'seldom', 'neither', 'nor']

  processnwords = negativewordproc(negative_words)
  # print(processnwords)
  negative_words = negative_words + processnwords
  
  # For SL feature
  SLpath = "/Users/blessythomas/Desktop/Fall (October) 2023 Term/IST 664 Natural Language Processing/Final Project/FinalProjectData/kagglemoviereviews/subjclueslen1-HLTEMNLP05.tff"
  SL = readSubjectivity(SLpath)
  
  # For LIWC feature
  poslist,neglist = sentiment_read_LIWC_pos_neg_words.read_words()
  poslist = poslist + negativewordproc(poslist)
  neglist = neglist + negativewordproc(neglist)
  
  uword_features = bag_of_words(unprocessedtokens)
  word_features = bag_of_words(filteredtokens)
  
  ubigramword_features = bag_of_words_bigram(unprocessedtokens, 300)
  bigramword_features = bag_of_words_bigram(filteredtokens, 300)
  
  print("---------------------------------------------------")
  print("Top 10 unprocessed tokens word features")
  print(uword_features[:10])
  print("---------------------------------------------------")
  print("Top 10 preprocessed tokens word features")
  print(word_features[:10])
  print("---------------------------------------------------")
  print("Top 10 unprocessed tokens word features(Bigrams)")
  print(ubigramword_features[:10])
  print("---------------------------------------------------")
  print("Top 10 preprocessed tokens word features(Bigrams)")
  print(bigramword_features[:10])
  print("---------------------------------------------------")
  
# Feature sets from a feature definition function(Without pre-processing)
  unigramsets_without_preprocessing = [(unigram_features(d, uword_features), s) for (d, s) in wordtoken]
  print(" ")
  print("Unigramsets_without_preprocessing -")
  print(unigramsets_without_preprocessing[0])
  writeFeatureSets(unigramsets_without_preprocessing,"unigramsets_without_preprocessing.csv")
  print(" ")

  bigramsets_without_preprocessing = [(bigram_features(d, uword_features,ubigramword_features), s) for (d, s) in wordtoken]
  print("Bigramsets_without_preprocessing -")
  print(bigramsets_without_preprocessing[0])
  writeFeatureSets(bigramsets_without_preprocessing,"bigramsets_without_preprocessing.csv")
  print(" ")
  
  negativesets_without_preprocessing = [(negative_features(d, uword_features,negative_words), s) for (d, s) in wordtoken]
  print("Negativesets_without_preprocessing  -")
  print(negativesets_without_preprocessing[0])
  writeFeatureSets(negativesets_without_preprocessing,"negativesets_without_preprocessing.csv")
  print(" ")
  
  possets_without_preprocessing = [(POS_features(d, uword_features), s) for (d, s) in wordtoken]
  print("POSsets_without_preprocessing -")
  print(possets_without_preprocessing[0])
  writeFeatureSets(possets_without_preprocessing,"possets_without_preprocessing.csv")
  print(" ")
  
  subjectivitysets_without_preprocessing = [(SL_features(d, uword_features,SL), s) for (d, s) in wordtoken]
  print("Subjectivitysets_without_preprocessing -")
  print(subjectivitysets_without_preprocessing[0])
  writeFeatureSets(subjectivitysets_without_preprocessing,"subjectivitysets_without_preprocessing.csv")
  print(" ")
  
  liwcsets_without_preprocessing = [(liwc_features(d, uword_features,poslist,neglist), s) for (d, s) in wordtoken]
  print("liwcsets_without_preprocessing -")
  print(liwcsets_without_preprocessing[0])
  writeFeatureSets(liwcsets_without_preprocessing,"liwcsets_without_preprocessing.csv")
  print(" ")
  
  sl_liwcsets_without_preprocessing = [(SL_liwc_features(d, uword_features,SL,poslist,neglist), s) for (d, s) in wordtoken]
  print("SL_liwcsets_without_preprocessing -")
  print(sl_liwcsets_without_preprocessing[0])
  writeFeatureSets(sl_liwcsets_without_preprocessing,"sl_liwcsets_without_preprocessing.csv")
  print(" ")
  
# Feature sets from a feature definition function(With pre-processing)
  unigramsets_with_preprocessing = [(unigram_features(d, word_features), s) for (d, s) in processtoken]
  print("Unigramsets_with_preprocessing -")
  print(unigramsets_with_preprocessing[0])
  writeFeatureSets(unigramsets_with_preprocessing,"unigramsets_with_preprocessing.csv")
  print(" ")
  
  bigramsets_with_preprocessing = [(bigram_features(d,word_features, bigramword_features), s) for (d, s) in processtoken]
  print("Bigramsets_with_preprocessing -")
  print(bigramsets_with_preprocessing[0])
  writeFeatureSets(bigramsets_with_preprocessing ,"bigramsets_with_preprocessing.csv")
  print(" ")
  
  possets_with_preprocessing = [(POS2_features(d, word_features), s) for (d, s) in wordtoken]
  print("POSsets_with_preprocessing -")
  print(possets_with_preprocessing[0])
  writeFeatureSets(possets_with_preprocessing,"possets_with_preprocessing.csv")
  print(" ")
  
  negativesets_with_preprocessing = [(negative_features(d, word_features, negative_words), s) for (d, s) in processtoken]
  print("Negativesets_with_preprocessing -")
  print(negativesets_with_preprocessing[0])
  writeFeatureSets(negativesets_with_preprocessing,"negativesets_with_preprocessing.csv")
  print(" ")

  subjectivitysets_with_preprocessing = [(SL_features(d, word_features, SL), s) for (d, s) in processtoken]
  print("Subjectivitysets_with_preprocessing -")
  print(subjectivitysets_with_preprocessing[0])
  writeFeatureSets(subjectivitysets_with_preprocessing,"subjectivitysets_with_preprocessing.csv")
  print(" ")

  liwcsets_with_preprocessing = [(liwc_features(d, word_features,poslist,neglist), s) for (d, s) in processtoken]
  print("liwcsets_with_preprocessing -")
  print(liwcsets_with_preprocessing[0])
  writeFeatureSets(liwcsets_with_preprocessing,"liwcsets_with_preprocessing.csv")
  print(" ")

  sl_liwcsets_with_preprocessing = [(SL_liwc_features(d, word_features,SL,poslist,neglist), s) for (d, s) in processtoken]
  print("SL_liwcsets_with_preprocessing -")
  print(sl_liwcsets_with_preprocessing[0])
  writeFeatureSets(sl_liwcsets_with_preprocessing,"sl_liwcsets_with_preprocessing.csv")
  print(" ")
  
# Accuracy comparison (All classifier, all featureset but singlefold)
  print("Naive Bayes Classifier")
  print("---------------------------------------------------")
  print("Accuracy with Unigramsets_without_preprocessing -: ")
  nltk_naive_bayes(unigramsets_without_preprocessing,0.1)
  print("---------------------------------------------------")
  print("Accuracy with Bigramsets_without_preprocessing -: ")
  nltk_naive_bayes(bigramsets_without_preprocessing,0.1)
  print("---------------------------------------------------")
  print("Accuracy with negativesets_without_preprocessing -: ")
  nltk_naive_bayes(negativesets_without_preprocessing,0.1)
  print("---------------------------------------------------")
  print("Accuracy with subjectivitysets_without_preprocessing -: ")
  nltk_naive_bayes(subjectivitysets_without_preprocessing,0.1)
  print("---------------------------------------------------")
  print("Accuracy with liwcsets_without_preprocessing-: ")
  nltk_naive_bayes(liwcsets_without_preprocessing,0.1)
  print("---------------------------------------------------")
  print("Accuracy with sl_liwcsets_without_preprocessing -: ")
  nltk_naive_bayes(sl_liwcsets_without_preprocessing,0.1)
  print("---------------------------------------------------")
  print("Accuracy with POSsets_without_preprocessing-: ")
  nltk_naive_bayes(possets_without_preprocessing,0.1)
  print("---------------------------------------------------")
  print("Accuracy with Unigramsets_with_preprocessing -: ")
  nltk_naive_bayes(unigramsets_with_preprocessing,0.1)
  print("---------------------------------------------------")
  print("Accuracy with Bigramsets_with_preprocessing -: ")
  nltk_naive_bayes(bigramsets_with_preprocessing,0.1)
  print("---------------------------------------------------")
  print("Accuracy with negativesets_with_preprocessing -: ")
  nltk_naive_bayes(negativesets_with_preprocessing,0.1)
  print("---------------------------------------------------")
  print("Accuracy with subjectivitysets_with_preprocessing -: ")
  nltk_naive_bayes(subjectivitysets_with_preprocessing,0.1)
  print("---------------------------------------------------")
  print("Accuracy with liwcsets_with_preprocessing-: ")
  nltk_naive_bayes(liwcsets_with_preprocessing,0.1)
  print("---------------------------------------------------")
  print("Accuracy with sl_liwcsets_with_preprocessing -: ")
  nltk_naive_bayes(sl_liwcsets_with_preprocessing,0.1)
  print("---------------------------------------------------")
  print("Accuracy with POSsets_with_preprocessing-: ")
  nltk_naive_bayes(possets_with_preprocessing,0.1) 
  print("SciKit Learner Classifier")
  print("---------------------------------------------------")
  print("Accuracy with Unigramsets_without_preprocessing -: ")
  sklearn(unigramsets_without_preprocessing,0.1)
  print("---------------------------------------------------")
  print("Accuracy with Bigramsets_without_preprocessing -: ")
  sklearn(bigramsets_without_preprocessing,0.1)
  print("---------------------------------------------------")
  print("Accuracy with negativesets_without_preprocessing -: ")
  sklearn(negativesets_without_preprocessing,0.1)
  print("---------------------------------------------------")
  print("Accuracy with subjectivitysets_without_preprocessing -: ")
  sklearn(subjectivitysets_without_preprocessing,0.1)
  print("---------------------------------------------------")
  print("Accuracy with liwcsets_without_preprocessing-: ")
  sklearn(liwcsets_without_preprocessing,0.1)
  print("---------------------------------------------------")
  print("Accuracy with sl_liwcsets_without_preprocessing -: ")
  sklearn(sl_liwcsets_without_preprocessing,0.1)
  print("---------------------------------------------------")
  print("Accuracy with POSsets_without_preprocessing-: ")
  sklearn(possets_without_preprocessing,0.1) 
  print("---------------------------------------------------")
  print("Accuracy with Unigramsets_with_preprocessing -: ")
  sklearn(unigramsets_with_preprocessing,0.1)
  print("---------------------------------------------------")
  print("Accuracy with Bigramsets_with_preprocessing -: ")
  sklearn(bigramsets_with_preprocessing,0.1)
  print("---------------------------------------------------")
  print("Accuracy with negativesets_with_preprocessing -: ")
  sklearn(negativesets_with_preprocessing,0.1)
  print("---------------------------------------------------")
  print("Accuracy with subjectivitysets_with_preprocessing -: ")
  sklearn(subjectivitysets_with_preprocessing,0.1)
  print("---------------------------------------------------")
  print("Accuracy with liwcsets_with_preprocessing-: ")
  sklearn(liwcsets_with_preprocessing,0.1)
  print("---------------------------------------------------")
  print("Accuracy with sl_liwcsets_with_preprocessing -: ")
  sklearn(sl_liwcsets_with_preprocessing,0.1)
  print("---------------------------------------------------")
  print("Accuracy with POSsets_with_preprocessing-: ")
  sklearn(possets_with_preprocessing,0.1)
  
# Training a classifier and showing its performance in cross-validation
  label_list = [0,1,2,3,4]
  num_folds = 5
  print("Unigramsets_without_preprocessing-")
  cross_validation.naive_bayes(num_folds,unigramsets_without_preprocessing, label_list)
  cross_validation.multinomialNB(num_folds,unigramsets_without_preprocessing, label_list)
  cross_validation.BernouliNB(num_folds,unigramsets_without_preprocessing, label_list)
  cross_validation.decisiontree(num_folds,unigramsets_without_preprocessing, label_list)
  cross_validation.logisticregression(num_folds,unigramsets_without_preprocessing, label_list)
  cross_validation.sgdc(num_folds,unigramsets_without_preprocessing, label_list)
  cross_validation.linearsvc(num_folds,unigramsets_without_preprocessing, label_list)
  cross_validation.randomforests(num_folds,unigramsets_without_preprocessing, label_list)
  
  print("Bigramsets_without_preprocessing-")
  cross_validation.naive_bayes(num_folds,bigramsets_without_preprocessing, label_list)
  cross_validation.multinomialNB(num_folds,bigramsets_without_preprocessing, label_list)
  cross_validation.BernouliNB(num_folds,bigramsets_without_preprocessing, label_list)
  cross_validation.decisiontree(num_folds,bigramsets_without_preprocessing, label_list)
  cross_validation.logisticregression(num_folds,bigramsets_without_preprocessing, label_list)
  cross_validation.sgdc(num_folds,bigramsets_without_preprocessing, label_list)
  cross_validation.linearsvc(num_folds,bigramsets_without_preprocessing, label_list)
  cross_validation.randomforests(num_folds,bigramsets_without_preprocessing, label_list)
  
  print("POSsets_without_preprocessing-")
  cross_validation.naive_bayes(num_folds,possets_without_preprocessing, label_list)
  cross_validation.multinomialNB(num_folds,possets_without_preprocessing, label_list)
  cross_validation.BernouliNB(num_folds,possets_without_preprocessing, label_list)
  cross_validation.decisiontree(num_folds,possets_without_preprocessing, label_list)
  cross_validation.logisticregression(num_folds,possets_without_preprocessing, label_list)
  cross_validation.sgdc(num_folds,possets_without_preprocessing, label_list)
  cross_validation.linearsvc(num_folds,possets_without_preprocessing, label_list)
  cross_validation.randomforests(num_folds,possets_without_preprocessing, label_list)
  
  print("Negativesets_without_preprocessing-")
  cross_validation.naive_bayes(num_folds,negativesets_without_preprocessing, label_list)
  cross_validation.multinomialNB(num_folds,negativesets_without_preprocessing, label_list)
  cross_validation.BernouliNB(num_folds,negativesets_without_preprocessing, label_list)
  cross_validation.decisiontree(num_folds,negativesets_without_preprocessing, label_list)
  cross_validation.logisticregression(num_folds,negativesets_without_preprocessing, label_list)
  cross_validation.sgdc(num_folds,negativesets_without_preprocessing, label_list)
  cross_validation.linearsvc(num_folds,negativesets_without_preprocessing, label_list)
  cross_validation.randomforests(num_folds,negativesets_without_preprocessing, label_list)
  
  print("Subjectivitysets_without_preprocessing")
  cross_validation.naive_bayes(num_folds,subjectivitysets_without_preprocessing, label_list)
  cross_validation.multinomialNB(num_folds,subjectivitysets_without_preprocessing, label_list)
  cross_validation.BernouliNB(num_folds,subjectivitysets_without_preprocessing, label_list)
  cross_validation.decisiontree(num_folds,subjectivitysets_without_preprocessing, label_list)
  cross_validation.logisticregression(num_folds,subjectivitysets_without_preprocessing, label_list)
  cross_validation.sgdc(num_folds,subjectivitysets_without_preprocessing, label_list)
  cross_validation.linearsvc(num_folds,subjectivitysets_without_preprocessing, label_list)
  cross_validation.randomforests(num_folds,subjectivitysets_without_preprocessing, label_list)
  
  print("LIWCsets_without_preprocessing")
  cross_validation.naive_bayes(num_folds,liwcsets_without_preprocessing, label_list)
  cross_validation.multinomialNB(num_folds,liwcsets_without_preprocessing, label_list)
  cross_validation.BernouliNB(num_folds,liwcsets_without_preprocessing, label_list)
  cross_validation.decisiontree(num_folds,liwcsets_without_preprocessing, label_list)
  cross_validation.logisticregression(num_folds,liwcsets_without_preprocessing, label_list)
  cross_validation.sgdc(num_folds,liwcsets_without_preprocessing, label_list)
  cross_validation.linearsvc(num_folds,liwcsets_without_preprocessing, label_list)
  cross_validation.randomforests(num_folds,liwcsets_without_preprocessing, label_list)
  
  print("SL+LIWC sets_without_preprocessing")
  cross_validation.naive_bayes(num_folds,sl_liwcsets_without_preprocessing, label_list)
  cross_validation.multinomialNB(num_folds,sl_liwcsets_without_preprocessing, label_list)
  cross_validation.BernouliNB(num_folds,sl_liwcsets_without_preprocessing, label_list)
  cross_validation.decisiontree(num_folds,sl_liwcsets_without_preprocessing, label_list)
  cross_validation.logisticregression(num_folds,sl_liwcsets_without_preprocessing, label_list)
  cross_validation.sgdc(num_folds,sl_liwcsets_without_preprocessing, label_list)
  cross_validation.linearsvc(num_folds,sl_liwcsets_without_preprocessing, label_list)
  cross_validation.randomforests(num_folds,sl_liwcsets_without_preprocessing, label_list)

  print("Unigramsets_with_preprocessing-")
  cross_validation.naive_bayes(num_folds,unigramsets_with_preprocessing, label_list)
  cross_validation.multinomialNB(num_folds,unigramsets_with_preprocessing, label_list)
  cross_validation.BernouliNB(num_folds,unigramsets_with_preprocessing, label_list)
  cross_validation.decisiontree(num_folds,unigramsets_with_preprocessing, label_list)
  cross_validation.logisticregression(num_folds,unigramsets_with_preprocessing, label_list)
  cross_validation.sgdc(num_folds,unigramsets_with_preprocessing, label_list)
  cross_validation.linearsvc(num_folds,unigramsets_with_preprocessing, label_list)
  cross_validation.randomforests(num_folds,unigramsets_with_preprocessing, label_list)
  
  print("Bigramsets_with_preprocessing-")
  cross_validation.naive_bayes(num_folds,bigramsets_with_preprocessing, label_list)
  cross_validation.multinomialNB(num_folds,bigramsets_with_preprocessing, label_list)
  cross_validation.BernouliNB(num_folds,bigramsets_with_preprocessing, label_list)
  cross_validation.decisiontree(num_folds,bigramsets_with_preprocessing, label_list)
  cross_validation.logisticregression(num_folds,bigramsets_with_preprocessing, label_list)
  cross_validation.sgdc(num_folds,bigramsets_with_preprocessing, label_list)
  cross_validation.linearsvc(num_folds,bigramsets_with_preprocessing, label_list)
  cross_validation.randomforests(num_folds,bigramsets_with_preprocessing, label_list)
  
  print("POSsets_with_preprocessing-")
  cross_validation.naive_bayes(num_folds,possets_with_preprocessing, label_list)
  cross_validation.multinomialNB(num_folds,possets_with_preprocessing, label_list)
  cross_validation.BernouliNB(num_folds,possets_with_preprocessing, label_list)
  cross_validation.decisiontree(num_folds,possets_with_preprocessing, label_list)
  cross_validation.logisticregression(num_folds,possets_with_preprocessing, label_list)
  cross_validation.sgdc(num_folds,possets_with_preprocessing, label_list)
  cross_validation.linearsvc(num_folds,possets_with_preprocessing, label_list)
  cross_validation.randomforests(num_folds,possets_with_preprocessing, label_list)
  
  print("Negativesets_with_preprocessing-")
  cross_validation.naive_bayes(num_folds,negativesets_with_preprocessing, label_list)
  cross_validation.multinomialNB(num_folds,negativesets_with_preprocessing, label_list)
  cross_validation.BernouliNB(num_folds,negativesets_with_preprocessing, label_list)
  cross_validation.decisiontree(num_folds,negativesets_with_preprocessing, label_list)
  cross_validation.logisticregression(num_folds,negativesets_with_preprocessing, label_list)
  cross_validation.sgdc(num_folds,negativesets_with_preprocessing, label_list)
  cross_validation.linearsvc(num_folds,negativesets_with_preprocessing, label_list)
  cross_validation.randomforests(num_folds,negativesets_with_preprocessing, label_list)
  
  print("Subjectivitysets_with_preprocessing-")
  cross_validation.naive_bayes(num_folds,subjectivitysets_with_preprocessing, label_list)
  cross_validation.multinomialNB(num_folds,subjectivitysets_with_preprocessing, label_list)
  cross_validation.BernouliNB(num_folds,subjectivitysets_with_preprocessing, label_list)
  cross_validation.decisiontree(num_folds,subjectivitysets_with_preprocessing, label_list)
  cross_validation.logisticregression(num_folds,subjectivitysets_with_preprocessing, label_list)
  cross_validation.sgdc(num_folds,subjectivitysets_with_preprocessing, label_list)
  cross_validation.linearsvc(num_folds,subjectivitysets_with_preprocessing, label_list)
  cross_validation.randomforests(num_folds,subjectivitysets_with_preprocessing, label_list)
  
  print("LIWCsets_with_preprocessing-")
  cross_validation.naive_bayes(num_folds,liwcsets_with_preprocessing, label_list)
  cross_validation.multinomialNB(num_folds,liwcsets_with_preprocessing, label_list)
  cross_validation.BernouliNB(num_folds,liwcsets_with_preprocessing, label_list)
  cross_validation.decisiontree(num_folds,liwcsets_with_preprocessing, label_list)
  cross_validation.logisticregression(num_folds,liwcsets_with_preprocessing, label_list)
  cross_validation.sgdc(num_folds,liwcsets_with_preprocessing, label_list)
  cross_validation.linearsvc(num_folds,liwcsets_with_preprocessing, label_list)
  cross_validation.randomforests(num_folds,liwcsets_with_preprocessing, label_list)
  
  print("SL + LIWCsets_with_preprocessing")
  cross_validation.naive_bayes(num_folds,sl_liwcsets_with_preprocessing, label_list)
  cross_validation.multinomialNB(num_folds,sl_liwcsets_with_preprocessing, label_list)
  cross_validation.BernouliNB(num_folds,sl_liwcsets_with_preprocessing, label_list)
  cross_validation.decisiontree(num_folds,sl_liwcsets_with_preprocessing, label_list)
  cross_validation.logisticregression(num_folds,sl_liwcsets_with_preprocessing, label_list)
  cross_validation.sgdc(num_folds,sl_liwcsets_with_preprocessing, label_list)
  cross_validation.linearsvc(num_folds,sl_liwcsets_with_preprocessing, label_list)
  cross_validation.randomforests(num_folds,sl_liwcsets_with_preprocessing, label_list)
  
# Classifer functions(Single-fold)
def nltk_naive_bayes(featuresets, percent):
  training_size = int(percent*len(featuresets))
  train_set, test_set = featuresets[training_size:], featuresets[:training_size]
  classifier = nltk.NaiveBayesClassifier.train(train_set)
  print("Naive Bayes Classifier ")
  print("Accuracy : ",nltk.classify.accuracy(classifier, test_set))
  print("Showing most informative features:")
  print(classifier.show_most_informative_features(20))
  confusionmatrix(classifier, test_set)
  print(" ")
  
def sklearn(featuresets,percent):
  training_size = int(percent*len(featuresets))
  train_set, test_set = featuresets[training_size:], featuresets[:training_size]
  classifier1 = SklearnClassifier(MultinomialNB())
  classifier1.train(train_set)
  print("ScikitLearn Classifier-MultinomialNB")
  print("Accuracy : ",nltk.classify.accuracy(classifier1, test_set))
  print(" ")
  classifier2 = SklearnClassifier(BernoulliNB())
  classifier2.train(train_set)
  print("ScikitLearn Classifier-BernoulliNB")
  print("Accuracy : ",nltk.classify.accuracy(classifier2, test_set))
  print(" ")
  classifier3 = SklearnClassifier(DecisionTreeClassifier())
  classifier3.train(train_set)
  print("ScikitLearn Classifier-Decision Tree")
  print("Accuracy : ",nltk.classify.accuracy(classifier3, test_set))
  print(" ")
  classifier4 = SklearnClassifier(LogisticRegression())
  classifier4.train(train_set)
  print("ScikitLearn Classifier-LogisticRegression")
  print("Accuracy : ",nltk.classify.accuracy(classifier4, test_set))
  print(" ")
  classifier5 = SklearnClassifier(SGDClassifier())
  classifier5.train(train_set)
  print("ScikitLearn Classifier-SGDCClassifier")
  print("Accuracy : ",nltk.classify.accuracy(classifier5, test_set))
  print(" ")
  classifier6 = SklearnClassifier(SVC())
  classifier6.train(train_set)
  print("ScikitLearn Classifier-SVC")
  print("Accuracy : ",nltk.classify.accuracy(classifier6, test_set))
  print(" ")
  classifier7 = SklearnClassifier(LinearSVC())
  classifier7.train(train_set)
  print("ScikitLearn Classifier-LinearSVC")
  print("Accuracy : ",nltk.classify.accuracy(classifier7, test_set))
  print(" ")
  classifier8 = SklearnClassifier(NuSVC(nu=0.01))
  classifier8.train(train_set)
  print("ScikitLearn Classifier-NuSVC")
  print("Accuracy : ",nltk.classify.accuracy(classifier8, test_set))
  print(" ")
  classifier9 = SklearnClassifier(RandomForestClassifier())
  classifier9.train(train_set)
  print("ScikitLearn Classifier-RandomForest")
  print("Accuracy : ",nltk.classify.accuracy(classifier9, test_set))
  print(" ")

def cross_validation_accuracy(num_folds, featuresets):
    subset_size = int(len(featuresets)/num_folds)
    accuracy_list = []
    pos_precision_list = []
    pos_recall_list = []
    pos_fmeasure_list = []
    neg_precision_list = []
    neg_recall_list = []
    neg_fmeasure_list = []
	
    # Iterating over the folds
    for i in range(num_folds):
        test_this_round = featuresets[(i*subset_size):][:subset_size]
        train_this_round = featuresets[:(i*subset_size)] + featuresets[((i+1)*subset_size):]
		
        # Training using train_this_round
        classifier = nltk.NaiveBayesClassifier.train(train_this_round)
		
        # Evaluating against test_this_round and saving accuracy
        accuracy_this_round = nltk.classify.accuracy(classifier, test_this_round)
        print(i, accuracy_this_round)
        accuracy_list.append(accuracy_this_round)
		
        refsets = collections.defaultdict(set)
        testsets = collections.defaultdict(set)
        for i, (feats, label) in enumerate(testfeats):
                refsets[label].add(i)
                observed = classifier.classify(feats)
                testsets[observed].add(i)
		
        cv_accuracy = nltk.classify.util.accuracy(classifier, testing_this_round)
        cv_pos_precision = nltk.metrics.precision(refsets['pos'], testsets['pos'])
        cv_pos_recall = nltk.metrics.recall(refsets['pos'], testsets['pos'])
        cv_pos_fmeasure = nltk.metrics.f_measure(refsets['pos'], testsets['pos'])
        cv_neg_precision = nltk.metrics.precision(refsets['neg'], testsets['neg'])
        cv_neg_recall = nltk.metrics.recall(refsets['neg'], testsets['neg'])
        cv_neg_fmeasure =  nltk.metrics.f_measure(refsets['neg'], testsets['neg'])
		
        pos_precision_list.append(cv_pos_precision)
        pos_recall_list.append(cv_pos_recall)
        neg_precision_list.append(cv_neg_precision)
        neg_recall_list.append(cv_neg_recall)
        pos_fmeasure_list.append(cv_pos_fmeasure)
        neg_fmeasure_list.append(cv_neg_fmeasure)
		
    # Finding the mean accuracy over all rounds
    print('mean accuracy-', sum(accuracy_list) / num_folds)
    print('precision-', (sum(pos_precision_list)/n + sum(neg_precision_list)/n) / 2)
    print('recall-', (sum(pos_recall_list)/n + sum(neg_recall_list)/n) / 2)
    print('fmeasure-', (sum(pos_fmeasure_list)/n + sum(neg_fmeasure_list)/n) / 2)
 

# Word tokenize functions
def word_token(phraselist):
  phrasedocs = []
  for phrase in phraselist:
    tokens = nltk.word_tokenize(phrase[0])
    phrasedocs.append((tokens, int(phrase[1])))
  return phrasedocs

def process_token(phraselist):
  phrasedocs2 = []
  for phrase in phraselist:
    tokens = nltk.word_tokenize(phrase[0])
    tokens = lower_case(tokens)
    tokens = clean_text(tokens)
    tokens = rem_no_punct(tokens)
    tokens = rem_stopword(tokens)
    tokens = stemmer(tokens)
    tokens = lemmatizer(tokens)
    phrasedocs2.append((tokens, int(phrase[1])))
  return phrasedocs2
 
def pprocess_token(phraselist):
  phrasedocs2 = []
  for phrase in phraselist:
    tokens = nltk.word_tokenize(phrase[0])
    tokens = lower_case(tokens)
    tokens = clean_text(tokens)
    tokens = rem_no_punct(tokens)
    tokens = rem_stopword(tokens)
    phrasedocs2.append((tokens, int(phrase[1])))
  return phrasedocs2
  
# Not useful tokenizers
def wordpunct_token(phraselist):
  phrasedocs3 = []
  for phrase in phraselist:
    wptokens = nltk.wordpunct_tokenize(phrase[0]) 
    phrasedocs3.append((wptokens, int(phrase[1])))
  return phrasedocs3

def countvector_token(phraselist):
  phrasedocs4 = []
  for phrase in phraselist:
    cvtokens = CountVectorizer().build_tokenizer()(phrase[0])
    phrasedocs4.append((cvtokens, int(phrase[1])))
  return phrasedocs4

# Pre-processing functions
def lower_case(doc):
  return [w.lower( ) for w in doc]
  
def clean_text(doc):
  cleantext = []
  for review_text in doc:
    review_text = re.sub(r"it 's", "it is", review_text)
    review_text = re.sub(r"that 's", "that is", review_text)
    review_text = re.sub(r"\'s", "\'s", review_text)
    review_text = re.sub(r"\'ve", "have", review_text)
    review_text = re.sub(r"wo n't", "will not", review_text)
    review_text = re.sub(r"do n't", "do not", review_text)
    review_text = re.sub(r"ca n't", "can not", review_text)
    review_text = re.sub(r"sha n't", "shall not", review_text)
    review_text = re.sub(r"n\'t", "not", review_text)
    review_text = re.sub(r"\'re", "are", review_text)
    review_text = re.sub(r"\'d", "would", review_text)
    review_text = re.sub(r"\'ll", "will", review_text)
    cleantext.append(review_text)
  return cleantext
	
def rem_no_punct(doc):
  remtext = []
  for text in doc:
    punctuation = re.compile(r'[-_.?!/\%@,":;\'{}<>~`\()|0-9]')
    word = punctuation.sub("", text)
    remtext.append(word)
  return remtext

def rem_stopword(doc):
  stopwords = nltk.corpus.stopwords.words('english')
  updatestopwords = [word for word in stopwords if word not in ['not', 'no', 'can','has','have','had','must','shan','do', 'should','was','were','won','are','cannot','does','ain', 'could', 'did', 'is', 'might', 'need', 'would']]
  return [w for w in doc if not w in updatestopwords]
  
def lemmatizer(doc):
  wnl = nltk.WordNetLemmatizer() 
  lemma = [wnl.lemmatize(t) for t in doc] 
  return lemma

def stemmer(doc):
  porter = nltk.PorterStemmer()
  stem = [porter.stem(t) for t in doc] 
  return stem
  
def negativewordproc(negativewords):
  nwords = []
  nwords = clean_text(negativewords)
  nwords = lemmatizer(nwords)
  nwords = stemmer(nwords)
  return nwords

def wordproc(word):
  wnl = nltk.WordNetLemmatizer()
  porter = nltk.PorterStemmer()
  nwords = wnl.lemmatize(word)
  nwords = porter.stem(nwords)
  return nwords
  
# Filtering functions
def rem_characters(doc):
  word_list=[]
  for (word,label) in doc:
    filtered_words = [x for x in word if len(x) > 2]
    word_list.extend(filtered_words)
  return word_list

# Other functions
def get_words(doc):
  word_list = []
  for (word, sentiment) in doc:
    word_list.extend(word)
  return word_list

# Returning a dictionary where words can be searched and the four items of subjectivity
#     information can be returned
def readSubjectivity(path):
    flexicon = open(path, 'r')
    sldict = { }
    for line in flexicon:
        fields = line.split()   #  split on whitespace
        # Splitting each field on the '=' and keeping the second part as the value
        strength = fields[0].split("=")[1]
        word = fields[2].split("=")[1]
        posTag = fields[3].split("=")[1]
        stemmed = fields[4].split("=")[1]
        polarity = fields[5].split("=")[1]
        if (stemmed == 'y'):
            isStemmed = True
        else:
            isStemmed = False
        # Putting a dictionary entry with the word as the keyword
        #     and a list of the other values
        procword = wordproc(word)
        sldict[procword] = [strength, posTag, isStemmed, polarity]
        sldict[word] = [strength, posTag, isStemmed, polarity]
    return sldict
		
def confusionmatrix(classifier_type, test_set):
  reflist = []
  testlist = []
  for (features, label) in test_set:
    reflist.append(label)
    testlist.append(classifier_type.classify(features))
  print("Confusion matrix:")
  cm = ConfusionMatrix(reflist, testlist)
  print(cm)

"""
commandline interface takes a directory name with kaggle subdirectory for train.tsv
   and a limit to the number of kaggle phrases to use
It then processes the files and trains a kaggle movie review sentiment classifier.

"""
if __name__ == '__main__':
    if (len(sys.argv) != 3):
        print ('usage: classifyKaggle.py <corpus-dir> <limit>')
        sys.exit(0)
    processkaggle(sys.argv[1], sys.argv[2])