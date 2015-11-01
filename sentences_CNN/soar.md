#Dynamic Convolutional Neural Network SOAR

Based on:
http://nal.co/papers/Kalchbrenner_DCNN_ACL14

##Specify
The hope is to take sentences or small segments of text to classify what the text is about or
predict sentiment.

First it will be applied to a sentence corpus from academic papers predicting one of 5 classes.
Paper with original results: http://www.cs.pomona.edu/~achambers/papers/thesis_final.pdf
The full sentence dataset: http://archive.ics.uci.edu/ml/datasets/Sentence+Classification

In addition to the pure classification that we get out of the model we also want to be able to create
embeddings from the sentences.

###Goal
The ultimate goal of this line of research is to support mission planning, course of action evaluation,
and predicting measures of effectiveness (i.e. mission outcomes).


##Observe

###Metrics
Aside from pure accuracy, the main metrics we will be looking at will be macro and micro F1-score.

The micro f1 first calculates one precision and one recall based on the true positives, false
positives, etc. of the individual classes

i.e. Micro-average of precision = (TP1+TP2..+TPn)/(TP1+TP2...+TPn+FP1+FP2...+TPn)

similarly for recall and then the harmonic mean is taken.
(I'm pretty sure in the case where you may only predict one class and only one class is correct that
this is the same as pure accuracy)

The macro f1 finds the precision of each class individually and then averages for a macro-precision

Macro-average precision = (P1+P2...+Pn)/N

and then similar average for recall and the harmonic mean is taken.
For more a little more : http://rushdishams.blogspot.com/2011/08/micro-and-macro-average-of-precision.html

###Implementation

The implementation is built to run on nvidia GPUs utilizing Theano.

It is currently restricted to 3 layers.

It is not mentioned in the paper but in the author's matlab code use of dropout is also included.

Using the tanh activation function also yielded better results for me.

We are also utilizing the AdaDelta modification to SGD to improve training speed.

Also available are some modified cost functions that are supposed to improve accuracy on datasets
with noisy labels:

bootstrap-soft and bootstrap-hard.

http://arxiv.org/pdf/1412.6596v2.pdf

I only had the chance to try using b-soft and it did not seem to have much effect. In the paper
it seemed to improve accuracy only a percent or two at most anyway unless there really is a
significant amount of noise (i.e. like 40% of labels are corrupted).



###Academic Sentences Dataset
3117 total labeled sentences

- Category - Proportion of labeled data
- AIMX - the aim or goal of the paper - 0.062
- BASE - the basis of the research - 0.019
- MISC - miscellaneous sentences - 0.585
- OWNX - references to other works by the author - 0.278
- CONT - contrasting this work to other works - 0.0545


##Analyze

###Parameters
The main parameters to change are the width and number of filters used in the convolution.
Wider and more filters improve performance but increase training time.
Thanks to AdaDelta changing the decay and epsilon for gradient descent do not have significant impact
on performance. Using or not using dropout also seemed to have little effect.
The layer at which folding is applied may also be changed but I never did experiments altering that.

###Academic Sentences Dataset
|Measure|Sent-LDA-S*|Sent-LDA-W*|MC-LDA*|SVM|Naive Bayes| d2v-logreg| _DCNN-alldata_ | _DCNN-20% untrained_ |
|-------|------------|----------|-------|----|----------|-----------|-----------------|------------------|
|Macro-F1| 0.30 | 0.32 | 0.33 | 0.23 | 0.22 | 0.34 | 0.81 | 0.64 |
|Micro-F1| 0.53 | 0.57 | 0.57 | 0.50 | 0.47 | 0.67 | 0.91 | 0.83 |

The first 5 columns are pulled from the phd dissertation associated with this dataset.
In that dissertation all the datasets that require supervised training were trained and evaluated on
the exact same set of all 100% of the labeled sentences. The special variants of LDA that are proposed in
the dissertation were actually trained on a separate set of unlabeled sentences and then evaluated
on the labeled data.

A doc2vec model was also trained on all the labeled sentences with seeded 50d word vectors.
70% of the labeled sentence vectors were then used to train a logistic regression. It was then
tested over all 100% of the labeled sentences to produce the metrics.

The DCNN was only trained on 70% of the labelled sentences. It was validated on 10% during training
to check that we did not overfit the training dataset. The alldata column shows metrics for testing
over all 100% of the data and the final column shows the metrics for the 20% of labeled sentences
that the model had never seen.

For more details and you can look at the cnn-SentCorpus markdown or ipython notebook.

##Recommend

It seems very promising.
Need to add the ability to add arbitrary number of layers, be more general.


