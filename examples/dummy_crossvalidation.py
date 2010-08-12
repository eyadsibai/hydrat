from hydrat.frameworks.crossvalidation import CrossValidation
from hydrat.corpora.dummy import unicode_dummy
from hydrat.classifier.NLTK import naivebayesL, decisiontreeL
from hydrat.classifier.SVM import bsvmL, libsvmExtL
from hydrat.classifier.knn import cosine_1nnL, skew_1nnL, oop_1nnL
from hydrat.classifier.nearest_prototype import cosine_mean_prototypeL 
#from hydrat.classifier.weka import weka_majorityclassL

if __name__ == "__main__":
  learners=\
    [ bsvmL(kernel_type='linear')
    #, naivebayesL()
    #, decisiontreeL()
    #, libsvmExtL(kernel_type='linear')
    #, cosine_1nnL()
    #, skew_1nnL()
    #, oop_1nnL()
    #, cosine_mean_prototypeL()
    #, weka_majorityclassL()
    ]
  cv = CrossValidation(unicode_dummy(), 'byte_unigram', 'dummy_default')
  for l in learners:
    cv.run_learner(l)
  cv.generate_output()
