from hydrat.frameworks.crossvalidation import CrossValidationFramework
import hydrat.corpora.dummy as dummy

#import hydrat.classifier.NLTK as nltk
#import hydrat.classifier.SVM as svm
import hydrat.classifier.knn as knn
#import hydrat.classifier.nearest_prototype as np
#import hydrat.classifier.weka as weka
#import hydrat.classifier.maxent as maxent

if __name__ == "__main__":
  learners=\
    [ knn.cosine_1nnL()
#    , nltk.naivebayesL()
#    , nltk.decisiontreeL()
#    , svm.libsvmExtL(kernel_type='linear')
#    , svm.bsvmL(kernel_type='linear')
    , knn.skew_1nnL()
#    , knn.oop_1nnL()
#    , np.cosine_mean_prototypeL()
#    , maxent.maxentLearner()
#    , weka.majorityclassL()
#    , weka.nbL()
#    , weka.j48L()
    ]
  cv = CrossValidationFramework(dummy.unicode_dummy_multiclass())
  cv.set_class_space('dummy_multiclass')
  cv.set_feature_space('byte_unigram')
  for l in learners:
    cv.set_learner(l)
    cv.run()
  cv.generate_output()
