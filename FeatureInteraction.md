# Introduction #

When modelling data, we often find that we come up with multiple feature sets to represent the same data. Often, it is interesting to look at different combinations of feature sets, to see how they interact with each other and with classifiers. This document describes the support available from hydrat in implementing such techniques.

# Feature Maps as atoms #
Feautre maps are generally considered atomic in hydrat. While not impossible, no facility is provided to extract subsets of them. The advantage of doing this is that we can come up with a fairly compact descriptor for describing sets of features.

# Feature Descriptors #
Feature descriptors are the cornerstone of hydrat's flexible feature management. They allow us to uniquely identify transformations that have been applied to a featureset. This serves two purposes:
  1. Identifying partial computations to avoid recomputation
  1. Identifying results

Feature descriptors are implemented as tuples. (TODO formalize! Describe how union and transforms are denoted).