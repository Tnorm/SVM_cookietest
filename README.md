# SVM_cookietest

A test for SVM algorithm with a simple kernel search used for a classification task on some datasets from UCI databse.

Read the instructions from: https://github.com/LeoYu/neural-tangent-kernel-UCI


How to run?

1- Run uci_test.py

2- Get the accuracies from https://github.com/LeoYu/neural-tangent-kernel-UCI

3- Run plotter to get the Wilcoxon p-val

----

Average accuracy of my simple test is 0.8155, and the p-val is around 0.37>0.05, which means we can't reject the null hypo.

The runtime of this code on my machine is not surprisingly around 50x faster compared to the code base of this repo.

Not sure if the Wilcox test used is valid in my code since looking at the dataset names, many of them don't seem to be independent of each other.

There is a huge literature for kernel search which can be probably used to boost the results. I don't have any plan for that, cleaning the codes, or do overfitting to the datasets.

The idea behind the Neural Tangent Kernel is quite beautiful. However, I believe that to show that it is superior to traditional kernel approaches the results in the paper https://arxiv.org/pdf/1910.01663.pdf is not enough to show the superiority compared to old school approaches.


Best wishes. :)

