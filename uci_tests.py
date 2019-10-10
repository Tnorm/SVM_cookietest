import argparse
import os
import numpy as np
from sklearn import svm
from sklearn.gaussian_process.kernels import RBF, DotProduct, ConstantKernel
import pickle
import time

hi = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('-file', default="result.log", type=str, help="Output File")

MAX_N_TOT = 5000
C_LIST = [10.0 ** i for i in [1.0, 0.0, -1.0]]

def SVMcookie(X1, X2, y1, y2, C, kernel):
    n_val, n_train = X2.shape[0], X1.shape[0]
    clf = svm.SVC(kernel=kernel, C=C, cache_size = 100000, shrinking=False, gamma='scale')
    clf.fit(X1, y1)
    z = clf.predict(X2)
    return 1.0 * np.sum(z == y2) / n_val

avg_acc_list = []
outf = open('res.pkl', "w")
print ("Dataset\tValidation Acc\tTest Acc", file = outf)

for idx, dataset in enumerate(sorted(os.listdir('data'))):
        if not os.path.isdir('data' + "/" + dataset):
            continue
        if not os.path.isfile('data' + "/" + dataset + "/" + dataset + ".txt"):
            continue
        dic = dict()
        for k, v in map(lambda x: x.split(), open('data' + "/" + dataset + "/" + dataset + ".txt", "r").readlines()):
            dic[k] = v
        c = int(dic["n_clases="])
        d = int(dic["n_entradas="])
        n_train = int(dic["n_patrons_entrena="])
        n_val = int(dic["n_patrons_valida="])
        n_train_val = int(dic["n_patrons1="])
        n_test = 0
        if "n_patrons2=" in dic:
            n_test = int(dic["n_patrons2="])
        n_tot = n_train_val + n_test

        if n_tot > MAX_N_TOT or n_test > 0:
            print (str(dataset) + '\t0\t0', file = outf)
            continue

        print (idx, dataset, "\tN:", n_tot, "\td:", d, "\tc:", c)

        # load data
        f = open("data/" + dataset + "/" + dic["fich1="], "r").readlines()[1:]
        X = np.asarray(list(map(lambda x: list(map(float, x.split()[1:-1])), f)))
        y = np.asarray(list(map(lambda x: int(x.split()[-1]), f)))

        # load training and validation set
        fold = list(map(lambda x: list(map(int, x.split())),
                        open('data' + "/" + dataset + "/" + "conxuntos.dat", "r").readlines()))
        train_fold, val_fold = fold[0], fold[1]
        best_acc = 0.0

        best_C = 1.0,
        best_kernel = 'rbf'

        for C in C_LIST:
                # You may do a smarter kernel search (there is a huge literature around). I am not smart and don't plan to overfit to accuracy
                for kernel in ['rbf', 'linear',
                               RBF(length_scale=0.1) + ConstantKernel(),
                               RBF(length_scale=0.01) + ConstantKernel(),
                               RBF(length_scale=10.0) + ConstantKernel(),
                               RBF()*DotProduct() + ConstantKernel(),
                               RBF(length_scale=0.1)*DotProduct() + ConstantKernel(),
                               RBF(length_scale=0.01)*DotProduct() + ConstantKernel(),
                               RBF(length_scale=10.0)*DotProduct() + ConstantKernel(),
                               RBF() + 0.1*DotProduct() + ConstantKernel(),
                               RBF() + 0.5*DotProduct() + ConstantKernel(),
                               RBF() + DotProduct() + ConstantKernel()]:
                    acc = SVMcookie(X[train_fold,:], X[val_fold,:], y[train_fold], y[val_fold],C, kernel)
                    if acc > best_acc:
                        best_acc = acc
                        best_C = C
                        best_kernel = kernel

        print(best_C, best_kernel)

        # 4-fold cross-validating
        avg_acc = 0.0
        fold = list(map(lambda x: list(map(int, x.split())),
                        open("data/" + dataset + "/" + "conxuntos_kfold.dat", "r").readlines()))
        for repeat in range(4):
            train_fold, test_fold = fold[repeat * 2], fold[repeat * 2 + 1]
            acc = SVMcookie(X[train_fold,:], X[test_fold,:], y[train_fold], y[test_fold], best_C, best_kernel)
            avg_acc += 0.25 * acc

        print ("acc:", avg_acc, "\n")
        print (str(dataset) + '\t' + str(best_acc * 100) + '\t' + str(avg_acc * 100), file = outf)
        avg_acc_list.append(avg_acc)

print ("avg_acc:", np.mean(avg_acc_list) * 100)
outf.close()

print("cookie monster")
with open('svm.pkl', 'wb') as resfile:
    pickle.dump(avg_acc_list, resfile)
resfile.close()

bye = time.time()

print(bye - hi)