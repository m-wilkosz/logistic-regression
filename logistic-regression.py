import numpy
import pandas

### Funkcja decyzyjna
def classify(theta_s, theta_vc, theta_vg, X):
    prob1 = logisticRegression(theta_s, X)
    prob2 = logisticRegression(theta_vc, X)
    prob3 = logisticRegression(theta_vg, X)
    return numpy.where(prob1 > prob2, numpy.where(prob1 > prob3, 'Iris-setosa', 'Iris-virginica'),
                                      numpy.where(prob2 > prob3, 'Iris-versicolor', 'Iris-virginica'))


### Funkcja dwuklasowej regresji logistycznej - nasza hipoteza h
def logisticRegression(theta, X):
    return 1.0 / (1.0 + numpy.exp(-X * theta))


# Gradient dla regresji logistycznej
def costFunctionGradient(h, theta, X, y):
    return 1.0 / len(y) * (X.T * (logisticRegression(theta, X) - y))


### Funkcja kosztu - nasza funkcja J
def costFunction(h, theta, X, y):
    m = len(y)
    h_val = logisticRegression(theta, X)
    s1 = numpy.multiply(y, numpy.log(h_val))
    s2 = numpy.multiply((1 - y), numpy.log(1 - h_val))
    return -numpy.sum(s1 + s2, axis=0) / m


### Funkcja wyznaczająca optymalne wartości współczynników dwuklasowej regresji logistycznej
def gradient_descent(h, fJ, fdJ, theta, X, y, alpha=0.01, eps=10**-3, maxSteps=10000):
    errorCurr = fJ(h, theta, X, y)
    errors = [[errorCurr, theta]]
    while True:
        # oblicz nowe theta
        theta = theta - alpha * fdJ(h, theta, X, y)
        # raportuj poziom błędu
        errorCurr, errorPrev = fJ(h, theta, X, y), errorCurr
        # kryteria stopu
        if abs(errorPrev - errorCurr) <= eps:
            break
        if len(errors) > maxSteps:
            break
        errors.append([errorCurr, theta]) 
    return theta, errors


### Funkcja standaryzująca
def standardize(val):
    return (val - numpy.mean(val)) / numpy.std(val)


### Funkcja normalizująca
def normalize(val):
    return val / numpy.amax(val, axis=0)


def main():
    numpy.seterr(over='raise')

    ### Wczytanie zbioru uczącego
    train = pandas.read_csv('multi_classes/train/train.csv', sep=';', header=None)

    ### Wybieramy kolumny, które będą naszymi informacjami a następne normalizujemy i standaryzujemy je
    X = train.iloc[:,:-1]
    X = standardize(normalize(X))
    Xm, Xn = X.shape
    X = X.values.reshape(Xm, Xn)
    X = numpy.matrix(numpy.concatenate((numpy.ones((Xm, 1)), X), axis=1)).reshape(Xm, Xn + 1)

    ### Wybieramy ze zbioru uczącego wartości zmiennej, która będzie naszą klasyfikacją
    y = train.iloc[:,-1:]
    y = y.values.reshape(Xm, 1)

    y_s = numpy.copy(y)
    y_vc = numpy.copy(y)
    y_vg = numpy.copy(y)
    for i, value in numpy.ndenumerate(y):
        if (value == 'Iris-setosa'):
            y_s[i], y_vg[i], y_vc[i] = 1, 0, 0
        elif (value == 'Iris-versicolor'):
            y_s[i], y_vg[i], y_vc[i] = 0, 0, 1
        else:
            y_s[i], y_vg[i], y_vc[i] = 0, 1, 0

    y_s = y_s.astype(float)
    y_vc = y_vc.astype(float)
    y_vg = y_vg.astype(float)

    ### Uruchamiamy trzykrotnie algorytm gradientu prostego do wyznaczenia optymalnych wartości współczynników dwuklasowej regresji logistycznej
    theta_s, errors_s = gradient_descent(logisticRegression, costFunction, costFunctionGradient, numpy.ones(Xn + 1).reshape(Xn + 1, 1), X, y_s, 0.1, 10**-7, 1000)
    theta_vc, errors_vc = gradient_descent(logisticRegression, costFunction, costFunctionGradient, numpy.ones(Xn + 1).reshape(Xn + 1, 1), X, y_vc, 0.1, 10**-7, 1000)
    theta_vg, errors_vg = gradient_descent(logisticRegression, costFunction, costFunctionGradient, numpy.ones(Xn + 1).reshape(Xn + 1, 1), X, y_vg, 0.1, 10**-7, 1000)

    ### Gdy już mamy funkcję regresji, możemy dokonać klasyfikacji
    dev = pandas.read_csv('multi_classes/dev/in.csv', sep=';', header=None)
    dev = standardize(normalize(dev))
    Dm, Dn = dev.shape
    dev = dev.values.reshape(Dm, Dn)
    dev = numpy.matrix(numpy.concatenate((numpy.ones((Dm, 1)), dev), axis=1)).reshape(Dm, Dn + 1)

    classification = classify(theta_s, theta_vc, theta_vg, dev)
    numpy.savetxt('multi_classes/dev/out.csv', classification, '%s')

    expected = pandas.read_csv('multi_classes/dev/expected.csv', sep=';', header=None)
    expected = expected.values
    ### Obliczamy miary ewaluacji
    accuracy = 0
    TP_s, TN_s, FP_s, FN_s, P_s, N_s = 0, 0, 0, 0, 0 , 0
    TP_vc, TN_vc, FP_vc, FN_vc, P_vc, N_vc = 0, 0, 0, 0, 0 , 0
    TP_vg, TN_vg, FP_vg, FN_vg, P_vg, N_vg = 0, 0, 0, 0, 0 , 0
    for i, value in numpy.ndenumerate(expected):
        if (expected[i] == classification[i]):
            accuracy += 1
        # Klasa Iris-setosa
        if (expected[i] == 'Iris-setosa'):
            P_s += 1
            if (classification[i] == 'Iris-setosa'):
                TP_s += 1
            else:
                FN_s += 1
        else:
            N_s += 1
            if (classification[i] != 'Iris-setosa'):
                TN_s += 1
            else:
                FP_s += 1
        # Klasa Iris-versicolor
        if (expected[i] == 'Iris-versicolor'):
            P_vc += 1
            if (classification[i] == 'Iris-versicolor'):
                TP_vc += 1
            else:
                FN_vc += 1
        else:
            N_vc += 1
            if (classification[i] != 'Iris-versicolor'):
                TN_vc += 1
            else:
                FP_vc += 1
        # Klasa Iris-virginica
        if (expected[i] == 'Iris-virginica'):
            P_vg += 1
            if (classification[i] == 'Iris-virginica'):
                TP_vg += 1
            else:
                FN_vg += 1
        else:
            N_vg += 1
            if (classification[i] != 'Iris-virginica'):
                TN_vg += 1
            else:
                FP_vg += 1
    # Obliczamy wartości makro
    macro_precision = 1/3 * ((TP_s/(TP_s+FP_s)) + (TP_vc/(TP_vc+FP_vc)) + (TP_vg/(TP_vg+FP_vg)))
    macro_recall = 1/3 * ((TP_s/P_s) + (TP_vc/P_vc) + (TP_vg/P_vg))
    macro_specifity = 1/3 * ((TN_s/N_s) + (TN_vc/N_vc) + (TN_vg/N_vg))
    macro_F1 = 1/3 * ((2 * TP_s/(TP_s+FP_s)*TP_s/P_s / (TP_s/(TP_s+FP_s) + TP_s/P_s)) +
                      (2 * TP_vc/(TP_vc+FP_vc)*TP_vc/P_vc / (TP_vc/(TP_vc+FP_vc) + TP_vc/P_vc)) +
                      (2 * TP_vg/(TP_vg+FP_vg)*TP_vg/P_vg / (TP_vg/(TP_vg+FP_vg) + TP_vg/P_vg)))

    # Obliczamy wartości mikro
    micro_precision = (TP_s + TP_vc + TP_vg)/(TP_s + TP_vc + TP_vg + FP_s + FP_vc + FP_vg)
    micro_recall = (TP_s + TP_vc + TP_vg)/(TP_s + TP_vc + TP_vg + FN_s + FN_vc + FN_vg)
    micro_specifity = (TN_s + TN_vc + TN_vg)/(TN_s + TN_vc + TN_vg + FP_s + FP_vc + FP_vg)
    micro_F1 = 2 * (micro_precision*micro_recall) / (micro_precision+micro_recall)

    accuracy = accuracy / len(expected)

    print('Accuracy: ' + str(accuracy))
    print('Macro-average precision: ' + str(macro_precision))
    print('Macro-average recall: ' + str(macro_recall))
    print('Macro-average specifity: ' + str(macro_specifity))
    print('Macro-average F1: ' + str(macro_F1))
    print('Micro-average precision: ' + str(micro_precision))
    print('Micro-average recall: ' + str(micro_recall))
    print('Micro-average specifity: ' + str(micro_specifity))
    print('Micro-average F1: ' + str(micro_F1))

main()
