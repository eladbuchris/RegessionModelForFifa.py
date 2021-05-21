import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

def readData():
    data = pd.read_csv("C:\\Users\\bboyc\\OneDrive\\שולחן העבודה\\סמסטר 5\\נושאים ברגרסיה\\players_20.csv")
    return data


def dataWithOutBadColumns(data):
    data = data.drop(["player_url", "long_name", "dob", "player_positions",
                      "work_rate", "body_type", "real_face", "player_tags", "team_jersey_number",
                      "loaned_from", "joined", "nation_position", "nation_jersey_number", "gk_diving", "gk_handling",
                      "gk_kicking", "gk_reflexes",
                      "gk_speed", "gk_positioning", "player_traits", "sofifa_id", "short_name",
                      "nationality", "club", "preferred_foot", "team_position"], axis=1)
    return data.astype(float)


def dataWithOutBadRows(data):
    data = data[data['pace'].notna()]
    data = data[data['release_clause_eur'].notna()]
    data.drop(data.tail(2).index, inplace=True)
    return data


def BoxCoxTransformation(λ, X):
    if λ == 0:
        return np.log(X)
    else:
        X = (np.power(X, λ) - 1) / λ
        return X


def ApplyTransformationToData(λ, X):
    for i in range(1, X.shape[1]):
        X[:, i] = BoxCoxTransformation(λ, X[:, i])
    return X


def linearRegression(X, Y):
    inverse_matrix = np.linalg.inv(np.dot(X.T, X))
    tmp = np.dot(inverse_matrix, X.T)
    beta = np.dot(tmp, Y)
    return beta


def crossValidationLinearRegression(X, Y, n_folds):
    k = 0
    i = int()
    n_k = int(X.shape[0] / n_folds)
    MSE_K = []
    while k < n_folds:
        X_test = X[i:n_k + i, :]
        X_train = np.delete(X, np.arange(i, stop=n_k + int(i)), 0)
        Y_test = Y[i:n_k + i]
        Y_train = np.delete(np.copy(Y), np.arange(i, stop=(n_k + int(i))))
        beta = linearRegression(np.copy(X_train), np.copy(Y_train))
        Y_test_hat = np.dot(X_test, beta.T)
        MSE_K.append(np.square(np.linalg.norm((Y_test - Y_test_hat))) / n_k)
        k = k + 1
        i = i + n_k
    CV_K = (1 / n_folds) * sum(MSE_K)
    return CV_K


def crossValidationBoxCox(X, Y, n_folds, λ):
    k = 0
    i = int()
    n_k = int(X.shape[0] / n_folds)
    MSE_K = []
    while k < n_folds:
        X_test = X[i:n_k + i, :]
        X_train = np.delete(X, np.arange(i, stop=n_k + int(i)), 0)
        Y_test = Y[i:n_k + i]
        Y_train = np.delete(np.copy(Y), np.arange(i, stop=(n_k + int(i))))
        X_train, beta = BoxCoxRegression(np.copy(X_train), λ, np.copy(Y_train))
        X_test = ApplyTransformationToData(λ, np.copy(X_test))
        Y_test_hat = np.dot(X_test, beta.T)
        MSE_K.append(np.square(np.linalg.norm((Y_test - Y_test_hat))) / n_k)
        k = k + 1
        i = i + n_k
    CV_K = (1 / n_folds) * sum(MSE_K)
    return CV_K

def model_for_kink(fifaData):
    X = np.zeros((fifaData.shape[0], 2))
    X[:, 0] = np.ones(fifaData.shape[0])
    X[:, 1] = fifaData["potential"]
    return X

def kink_transformation(X,c):
    X_transformed = np.zeros((X.shape[0],3))
    X_transformed[:, 0] = np.ones(X.shape[0])
    i = 0
    for x in X[:,1]:
        X_transformed[i,1] = np.max([-(x-c),0])
        X_transformed[i,2] = np.max([x-c,0])
        i += 1
    return X_transformed

def choose_c(model, Y):
    c_range = np.arange(50, 94, 0.5)
    S_n_c = []
    for c in c_range:
        X_c, beta_c = kink_Regression(np.copy(model), c, np.copy(Y))
        S_n_c.append(np.sum(np.square(Y - np.dot(X_c, beta_c.T))) / X_c.shape[0])
    plt.plot(c_range, S_n_c)
    plt.show()
    return 50 + 0.5 * np.argmin(np.array(S_n_c))


def kink_Regression(X, c, Y):
    X = kink_transformation(np.copy(X),c)
    inverse_matrix = np.linalg.inv(np.dot(X.T, X))
    tmp = np.dot(inverse_matrix, X.T)
    beta_c = np.dot(tmp, Y)
    return X, beta_c


def model1(fifaData):
    # This model tries to predict the player overall grade by his potential
    X = np.zeros((fifaData.shape[0], 2))
    X[:, 0] = np.ones(fifaData.shape[0])
    X[:, 1] = fifaData["potential"]
    return X


def model2(fifaData):
    # This model tries to predict the player overall grade by his potential and realese clause
    X = np.zeros((fifaData.shape[0], 3))
    X[:, 0] = np.ones(fifaData.shape[0])
    X[:, 1] = fifaData["potential"]
    X[:, 2] = np.log(fifaData["release_clause_eur"])
    return X


def model3(fifaData):
    # This model tries to predict the player overall grade by his potential and value
    X = np.zeros((fifaData.shape[0], 4))
    X[:, 0] = np.ones(fifaData.shape[0])
    X[:, 1] = fifaData["potential"]
    X[:, 2] = np.log(fifaData["value_eur"])
    X[:, 3] = fifaData["age"]
    return X


def model4(fifaData):
    # This model tries to predict the player overall grade by his finanncial data
    X = np.zeros((fifaData.shape[0], 4))
    X[:, 0] = np.ones(fifaData.shape[0])
    X[:, 1] = np.log(fifaData["release_clause_eur"])
    X[:, 2] = np.log(fifaData["value_eur"])
    X[:, 3] = np.log(fifaData["wage_eur"])
    return X


def model5(fifaData):
    # This model tries to predict the player overall grade by his skill moves and weak foot abilities
    X = np.zeros((fifaData.shape[0], 3))
    X[:, 0] = np.ones(fifaData.shape[0])
    X[:, 1] = fifaData["skill_moves"]
    X[:, 2] = fifaData["weak_foot"]
    return X


def model6(fifaData):
    # This model tries to predict the player overall grade by his playing abilities
    X = np.zeros((fifaData.shape[0], 7))
    X[:, 0] = np.ones(fifaData.shape[0])
    X[:, 1] = fifaData["pace"]
    X[:, 2] = fifaData["shooting"]
    X[:, 3] = fifaData["passing"]
    X[:, 4] = fifaData["dribbling"]
    X[:, 5] = fifaData["defending"]
    X[:, 6] = fifaData["physic"]
    return X


def BoxCoxRegression(X, λ, Y):
    X = ApplyTransformationToData(λ, np.copy(X))
    inverse_matrix = np.linalg.inv(np.dot(X.T, X))
    tmp = np.dot(inverse_matrix, X.T)
    beta_λ = np.dot(tmp, Y)
    return X, beta_λ


def chooseλ(model, Y):
    lambda_range = np.arange(0, 5, 0.1)
    S_n_λ = []
    for λ in lambda_range:
        X_λ, beta_λ = BoxCoxRegression(np.copy(model), λ, np.copy(Y))
        S_n_λ.append(np.sum(np.square(Y - np.dot(X_λ, beta_λ.T))) / X_λ.shape[0])
    # plt.plot(lambda_range, S_n_λ)
    # plt.show()
    return -1 + 0.1 * np.argmin(np.array(S_n_λ))


def calculateMTheta2(X, λ,beta):
    M_theta = np.zeros((X.shape[0], 2 * X.shape[1] - 1))
    M_theta[:, 0] = X[:, 0]
    len_columns = X.shape[1]
    for i in range(1, len_columns):
        M_theta[:, i] = BoxCoxTransformation(λ, X[:, i])
    for i in range(len_columns, 2 * len_columns - 1):
        M_theta[:, i] = beta[i-len_columns+1]*((1 - np.power(X[:, i - len_columns + 1], λ) +
                         λ * np.power(X[:, i - len_columns + 1], λ) * np.log(X[:, i - len_columns + 1]))) / np.square(λ)
    return M_theta

def calculateMTheta(X, λ):
    M_theta = np.zeros((X.shape[0], 2 * X.shape[1] - 1))
    M_theta[:, 0] = X[:, 0]
    len_columns = X.shape[1]
    for i in range(1, len_columns):
        M_theta[:, i] = BoxCoxTransformation(λ, X[:, i])
    for i in range(len_columns, 2 * len_columns - 1):
        M_theta[:, i] = (1 - np.power(X[:, i - X.shape[1] + 1], λ) +
                         λ * np.power(X[:, i - X.shape[1] + 1], λ) * np.log(X[:, i - X.shape[1] + 1])) / np.square(λ)
    return M_theta

def calculateQ(M):
    Q = np.dot(M.T, M) / M.shape[0]
    return Q


def calculateOmegaKink(X, Y, c):
    X, beta = kink_Regression(np.copy(X), c, np.copy(Y))
    D = np.diag(np.square(Y - np.dot(X, beta.T)))
    tmp = np.dot(X.T, D)
    Omega = np.dot(tmp, X) / X.shape[0]
    return Omega


def calculateOmegaBoxCox(M, X, Y, λ):
    X, beta = BoxCoxRegression(np.copy(X), λ, np.copy(Y))
    D = np.diag(np.square(Y - np.dot(X, beta.T)))
    tmp = np.dot(M.T, D)
    Omega = np.dot(tmp, M) / M.shape[0]
    return Omega


def calculateOmegaLinear(X, Y):
    beta = linearRegression(np.copy(X), np.copy(Y))
    D = np.diag(np.square(Y - np.dot(X, beta.T)))
    tmp = np.dot(X.T, D)
    Omega = np.dot(tmp, X) / X.shape[0]
    return Omega


def calculateV(Omega, Q):
    tmp = np.dot(np.linalg.inv(Q), Omega)
    V = np.dot(tmp, np.linalg.inv(Q))
    return V

def overFittingCheck(fifaData,Y):
    a = np.random.randint(low = 0,high=15075)
    b = np.random.randint(low = a,high=15075)
    X, beta = BoxCoxRegression(np.copy(model3(fifaData)[a:b,:]), 0.5, np.copy(Y[a:b]))
    X_test = np.delete(np.copy(model3(fifaData)), np.arange(a, stop=b),0)
    Y_test = np.delete(np.copy(Y), np.arange(a, stop=b))
    prediction = np.dot(ApplyTransformationToData(0.5,X_test),beta.T)
    return np.sum(np.square(Y_test-prediction))/X_test.shape[0]


def calculateMTheta2(X, λ,beta):
    M_theta = np.zeros((X.shape[0], 2 * X.shape[1] - 1))
    M_theta[:, 0] = X[:, 0]
    len_columns = X.shape[1]
    for i in range(1, len_columns):
        M_theta[:, i] = BoxCoxTransformation(λ, X[:, i])
    for i in range(len_columns, 2 * len_columns - 1):
        M_theta[:, i] = beta[i-X.shape[1]+1]*((1 - np.power(X[:, i - X.shape[1] + 1], λ) +
                         λ * np.power(X[:, i - X.shape[1] + 1], λ) * np.log(X[:, i - X.shape[1] + 1]))) / np.square(λ)
    return M_theta

def crossValidationLasso(X, Y, n_folds, L_rate):
    k = 0
    i = int()
    n_k = int(X.shape[0] / n_folds)
    MSE_K = []
    while (k < n_folds):
        X_test = X[i:n_k + i, :]
        X_train = np.delete(X, np.arange(i, stop=n_k + int(i)), 0)
        Y_test = Y[i:n_k + i]
        Y_train = np.delete(np.copy(Y), np.arange(i, stop=(n_k + int(i))))
        Y_test_hat = linear_model.Lasso(alpha=L_rate, max_iter=3000).fit(X_train, Y_train).predict(X_test)
        MSE_K.append(np.square(np.linalg.norm((Y_test - Y_test_hat))) / n_k)
        k = k + 1
        i = i + n_k
    CV_K = (1 / n_folds) * sum(MSE_K)
    return CV_K


def main():
    lambdas = np.arange(6,18,1)
    cvk_vec =[]
    fifaData = readData()
    fifaData = dataWithOutBadColumns(fifaData)
    fifaData = dataWithOutBadRows(fifaData)
    Y = fifaData["overall"]
    for l in lambdas:
        cvk_vec.append(crossValidationLasso(fifaData.to_numpy(),Y,5,l))
    plt.plot(cvk_vec)
    plt.show()

    #print(choose_c(model_for_kink(fifaData),Y))
    X_kink, beta_king = kink_Regression(np.copy(model_for_kink(fifaData)),84.5,np.copy(Y))
    prediction = np.dot(X_kink,beta_king.T)
    print(beta_king)
    mse = np.square(np.linalg.norm(Y-prediction)) / len(prediction)
    plt.scatter(x = fifaData["potential"],y = prediction)
   # plt.scatter(x = fifaData["potential"],y = Y,color = 'orange')
    plt.legend(["true"])
    plt.title("true - player overall grade")
    plt.xlabel("potential")
    plt.ylabel("overall")
    plt.show()
    Q_kink = calculateQ(X_kink)
    Omega_kink = calculateOmegaKink(np.copy(model_for_kink(fifaData)),Y,84.5)
    V_kink = calculateV(Omega_kink,Q_kink)
    print("nice")
    models = [model1(fifaData), model2(fifaData), model3(fifaData), model4(fifaData), model5(fifaData),
              model6(fifaData)]
    models_lambda = []
    models_cvk_BoxCox = []
    model_cvk_linear = []
    for model in models:
        models_lambda.append(chooseλ(model, np.copy(Y)))
        models_cvk_BoxCox.append(crossValidationBoxCox(model, Y, 5, chooseλ(model, np.copy(Y))))
        model_cvk_linear.append(crossValidationLinearRegression(np.copy(model), np.copy(Y), 5))
    print("The best lambda for each model is : ")
    print(models_lambda)
    print("The best model of the BoxCox Regression via cross validation is : " + str(np.argmin(models_cvk_BoxCox) + 1)
          + " And it's CV error is : " + str(np.min(models_cvk_BoxCox)))
    print("The best model of the linear Regression via cross validation is : " + str(np.argmin(model_cvk_linear) + 1) +
          " And it's CV error is : " + str(np.min(model_cvk_linear)))
    Q_linear = calculateQ(model3(fifaData))
    Omega_linear = calculateOmegaLinear(model3(fifaData), Y)
    V_linear = calculateV(Omega_linear, Q_linear)
    X, beta = BoxCoxRegression(np.copy(model1(fifaData)), 2.1, np.copy(Y))
    M_BoxCox2 = calculateMTheta2(model1(fifaData), 0.9,beta)
    Q_BoxCox2 = calculateQ(M_BoxCox2)
    Omega_BoxCox2 = calculateOmegaBoxCox(M_BoxCox2, model1(fifaData), Y, 0.9)
    V_BoxCox2 = calculateV(Omega_BoxCox2, Q_BoxCox2)
    beta_linear = linearRegression(np.copy(model3(fifaData)),np.copy(Y))
    prediction = np.dot(X, beta.T)
    prediction_linear = np.dot(np.copy(model3(fifaData)),beta_linear.T)
    R_squared = np.square(np.linalg.norm(prediction_linear-np.mean(Y)))/np.square(np.linalg.norm(Y-np.mean(Y)))

    plt.scatter(x = fifaData["potential"],y = prediction)
    plt.scatter(x = fifaData["age"],y = prediction_linear)
    plt.scatter(x = fifaData["potential"],y = Y)
    plt.legend()
    plt.show()

    overFittingCheck(fifaData, Y)
    print(beta)
    print(model_cvk_linear)
    print(models_cvk_BoxCox)
    print("good job")


if __name__ == '__main__':
    main()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
