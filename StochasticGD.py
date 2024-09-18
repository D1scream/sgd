from time import sleep
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split


from Standart_Scaler import StandartScaler


class SGDRidgeModel:
    
    def __init__(self, iters=10000, tau=0.001, eps=1e-3,alpha=0.0001):
        """
        iters: Количество эпох, сколько раз модель пройдет по данным.
        tau: Коэффициент регуляризации (L2). Контролирует силу регуляризации.
        eps: Критерий сходимости. Если изменение функции потерь меньше этого значения, обучение останавливается.
        alpha: параметр сглаживания для EMA. 
        """
        self.iters = iters
        self.tau = tau
        self.eps = eps
        self.alpha=alpha
        self.weights = None # Веса модели, инициализируются нулями
        self.bias = 0 # Смещение (свободный член), инициализируется нулем
        self.q_values = []  # Для хранения значений функции потерь

    def fit(self, X, y):
        """
        Обучение модели с использованием SGD.
        Минимизируется функция потерь
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)  # Инициализация весов
        #self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0
        self.q_values = [] 
        ema_loss = self.compute_ridge_loss(X, y)

        # перемешивание данных6
        random_indices = np.random.randint(0, n_samples, size=self.iters)

        # Основной цикл обучения
        for i in range(0, self.iters):
        
            # Стохастическое обновление весов для каждого объекта
            X_i = X[random_indices[i]]
            y_i = y[random_indices[i]]
            h = 1.0 / (i + 1)  # Уменьшающийся темп обучения

            prediction = self.predict(X_i)
            
            # Обновление весов с учетом регуляризации (ридж-регрессия)
            self.weights = self.weights * (1 - h * self.tau) - h * (prediction - y_i) * X_i
            self.bias = self.bias - h * (prediction - y_i)
        
            # Вычисление функции потерь на каждой итерации
            q_i = self.compute_ridge_loss(X, y)
            self.q_values.append(q_i)
            current_loss = q_i
            ema_loss = self.alpha * current_loss + (1 - self.alpha) * ema_loss
            
            # Критерий сходимости на основе EMA
            if i > 1 and abs(current_loss - ema_loss) / abs(ema_loss) < self.eps:
                print(f"Сходимость достигнута на итерации {i}")
                break

    def compute_ridge_loss(self, X, y):
        """
        Вычисление функции потерь для Ridge-регрессии
        
        L(w, b) = (1/n) * sum((w^T * x_i + b - y_i)^2) + (tau / 2) * sum(w_j^2)
        где первая часть — среднеквадратичная ошибка (MSE), вторая часть — регуляризация (L2-норма)
        """
        
        y_pred = self.predict(X)
        errors = y_pred - y  # Ошибки модели
        mse_loss = np.mean(errors ** 2)  # Среднеквадратичная ошибка (MSE)
        regularization = (self.tau / 2) * np.sum(self.weights ** 2)  # Регуляризация (L2-норма)
        return mse_loss + regularization # значение функции потерь
                
    def predict(self, X):
        """
        Предсказание значений целевой переменной для новых данных
        y_pred = w^T * X + b

        yi_pred = w1 * xi1 + w2 * xi2 + ... + wm * xim + b, где
        wm: вес m признака
        xi: объект, набор признаков
        b: свободный коэффициент

        """
        return np.dot(X, self.weights) + self.bias #Вектор предсказанных значений
    
    def r2(self, X, y):
        """
        Вычисление коэффициента детерминации (R^2)
        Коэффициент детерминации используется для оценки качества модели регрессии
        Он показывает, какую долю изменчивости зависимой переменной объясняет модель
        R^2 = 1 - (RSS / TSS)
        где 
        RSS -- сумма квадратов ошибок (Residual Sum of Squares),
        TSS -- полная сумма квадратов (Total Sum of Squares)
        """
        y_pred = self.predict(X)  # Предсказанные значения
        rss = np.sum((y - y_pred) ** 2)  # Остаточная сумма квадратов (RSS)
        tss = np.sum((y - np.mean(y)) ** 2)  # Полная сумма квадратов (TSS)
        if(1-rss/tss<0):
            pass
        return 1 - (rss / tss)  # Вычисление R^2 показывает долю дисперсии, объясненную моделью


if __name__ == "__main__":
    '''
        Задача: нужно предсказать цену недвижимости по некоторым признакам, используя данные уже известных недвижимостей и их цен
        для этого необходимо построить модель линейной регрессии, которая будет предсказывать цену нового объекта
        нужно минимизировать разницу между предсказанными ценами и фактическими ценами
        L(w,b) = 1/n*sum(y - y_pred)^2, где
            w: веса
            b: смещение (bias)
            y: вектор ответов 
            y_pred: вектор предсказанных моделью ответов
        L(w,b) - среднеквадратичная ошибка (MSE)
        y_pred = w^T * X + b

        Чтобы минимизировать функцию потерь, нужно праивльно настроить веса,
        для этого воспользуемся методом стохастического градиентного спуска SGD
        В SGD происходит обновление весов на основе одного случайного наблюдения в каждой итерации


    '''
    data = fetch_california_housing()
    X, y = data.data, data.target
    #print(data.feature_names)
    # Список признаков
    # 'MedInc'      Median income (Медианный доход) в районе
    # 'HouseAge'    Средний возраст домов в районе
    # 'AveRooms'    Среднее количество комнат на дом
    # 'AveBedrms'   Среднее количество спален на дом
    # 'Population'  Численность населения в районе
    # 'AveOccup'    Среднее количество жильцов на дом
    # 'Latitude'    Широта
    # 'Longitude'   Долгота

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    #Стандартизация признаков
    scaler = StandartScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    from SGDExperiments import SGDExperiment
    experiment = SGDExperiment(X_train_scaled, X_test_scaled, y_train, y_test)

    # experiment.run_sgd_regressor()
    # experiment.run_sgd_ridge()

    experiment.run_tau_experiment()

