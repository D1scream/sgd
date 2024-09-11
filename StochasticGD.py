import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split


from Standart_Scaler import StandartScaler


class SGDRidgeModel:
    
    def __init__(self, epochs=1000, tau=0.001, eps=1e-3):
        """
        Параметры:
        epochs: Количество эпох, сколько раз модель пройдет по данным.
        tau: Коэффициент регуляризации (L2). Контролирует силу регуляризации.
        eps: Критерий сходимости. Если изменение функции потерь меньше этого значения, обучение останавливается.
        
        Атрибуты:
        weights (np.ndarray): Веса модели, инициализируются нулями.
        bias (float): Смещение (свободный член), инициализируется нулем.
        q_values (list): Список значений функции потерь на каждой эпохе.
        """
        self.epochs = epochs
        self.tau = tau
        self.eps = eps
        self.weights = None
        self.bias = 0
        self.q_values = []  # Для хранения значений функции потерь

    def fit(self, X, y):
        """
        Обучение модели с использованием стохастического градиентного спуска.
        Минимизируется функция потерь
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)  # Инициализация весов нулями
        #self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0
        self.q_values = []  # Обнуляем значения функции потерь

        # Основной цикл обучения
        for epoch in range(1, self.epochs + 1):
            # Перемешивание данных
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # Стохастическое обновление весов для каждого объекта
            for i in range(n_samples):
                X_i = X_shuffled[i]
                y_i = y_shuffled[i]
                learning_rate = 1.0 / (i + epoch * n_samples + 1)  # Уменьшающийся темп обучения
                dot_product = np.dot(self.weights, X_i)

                # Обновление весов с учетом регуляризации (ридж-регрессия)
                self.weights = self.weights * (1 - learning_rate * self.tau) - learning_rate * (dot_product - y_i) * X_i
                self.bias -= learning_rate * (dot_product + self.bias - y_i)  # Обновление смещения
            
            # Вычисление функции потерь на каждой эпохе
            q_i = self.compute_ridge_loss(X, y)
            self.q_values.append(q_i)
            
            # Критерий сходимости: если разница в значениях функции потерь меньше eps, обучение останавливается
            if epoch > 1 and abs(self.q_values[-1] - self.q_values[-2]) < self.eps:
                print(f"Сходимость достигнута на эпохе {epoch}")
                break

    def compute_ridge_loss(self, X, y):
        """
        Вычисление функции потерь для Ridge-регрессии
        
        Q(w, b) = (1/n) * sum((w^T * x_i + b - y_i)^2) + (tau / 2) * sum(w_j^2)
        где первая часть — среднеквадратичная ошибка (MSE), вторая часть — регуляризация (L2-норма)
        
        Возвращает:
        Значение функции потерь
        """
        
        predictions = self.predict(X)  # Прогнозируемые значения
        errors = predictions - y  # Ошибки модели
        mse_loss = np.mean(errors ** 2)  # Среднеквадратичная ошибка (MSE)
        regularization = (self.tau / 2) * np.sum(self.weights ** 2)  # Регуляризация (L2-норма)
        return mse_loss + regularization
                
    def predict(self, X):
        """
        Предсказание значений целевой переменной для новых данных
        Прогнозируется с помощью линейной модели: y_pred = w^T * X + b
        
        Возвращает:
        Вектор предсказанных значений
        """
        return np.dot(X, self.weights) + self.bias
    
    def r2_score(self, X, y):
        """
        Вычисление коэффициента детерминации (R^2)
        Коэффициент детерминации R^2:
        R^2 = 1 - (RSS / TSS)
        где RSS — сумма квадратов ошибок (Residual Sum of Squares),
        TSS — полная сумма квадратов (Total Sum of Squares)
        
        Возвращает:
        Значение R^2, которое показывает долю дисперсии, объясненную моделью
        """
        y_pred = self.predict(X)  # Предсказанные значения
        rss = np.sum((y - y_pred) ** 2)  # Остаточная сумма квадратов (RSS)
        tss = np.sum((y - np.mean(y)) ** 2)  # Полная сумма квадратов (TSS)
        return 1 - (rss / tss)  # Вычисление R^2


if __name__ == "__main__":
    
    data = fetch_california_housing()
    X, y = data.data, data.target
    print(data.feature_names)
    # Список признаков
    features = ['MedInc',      # Median income (Медианный доход) в районе
                'HouseAge',    # Средний возраст домов в районе
                'AveRooms',    # Среднее количество комнат на дом
                'AveBedrms',   # Среднее количество спален на дом
                'Population',  # Численность населения в районе
                'AveOccup',    # Среднее количество жильцов на дом
                'Latitude',    # Широта
                'Longitude']   # Долгота
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandartScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    from SGDExperiments import SGDExperiment
    experiment = SGDExperiment(X_train_scaled, X_test_scaled, y_train, y_test)

    tau_values = [1 / (2 ** i) for i in range(0, 9)]
    print(tau_values)
    experiment.run_tau_experiment(tau_values)