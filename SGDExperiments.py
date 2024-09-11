import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import SGDRegressor
from sklearn.metrics import r2_score
from StochasticGD import SGDRidgeModel

class SGDExperiment:

    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def run_tau_experiment(self, tau_values):
        """
        Проводит серию экспериментов с моделью SGDRidgeModel для различных значений гиперпараметра tau
        Строит графики сходимости функции потерь для каждого значения tau, а также зависимость коэффициента детерминации R^2 от tau
        """
        Qi_scores = []  # Список для хранения значений функции потерь (Q_i) для каждого tau
        r2_scores = []  # Список для хранения значений R^2 для обучающей и тестовой выборок

        for tau in tau_values:
            print(f"Tau Test = {tau}")
            sgd_ridge = self.run_sgd_ridge(tau=tau)  # Обучаем модель SGDRidgeModel для текущего tau

            Qi_scores.append((tau, sgd_ridge.q_values))  # Сохраняем tau и его значения функции потерь
            
            # Рассчитываем R^2 для обучающей и тестовой выборок
            r2_train = sgd_ridge.r2_score(self.X_train, self.y_train)
            r2_test = sgd_ridge.r2_score(self.X_test, self.y_test)
            r2_scores.append([r2_train, r2_test])  # Сохраняем R^2

        # Настройка графиков
        num_plots = len(Qi_scores)
        num_cols = 3  
        num_rows = (num_plots + num_cols - 1) // num_cols

        fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 4))

        for i, (plot, ax) in enumerate(zip(Qi_scores, axs.flat)):
            tau, q_values = plot
            ax.plot(q_values)
            ax.set_xlabel('Эпоха')
            ax.set_ylabel('Функция потерь (Q_i)')
            ax.set_title(f'График сходимости\nTau: {tau}')
        
        for j in range(i + 1, num_rows * num_cols):
            fig.delaxes(axs.flat[j])

        plt.tight_layout()
        plt.show()

        plt.plot(r2_scores)
        plt.xlabel('1/10^i')
        plt.ylabel('R2')
        plt.title('Зависимость R2 от Tau')
        plt.show()

    def run_sgd_regressor(self):
        """
        Обучает и оценивает модель SGDRegressor из библиотеки sklearn.
        Выводит значения коэффициента детерминации R^2 для обучающей и тестовой выборок.
        """
        print("-----------------------------------------------------------------")
        print("Running SGDRegressor:")
        
        sgd = SGDRegressor(random_state=42)
        sgd.fit(self.X_train, self.y_train)
        
        r2_train_sgd = r2_score(self.y_train, sgd.predict(self.X_train))
        r2_test_sgd = r2_score(self.y_test, sgd.predict(self.X_test))
        
        print(f"R^2 на обучающей выборке для SGDRegressor: {r2_train_sgd}")
        print(f"R^2 на тестовой выборке для SGDRegressor: {r2_test_sgd}")
        print("-----------------------------------------------------------------")
        
        return sgd

    def run_sgd_ridge(self, tau=0.001):
        """
        Обучает и оценивает модель SGDRidgeModel с заданным значением tau (гиперпараметра регуляризации).
        Выводит значения коэффициента детерминации R^2 для обучающей и тестовой выборок.
        """
        print("-----------------------------------------------------------------")
        print("Running SGDRidgeModel:")
        
        sgd = SGDRidgeModel(tau=tau)
        sgd.fit(self.X_train, self.y_train)
        
        r2_train = sgd.r2_score(self.X_train, self.y_train)
        r2_test = sgd.r2_score(self.X_test, self.y_test)
        
        print(f"R^2 на обучающей выборке для SGDRidge: {r2_train}")
        print(f"R^2 на тестовой выборке для SGDRidge: {r2_test}")
        print("-----------------------------------------------------------------")
        
        return sgd
