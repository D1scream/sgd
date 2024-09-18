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

    def run_tau_experiment(self):
        """
        Проводит серию экспериментов с моделью SGDRidgeModel для различных значений гиперпараметра tau.
        Строит графики сходимости функции потерь для каждого значения tau и зависимости коэффициента детерминации R^2 от tau.
        """
        tau_values = np.logspace(-5, 0, 10)

        Qi_scores = []
        r2_scores = []
        best_tau = None
        best_r2_test = -np.inf

        for tau in tau_values:
            print(f"Tau = {tau:.8f}")
            sgd_ridge = self.run_sgd_ridge(tau=tau) 
            
            Qi_scores.append((tau, sgd_ridge.q_values))  # Сохраняем tau и его значения функции потерь
            
            # Рассчитываем R^2 для обучающей и тестовой выборок
            r2_train = sgd_ridge.r2(self.X_train, self.y_train)
            r2_test = sgd_ridge.r2(self.X_test, self.y_test)
            r2_scores.append([r2_train, r2_test])
            
            if r2_test > best_r2_test:
                best_r2_test = r2_test
                best_tau = tau

        # Настройка графиков функции потерь
        fig, ax = plt.subplots(figsize=(10, 6))
        best_q_values = None
        
        for i, (tau, q_values) in enumerate(Qi_scores):
            r2_train, r2_test = r2_scores[i]
            label = f'Tau: {tau:.8f}, r2 (train): {r2_train:.4f}, r2 (test): {r2_test:.4f}'
            
            if tau == best_tau:
                ax.plot(q_values, label=f'Лучший Tau: {best_tau:.8f}, r2 (train): {r2_train:.4f}, r2 (test): {r2_test:.4f}', color='blue', linewidth=2)
            else:
                ax.plot(q_values, label=label, alpha=0.5)

        ax.set_xlabel('i')
        ax.set_ylabel('Функция потерь (Q_i)')
        ax.set_title('Графики сходимости для разных Tau и R²')
        ax.legend(loc='best')
        plt.show()

        # График зависимости R² от Tau
        taus = [tau for tau, _ in Qi_scores]
        r2_train_scores, r2_test_scores = zip(*r2_scores)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(taus, r2_train_scores, marker='o', label='R² (train)')
        ax.plot(taus, r2_test_scores, marker='o', label='R² (test)')
        
        ax.set_xlabel('Tau')
        ax.set_ylabel('R²')
        ax.set_title('Зависимость R² от Tau')
        ax.set_xscale('log')
        ax.legend()
        plt.show()
    def run_sgd_sclearn(self,tau=0.001):
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

        print("-----------------------------------------------------------------")
        print("Running SGDRidgeModel:")
        
        sgd = SGDRidgeModel(tau=tau)
        sgd.fit(self.X_train, self.y_train)
        
        r2_train = sgd.r2(self.X_train, self.y_train)
        r2_test = sgd.r2(self.X_test, self.y_test)
        
        print(f"R^2 на обучающей выборке для SGDRidge: {r2_train}")
        print(f"R^2 на тестовой выборке для SGDRidge: {r2_test}")
        print("-----------------------------------------------------------------")
        
        return sgd
