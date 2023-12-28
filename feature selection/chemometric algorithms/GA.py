from deap import base
from deap import creator
from deap import tools
import pandas as pd
import numpy as np
import random
from sklearn import model_selection
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc, roc_curve

creator.create('FitnessMax', base.Fitness, weights=(1.0,))  # for minimization, set weights as (-1.0,)
creator.create('Individual', list, fitness=creator.FitnessMax)

def GA(X, y, number_of_generation=10):

    scaled_x_train = X
    scaled_y_train = y
    toolbox = base.Toolbox()
    min_boundary = np.zeros(X.shape[1])
    max_boundary = np.ones(X.shape[1]) * 1.0
    # 基础参数
    probability_of_crossover = 0.5
    probability_of_mutation = 0.2
    threshold_of_variable_selection = 0.5

    def create_ind_uniform(min_boundary, max_boundary):
        index = []
        for min, max in zip(min_boundary, max_boundary):
            index.append(random.uniform(min, max))
        return index

    # individual 个体
    # population 种群
    toolbox.register('create_ind', create_ind_uniform, min_boundary, max_boundary)
    toolbox.register('individual', tools.initIterate, creator.Individual, toolbox.create_ind)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)

    def evalOneMax(individual):
        individual_array = np.array(individual)
        selected_x_variable_numbers = np.where(individual_array > threshold_of_variable_selection)[0]
        selected_scaled_x_train = scaled_x_train[:, selected_x_variable_numbers]
        max_number_of_components = 10
        if len(selected_x_variable_numbers):
            # 交叉验证
            pls_components = np.arange(1, min(np.linalg.matrix_rank(selected_scaled_x_train) + 1,
                                              max_number_of_components + 1), 1)
            tprs = []
            r2_cv_all = []
            for pls_component in pls_components:
                # PLS-DA判别分析中转换y
                scaled_y_train = pd.get_dummies(y)
                model_in_cv = PLSRegression(n_components=pls_component)
                y_predict = model_selection.cross_val_predict(model_in_cv, selected_scaled_x_train, scaled_y_train, cv=10)
                # 将预测结果即类别矩阵转换为数值标签
                y_predict = np.array([np.argmax(i) for i in y_predict])


                mean_fpr = np.linspace(0, 1, 100)
                fpr, tpr, thresholds = roc_curve(y, y_predict)
                tprs.append(np.interp(mean_fpr, fpr, tpr))
                tprs[-1][0] = 0.0
                roc_auc = auc(fpr, tpr)
                print('roc_auc', roc_auc)
                r2_cv_all.append(roc_auc)

                # accuracy = accuracy_score(y, y_predict)
                # r2_cv_all.append(accuracy)
            # 精确度越高则个体适应度越大
            value = [np.mean(r2_cv_all)]
            print('value', value)
        return value

    toolbox.register('evaluate', evalOneMax)
    # 加入交叉变换
    toolbox.register('mate', tools.cxTwoPoint)
    # 设置突变几率
    toolbox.register('mutate', tools.mutFlipBit, indpb=0.05)
    # 挑选个体
    toolbox.register('select', tools.selTournament, tournsize=3)
    # 种群
    random.seed()
    pop = toolbox.population(n=len(y))

    fitness = []
    for generation in range(number_of_generation):
        print('-- Generation {0} --'.format(generation + 1))

        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < probability_of_crossover:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < probability_of_mutation:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        # 选出来的个体(描述符)
        print('  Evaluated %i individuals' % len(invalid_ind))

        pop[:] = offspring
        fits = [ind.fitness.values[0] for ind in pop]
        length = len(pop)
        mean = sum(fits) / length
        fitness.append(mean)
        print('fitness', fitness)

        print('generation', generation+1)
        print('mean = sum(fits) / len(pop)', mean)
        print('  Min %s' % min(fits))
        print('  Max %s' % max(fits))

    # 准备数据
    data_df = pd.DataFrame(fitness)  # 关键1，将ndarray格式转换为DataFrame
    # 更改表的索引
    data_df.columns = ['A']  # 将第一行的0,1,2,...,9变成A,B,C,...,J
    # 将文件写入excel表格中
    writer = pd.ExcelWriter('方法GA_fitness.xlsx')  # 关键2，创建名称为hhh的excel表格
    data_df.to_excel(writer, 'page_1', float_format='%.5f')  # 关键3，float_format 控制精度，将data_df写到hhh表格的第一页中。若多个文件，可以在page_2中写入
    writer.save()  # 关键4

    best_individual = tools.selBest(pop, 1)[0]
    best_individual_array = np.array(best_individual)
    selected_x_variable_numbers = np.where(best_individual_array > threshold_of_variable_selection)[0]
    print("GA的筛选变量索引：", selected_x_variable_numbers)

    return selected_x_variable_numbers

if __name__ == '__main__':
    pass