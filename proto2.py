import random
import numpy as np
from deap import base, tools, creator, algorithms
from random import choice
import matplotlib.pyplot as plt
import time

LenOfPopulation = 50
SizeOfIndivid = 4
P_CROSS = 0.9
P_MUTATE = 0.0



def printIndividual(schedule):
    day = 0
    for i in schedule:
        print(days[day])
        for j in i:
            print(j)
        print()
        day += 1
        day %= len(days)


importantSubjects = {
    "Русский язык": 3,
    "Русская литература": 4,
    "Родной язык": 1,
    "Родная литература": 1,
    "Математика": 4,  # 0
    "Окружающий мир": 2,  # 1
    "Инностранный язык": 1
}

nonImportantSubjects = {
    "Основы религиозных культур и светской этики": 1,
    "Искусство": 1,
    "Технология": 1,
    "Физра": 2
}


days = {
    0: "Понедельник",
    1: "Вторник",
    2: "Среда",
    3: "Четверг",
    4: "Пятница"
}

allSubjects = importantSubjects.copy()
allSubjects.update(**nonImportantSubjects)
creator.create("Fitness", base.Fitness, weights=(1.0, ))
creator.create("Individual", list, fitness=creator.Fitness)


def generateIndividual():
    buffer = allSubjects.copy()
    schedule = creator.Individual()
    while len(schedule) != 5:
        schedule.append([])
        while len(schedule[-1]) != 4:
            rand = choice([i for i in buffer if buffer[i] > 0])
            if rand == "Физра" and len(schedule[-1]) <= 1:
                continue
            schedule[-1].append(rand)
            buffer[rand] -= 1
    return schedule


def generatePopulation():
    population = [generateIndividual() for i in range(LenOfPopulation)]
    return population


def oneChromoFitness(individual):
    quality = [1, 1]
    delta = 1 / sum(allSubjects.values())
    delta1 = 0.5
    for day in individual:
        for i in range(len(day) - 1):
            if importantSubjects.get(day[i], 0) and importantSubjects.get(day[i + 1], 0):
                quality[0] -= delta
            elif day.count(day[i]) >= 2:
                quality[1] = 0
    return sum(quality),


def passMutate(individual, indpb=0.1):
    return individual,

# Два класса для коеффициентов к приспособленности и для индивида


def crossingover(ind1, ind2):
    return ind1, ind2


toolbox = base.Toolbox()
toolbox.register("CreatePopulation", tools.initRepeat, list, generateIndividual)
toolbox.register("evaluate", oneChromoFitness)


# Genetic operations with individuals
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", crossingover)
toolbox.register("mutate", passMutate, indpb=1/SizeOfIndivid)

stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("max", np.max)
stats.register("avg", np.mean)

begin = time.time()

first_population = generatePopulation()
last_population, logbook = algorithms.eaSimple(first_population,
                                               toolbox,
                                               cxpb=P_CROSS,
                                               mutpb=P_MUTATE,
                                               ngen=20000,
                                               verbose=False)
print(oneChromoFitness(last_population[0]))
printIndividual(last_population[0])
print(time.time() - begin)
maxFitnessValues, meanFitnessValues = logbook.select("max", "avg")

plt.plot(maxFitnessValues, color='red')
plt.plot(meanFitnessValues, color='green')
plt.xlabel('Поколение')
plt.ylabel('Макс/средняя приспособленность')
plt.title('Зависимость максимальной и средней приспособленности от поколения')
# plt.show()