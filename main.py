import numpy as np
import math
import sympy as sy
import matplotlib.pyplot as plt
import random


def function(x1):
    return 1 / x1


def function1(x2):
    return x2 ** 2


a = -2
b = 2
n = 3  # 20
arr = [[a, function1(a)]]
div = (b - a) / (n - 1)
for i in range(n - 2):
    arr.append([a + div * (i + 1), function1(a + div * (i + 1))])
arr.append([b, function1(b)])
arr = np.array(arr)
print(arr)

# Ньютона назад

# Генерація таблиці зворотних різниць

# Створення масиву numpy розміром n & n x n та ініціалізація
# до нуля для збереження значень x і y разом із різницями y
x = np.zeros(n)  # Функція zeros() повертає новий масив заданої форми і типу, заповненого нулями.
y = np.zeros((n, n))


# функція для обчислення коефіцієнта Y
def p_cal(p, n):
    temp = p
    for i in range(1, n):
        if (i % 2 == 1):
            temp * (p - i)
        else:
            temp * (p + i)
    return temp


def fact(n):  # факторіал
    f = 1
    for i in range(2, n + 1):
        f *= i
    return f


print('Enter data for x and y:\n')
for i in range(n):  # вводимо дані для точок
    x[i] = arr[i][0]
    y[i][0] = arr[i][1]

for i in range(1, n):  # Створення таблиці зворотних різниць
    for j in range(n - 1, i - 2, -1):
        y[j][i] = y[j][i - 1] - y[j - 1][i - 1]

print('\ntable of differences back\n')

for i in range(0, n):  # виведення таблиці
    print('%0.2f' % (x[i]), end='')  # виведення колонки першої точки
    for j in range(0, i + 1):
        print('\t%0.2f' % (y[i][j]), end='')  # виведення колонки 2 точки і її продовження
    print()

e = random.uniform(a, b)
# we calculate the e of the function at the point
sum = y[n - 1][0]
u = (e - x[n - 1]) / (x[1] - x[0])
for i in range(1, n):
    sum = sum + (p_cal(u, i) * y[n - 1][i]) / fact(i)

print("\nvalues with precision up to e(", e, "): is", sum)

# Гауса вперед

print('input Enter data of x:\n')
x = [0 for i in range(n)]
for i in range(n):  # вводимо першу точку відрізка
    x[i] = arr[i][0]

y = [[0 for i in range(n)]
     for j in range(n)]
print('\nEnter data of y:\n')
for i in range(n):  # вводимо другу точку відрізка
    y[i][0] = arr[i][1]

for i in range(1, n):  # Створення трикутника Гаусса
    for j in range(
            n - i):  # numpy.round_() — це математична функція, яка округлює масив до заданої кількості десяткових знаків.
        y[j][i] = np.round((y[j + 1][i - 1] - y[j][i - 1]), 4)

print('\n')
for i in range(n):  # Друк трикутника
    print('%0.2f' % x[i], end="")
    for j in range(n - i):
        print('\t%0.2f' % y[i][j], end="\t")
    print()

# we calculate the value of the function at the point
sum1 = y[int(n / 2)][0]
p = (e - x[int(n / 2)]) / (x[1] - x[0])
for i in range(1, n):
    sum1 = sum1 + (p_cal(p, i) * y[int((n - i) / 2)][i]) / fact(i)

print("\nvalues with precision up to e(", e, "): is", round(sum1, n - 1))

# other ex

'''


print("Polynomial 1 =", polynomial)  # поліном
polynomial2 = sm.integrate(function1(x_), (x_, 0, formula))
print("Polynomial 2 =", polynomial2)
print("D = ", abs(function(x) - polynomial))  # похибка
print("D = ", abs(function1(x) - polynomial2))

# graph
fig, axe = plt.subplots()
x_gr = np.arange(a, b, e)
s = function1(x_gr)
axe.plot(x_gr, s)
t1 = random.uniform(0.35, 1)
axe.plot([function1(x), sum], [t1, t1])
plt.title('graph: ')
plt.show()


arr = [[a, function1(a)]]
div = (b - a) / (n - 1)
for i in range(n - 2):
    arr.append([a + div * (i + 1), function1(a + div * (i + 1))])
arr.append([b, function1(b)])
arr = np.array(arr)
x = float(input('input x from 1 to 3: '))
if x > 3 or x < 1:
    x = random.uniform(a, b)
    print('x: ', x)
h = arr[1][0] - arr[0][0]
formula = (x - arr[-1][0]) / h
#print(formula)
x_ = sm.Symbol('x_')
polynomial = sm.integrate(function1(x_), (x_, 0, formula))

def pol(x3):
    x3 = sm.Symbol('x3')
    return sm.integrate(function1(x3))


a = -10
b = 10
t = np.arange(a, b, e)
fig, ax = plt.subplots()
ax.plot(t, (t))
ax.grid(True)
plt.show()




for a in sy.preorder_traversal(Ln):
        if isinstance(a, sy.Float):
            ex = ex.subs(a, round(a, 4))
            
            
x = random.uniform(a, b)
h = arr[1][0] - arr[0][0]
formula = (x - arr[-1][0]) / h
# print(formula)
x_ = sm.Symbol('x_')
#polynomial = sm.integrate(function(x_), (x_, 0, formula))



#polynomial = sm.integrate(function(x_), (x_, 0, formula))
#print("Polynomial 1 =", polynomial)  # поліном
polynomial2 = Polinom(arr)
print("Polynomial 2 =", sm.integrate(function1(Polinom(arr)), (t, 0, formula))
#print("D = ", abs(function(x) - polynomial))  # похибка
print("D = ", abs(function1(x) - polynomial2))
'''


def Polinom(data):
    n = len(data) - 1
    t = sy.Symbol('t')
    arr = []

    for i in range(n + 1):
        arr.append([])

    for i in range(n + 1):
        arr[0].append(data[i][1])

    for i in range(1, n + 1):
        for j in range(n - i + 1):
            arr[i].append(arr[i - 1][j + 1] - arr[i - 1][j])
    Ln = "+"
    for i in range(n):
        exp = "1"
        for j in range(i + 1):
            exp += "*(t-{0})".format(j)
        exp += "*{0}".format(arr[i + 1][-1]) + "/{0}".format(math.factorial(i + 1))
        exp = sy.expand(exp)
        Ln = Ln + "(" + str(sy.collect(exp, t)) + ")+"
        print()
    Ln = sy.sympify(Ln[0:-1]) / float("{0}".format(arr[1][-1])) / float("{0}".format(arr[1][-1]))
    return Ln


e = random.uniform(a, b)
print('x =', e, end='')
h = arr[1][0] - arr[0][0]
formula = (e - arr[-1][0]) / h
print(formula)
p = Polinom(arr)
print(p)
Lnx = eval(str(p).replace("t", str(formula)))
print("D =", abs(function1(e) - Lnx))
print('_____')
sy.plotting.plot(p)
plt.show()
plt.close()

# polynomial = sm.integrate(function(x_), (x_, 0, formula))
# print("Polynomial 1 =", polynomial)  # поліном
# polynomial2 = Polinom(arr)
# print("Polynomial 2 =", sm.integrate(function1(Polinom(arr)), (t, 0, formula))
# print("D = ", abs(function(x) - polynomial))  # похибка
# print("D = ", abs(function1(x) - polynomial2))
# polynomial = sm.integrate(function(x_), (x_, 0, formula))
# print("Polynomial 1 =", polynomial)  # поліном
