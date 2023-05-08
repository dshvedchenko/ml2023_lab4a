# TODO
# 1. read input data
# 2. define function_image
# 3. analyze how good is function on -1,-2,-3 points
# 4. convert to classes
# 5. account to RMSQ




import numpy as np

def xN(ix):
    def f(data):
        return data[ix+len(data)]
    return f

def mul(*funcs):
    def f(data):
        r = 1
        for f in funcs:
            r *= f(data)
        return r
    return f

def pow(func, d):
    def f(data):
        return func(data)**d
    return f

def one():
    def f(*a, **k):
        return 1
    return f

# input_data = np.array([10,15,13,19,14,18,17,11,12,14])
input_data = np.array([10,15,13,14,10,16,13,11,15,18])

F = [one(), xN(-1), xN(-2), mul(xN(-1),xN(-2)), pow(xN(-1),2), pow(xN(-2),2)]
deep_t = 2

# create whole linear system
Xh = []
for t in range(deep_t,input_data.shape[0]):
    res = [f(input_data[:t]) for f in F]
    Xh.append(res)

X = np.array(Xh)
# create real_pred_values
b = input_data[deep_t:]

# this is from lab data , to just remind
# X = np.array([[1,10,15,10*15],[1,15,13,15*13],[1,13,19,13*19],[1,19,14,19*14],[1,14,18,14*18],[1,18,17,17*18]])

# create normal linear system ( equations == num of unknowns alphas )
# c = X.shape[1]
c = len(F)
normX = []
normB = []
for sc in range(c):
    XC = X.copy()
    bC = b.copy()
    for r in range(X.shape[0]):
        XC[r,:] = XC[r,:] * X[r,sc]
        bC[r] = bC[r] * X[r,sc]
    normX.append(np.sum(XC,axis=0))
    normB.append(np.sum(bC))

# roots
alphas = np.linalg.solve(normX,normB)

rmsq = np.sqrt(np.sum((np.sum(alphas * X, axis=1) - b)**2)/b.shape[0])

inp_pred = [f([13,11,15]) for f in F]

pred = np.sum(alphas * inp_pred)

print(f"Predicted: {pred}, Real: 18")
print(f"RMSQ: {rmsq}")
print(np.sum(alphas * X, axis=1))
print(b)