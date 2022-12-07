# (i)
import numpy as np
def delta(x):
    delta = len(x)* sum(x**2) - (sum(x)**2)
    return delta
def m(x, y):
     # x and y should be same size thus x is size of both (N_x == N_y)
    d = delta(x) # calculate delta
    slope = (1/d)*(len(x) * sum(x*y) - sum(x)*sum(y))
    return slope

# (ii)
def b(x, y):
    d = delta(x) # calculate delta
    intercept = (1/d)*(sum(x**2)*sum(y) - sum(x)*sum(x*y))
    return intercept

# (iii)
def sigma_y(x, y):
    slope = m(x,y)
    intercept = b(x,y)
    temp = sum((y - slope * x - intercept)**2)
    s_y = np.sqrt((1/(len(x)-2))*temp)
    return s_y
# (iv)
def sigma_m(x,y):
    d = delta(x)
    s_y = sigma_y(x, y)
    s_m = s_y * np.sqrt(len(x)/d)
    return s_m
# (v)
def sigma_b(x, y):
    d = delta(x)
    s_y = sigma_y(x,y)
    s_b = s_y * np.sqrt(sum(x**2)/d)
    return s_b

# (viii)

def w(y_err): # helper function
    weights = (1/(y_err)**2)
    return weights
def delta_wtd(x, y_err): # helper function
    weights = w(y_err)
    d_w = sum(weights) * sum(weights * x**2) - (sum(weights*x))**2
    return d_w
def m_wtd(x,y,y_err):
    weights = w(y_err) 
    d_W = delta_wtd(x, y_err)
    temp = sum(weights) * sum(weights*x*y) - sum(weights*x) * sum(weights*y)
    m = (1/d_W) * temp
    return m

def b_wtd(x,y,y_err):
    weights = w(y_err) 
    d_W = delta_wtd(x, y_err)
    temp = sum(weights * x**2) * sum(weights * y) - sum(weights * x) * sum(weights*x*y)
    b = (1/d_W) * temp
    return b

def sigma_m_wtd(x,y_err):
    weights = w(y_err)
    d_W = delta_wtd(x, y_err)
    s_m_wtd = np.sqrt((sum(weights))/d_W)
    return s_m_wtd

def sigma_b_wtd(x,y_err):
    weights = w(y_err)
    d_W = delta_wtd(x, y_err)
    s_b_wtd = np.sqrt((sum(weights*x**2))/d_W)
    return s_b_wtd
