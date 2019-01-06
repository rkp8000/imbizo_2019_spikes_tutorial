import numpy as np


DT = 0.001


def sgmd(x):
    return 1 / (1 + np.exp(-x))

# nonlin
def calc_g(s_0, s_1):
    r = np.sqrt(s_0**2 + s_1**2)
    return .1 * sgmd((r - .004)/.0005)

# filters
t_h = np.arange(0, 0.3, DT)

h_0 = np.sin(2*np.pi*t_h/.1)
m = np.exp(-(t_h-.08)/.02)
m[m > 1] = 1
h_0 *= m
h_0 /= np.sqrt(np.sum(h_0**2))

h_1 = np.sin(2*np.pi*(t_h-.025)/.1)
m = np.exp(-(t_h-.09)/.02)
m[m > 1] = 1
h_1 *= m
h_1 /= np.sqrt(np.sum(h_1**2))


def gen_spks(t, s):
    s_0 = np.nan * np.zeros(t.shape)
    s_1 = np.nan * np.zeros(t.shape)

    p_spk = np.nan * np.zeros(t.shape)
    fr = np.nan * np.zeros(t.shape)

    for t_ctr in range(len(t_h), len(t)):
        s_ = s[t_ctr-len(t_h):t_ctr]

        s_0_ = np.dot(h_0[::-1], s_) * DT
        s_1_ = np.dot(h_1[::-1], s_) * DT

        p_spk[t_ctr] = calc_g(s_0_, s_1_)
        fr[t_ctr] = p_spk[t_ctr] / DT

        s_0[t_ctr] = s_0_
        s_1[t_ctr] = s_1_
        
    # gen spks
    spks = np.random.rand(len(t)) < p_spk
    
    return spks

