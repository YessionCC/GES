import re

last_iter = 72000

pat = re.compile(r'\[ITER (\S+)\] Evaluating test: PSNR (\S+), SSIM (\S+), LPIPS (\S+) .*')
with open('nohup.out') as f:
    cc = f.read()

res = pat.findall(cc)

last_ps = []
last_ss = []
last_lps = []
last_pn = []
best_ps = []
best_ss = []
best_lps = []
best_iters = []

score = 0
cb_lps = 0
cb_ss = 0
cb_ps = 0
cb_iter = 0

for rt in res:
    if int(rt[0]) < 30_000: continue
    p = float(rt[1])
    s = float(rt[2])
    l = float(rt[3])
    sc = p + 10*(1-l)
    if sc > score:
        score = sc
        cb_lps = rt[3]
        cb_ss = rt[2]
        cb_ps = rt[1]
        cb_iter = rt[0]
    if rt[0] == str(last_iter):
        best_lps.append(cb_lps)
        best_ss.append(cb_ss)
        best_ps.append(cb_ps)
        best_iters.append(cb_iter)
        last_lps.append(rt[3])
        last_ss.append(rt[2])
        last_ps.append(rt[1])
        #last_pn.append(rt[4])
        score = 0
    

print(' '.join(best_ps))
print(' '.join(best_ss))
print(' '.join(best_lps))
print(' '.join(best_iters))

print(' '.join(last_ps))
print(' '.join(last_ss))
print(' '.join(last_lps))
print(' '.join(last_pn))