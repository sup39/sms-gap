'''
MIT License

Copyright (c) 2022 sup39[サポミク]

Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.
'''

import numpy as np
import torch
import os
device = os.environ.get('SMS_PYTORCH_DEVICE', 'cpu') # TODO

# cast value to float32/float64
f32 = lambda x: (x.to(torch.float32) if type(x) == torch.Tensor else torch.tensor(x, dtype=torch.float32)).to(device)
f64 = lambda x: (x.to(torch.float64) if type(x) == torch.Tensor else torch.tensor(x, dtype=torch.float64)).to(device)
# index of float
fidx = lambda x: np.frombuffer(np.array((x.cpu() if type(x)==torch.Tensor else x), 'f'), 'i')[0]
fidxs = lambda x: np.frombuffer(np.array((x.cpu() if type(x)==torch.Tensor else x), 'f'), 'i')
# prev/next float32
inf = f32(torch.inf)
minf = f32(-torch.inf)
pf32 = lambda x: torch.nextafter(f32(x), minf)
nf32 = lambda x: torch.nextafter(f32(x), inf)
# utils
def binsearch(arr, q, **kwargs):
  arr = arr.contiguous()
  return torch.searchsorted(arr, q, **kwargs) if arr[0]<=arr[-1] else torch.searchsorted(-arr, -q, **kwargs)

# test range
def frange(a, b):
  a, b = map(fidx, (a, b))
  assert np.sign(a)*np.sign(b)==1, '0 must NOT contained in the interval'
  if a > b: a, b = b, a
  args = (a, b+1) if a>0 else (b, a-1, -1)
  return f32(np.frombuffer(np.arange(*args, dtype='i').tobytes(), 'f'))

# utils for inverse function
def floor32(x):
  x32 = f32(x)
  return torch.where(x32<=x, x32, pf32(x32))
def ceil32(x):
  x32 = f32(x)
  return torch.where(x32>=x, x32, nf32(x32))

def rf32lt(B):
  B1 = f32(B)
  B2 = torch.where(B>B1, B1, pf32(B1))
  B2m = (f64(B2)+nf32(B2))/2
  B2eq = f32(B2m)==B2
  return B2m, B2eq
def rf32ltle(B, eq):
  B1 = f32(B)
  B2 = torch.where(torch.where(eq, B>=B1, B>B1), B1, pf32(B1))
  B2m = (f64(B2)+nf32(B2))/2
  B2eq = f32(B2m)==B2
  return B2m, B2eq

def rf32gt(C):
  C1 = f32(C)
  C2 = torch.where(C<C1, C1, nf32(C1))
  C2m = (f64(C2)+pf32(C2))/2
  C2eq = f32(C2m)==C2
  return C2m, C2eq
def rf32gtge(C, eq):
  C1 = f32(C)
  C2 = torch.where(torch.where(eq, C<=C1, C<C1), C1, nf32(C1))
  C2m = (f64(C2)+pf32(C2))/2
  C2eq = f32(C2m)==C2
  return C2m, C2eq

# inverse function
def find_z(x, boundary):
  x = f32(x)
  x0, z0, x1, z1 = f32(boundary).reshape(-1)
  ## guard x0 <= x <= x1
  if x0 > x1: x0, z0, x1, z1 = x1, z1, x0, z0
  # prepare
  dx0 = x0-x
  dx1 = x1-x
  D = f64(-1-2**-24)
  Rx = x1-x0
  Rz = z1-z0
  # inverse
  A0 = (D+f64(dx0*Rz))/Rx
  A1 = (-D+f64(dx1*Rz))/Rx
  B2m, B2eq = rf32lt(A0)
  C2m, C2eq = rf32gt(A1)
  # finalize
  B4 = z0-B2m
  C4 = z1-C2m
  B4f = ceil32(B4)
  C4f = floor32(C4)
  B4r = torch.where(~B2eq & (B4==B4f), nf32(B4f), B4f)
  C4r = torch.where(~C2eq & (C4==C4f), pf32(C4f), C4f)
  # done
  return B4r, C4r

def find_x(z, boundary):
  z = f32(z)
  x0, z0, x1, z1 = f32(boundary).reshape(-1)
  ## guard z0 <= z <= z1
  if z0 > z1: x0, z0, x1, z1 = x1, z1, x0, z0
  # prepare
  dz0 = z0-z
  dz1 = z1-z
  D = f64(-1-2**-24)
  Rx = x1-x0
  Rz = z1-z0
  # inverse 1
  A0 = f64(dz0)*f64(Rx)-D
  A1 = f64(dz1)*f64(Rx)+D
  B2m, B2eq = rf32gt(A0)
  C2m, C2eq = rf32lt(A1)
  # inverse 2
  A0 = B2m/Rz
  A1 = C2m/Rz
  B2m, B2eq = rf32gtge(A0, B2eq)
  C2m, C2eq = rf32ltle(A1, C2eq)
  # finalize
  B4 = x0-B2m
  C4 = x1-C2m
  B4f = floor32(B4)
  C4f = ceil32(C4)
  B4r = torch.where(~B2eq & (B4==B4f), pf32(B4f), B4f)
  C4r = torch.where(~C2eq & (C4==C4f), nf32(C4f), C4f)
  # done
  return C4r, B4r

def find_neigh(dir, xC, r, bd, max_count=1e8):
  assert dir in ['x', 'z'], "dir(方向)は'x'か'z'でなければなりません"
  r = abs(r)
  qx0, qx1 = map(f32, (xC-r, xC+r))
  assert torch.sign(qx0)*torch.sign(qx1)==1, '検索範囲に1を含んではいけません'
  estcnt = int(2**23 * f32([qx0, qx1]).abs().log2().diff().abs().item())
  assert estcnt<=max_count, '推定の探索数(%.3E)が制限(%.3E)を超えました。探索範囲を小さくするか、max_countを大きくしてください'%(estcnt, max_count)
  x = frange(qx0, qx1)
  zL, zU = (find_z if dir=='x' else find_x)(x, bd)
  return torch.column_stack([x, zL, zU]).contiguous(), zL<=zU

def find_all(bd):
  # guard x0 <= x1
  x0, z0, x1, z1 = f32(bd).reshape(-1)
  if x0 > x1: x0, z0, x1, z1 = x1, z1, x0, z0
  # make x, z input
  if x0 == x1: # x軸に平行
    x = f32([x0])
    z = f32([])
  elif z0 == z1: # z軸に平行
    x = f32([])
    z = f32([z0])
  else:
    x = f32([])
    z = f32([])
    # https://twitter.com/sup39x1207/status/1534935579678605312
    # TODO weighted by execution time
    m = (z1-z0)/(x1-x0)
    b = z0-m*x0
    xx = -b/2/m
    zz = m*xx+b
    # split into 2 segments
    assert torch.sign(x0)*torch.sign(x1)==1 or xx != 0, '原点を通る境界の探索にはまだ対応していません'
    if xx >= 0:
      # P0 [z] P* [x] P1
      if xx <= x1: x = frange(xx.clamp(min=x0), x1)
      if z0 <= zz: z = frange(z0, zz.clamp(max=z1))
    else:
      # P0 [x] P* [z] P1
      if x0 <= xx: x = frange(x0, xx.clamp(max=x1))
      if zz <= z1: z = frange(zz.clamp(min=z0), z1)
  # find all
  zL, zU = find_z(x, bd)
  xL, xU = find_x(z, bd)
  return tuple(
    torch.column_stack(a).contiguous()
    for a in ((x, zL, zU), (z, xL, xU))
  )

def find_in_result(dir, x, xzzV, zxxV):
  assert dir in ['x', 'z'], "dir(方向)は'x'か'z'でなければなりません"
  x = f32(x).reshape(-1)
  if dir == 'z': xzzV, zxxV = zxxV, xzzV # swap x, z if z
  # ans1 := solution in xzz
  ix = binsearch(xzzV[:, 0], x, side='left').clamp(max=xzzV.shape[0]-1)
  ans1 = xzzV[ix]
  ans1 = ans1[ans1[:,0]==x] # filter out wrong result
  # ans2 := solution in xzz
  ix0 = binsearch(zxxV[:, 2], x, side='left')
  ix1 = binsearch(zxxV[:, 1], x, side='right')-1
  ixValid = ix0<=ix1
  ans2 = torch.column_stack([
    x[ixValid],
    zxxV[ix0[ixValid], 0],
    zxxV[ix1[ixValid], 0],
  ])
  # combine ans1 ans ans2
  return torch.vstack(sorted(
    (ans1, ans2),
    key=lambda a: 0 if a.shape[0]==0 else a[0,0]
  ))

def verify(x, z, boundary):
  x, z = map(f32, (x, z))
  x0, z0, x1, z1 = f32(boundary).reshape(-1)
  # prepare
  Rx = x1-x0
  Rz = z1-z0
  dx0 = x0-x
  dx1 = x1-x
  dz0 = z0-z
  dz1 = z1-z
  # discriminant
  r0, r1 = f32(f64(dz0)*f64(Rx)-f64(dx0*Rz)), f32(-f64(dz1)*f64(Rx)+f64(dx1*Rz))
  return (r0<-1) & (r1<-1)
