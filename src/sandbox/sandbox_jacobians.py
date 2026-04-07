import opensim as osim
import casadi as ca
import numpy as np

class PositionCallback(ca.Callback):
    def __init__(self, name, model, body, opts={}):
        ca.Callback.__init__(self)
        self.model = model
        self.state = self.model.initSystem()
        self.matter = self.model.getMatterSubsystem()
        self.body = model.getBodySet().get(body)
        self.mobod_index = self.body.getMobilizedBodyIndex()

        self.construct(name, opts)

    def get_n_in(self): return 1
    def get_n_out(self): return 1
    def get_sparsity_in(self,i):
        return ca.Sparsity.dense(self.state.getNQ(),1)
    def get_sparsity_out(self,i):
        return ca.Sparsity.dense(3,1)

    def eval(self, arg):
        self.state.setQ(osim.Vector.createFromMat(np.squeeze(arg[0].full())))
        self.model.realizePosition(self.state)
        position = self.body.getPositionInGround(self.state).to_numpy()
        return [position]

class PositionCallback_Jac(PositionCallback):
  def has_jacobian(self): return True
  def get_jacobian(self,name,inames,onames,opts):
    class JacFun(ca.Callback):
      def __init__(self, model, matter, state, mobod_index, opts={}):
        ca.Callback.__init__(self)
        self.model = model
        self.matter = matter
        self.state = state
        self.mobod_index = mobod_index
        self.construct(name, opts)

      def get_n_in(self): return 2
      def get_n_out(self): return 1

      def get_sparsity_in(self,i):
        if i==0: # nominal input
          return ca.Sparsity.dense(self.state.getNQ(),1)
        elif i==1: # nominal output
          return ca.Sparsity(3,1)

      def get_sparsity_out(self,i):
        return ca.Sparsity.dense(3, self.state.getNQ())

      # Evaluate numerically
      def eval(self, arg):

        self.state.setQ(osim.Vector.createFromMat(np.squeeze(arg[0].full())))
        self.model.realizePosition(self.state)

        matrix = osim.Matrix()
        self.matter.calcStationJacobian(self.state, self.mobod_index, osim.Vec3(0),
                                        matrix)
        return [matrix.to_numpy()]

    # You are required to keep a reference alive to the returned Callback object
    self.jac_callback = JacFun(self.model, self.matter, self.state, self.mobod_index)
    return self.jac_callback


model = osim.Model('unscaled_generic.osim')
state = model.initSystem()

# Use the function
f = PositionCallback('f', model, 'pelvis')
f_jac = PositionCallback_Jac('f_jac', model, 'pelvis')

# Use the function.
q = state.getQ().to_numpy()
res = f(q)
print(res)


# You may call the Callback symbolically
x = ca.MX.sym("x", state.getNQ())
print(f(x))

# Manual finite-differences
J = np.zeros((3, state.getNQ()))
eps = 1e-5
for i in range(state.getNQ()):
    x_plus = q.copy()
    x_plus[i] += eps
    J[:,i] = np.squeeze((f(x_plus)-f(q))/eps)
print(J)

# CasADi finite-differences
f = PositionCallback('f', model, 'pelvis', {"enable_fd":True})
J_fd = ca.Function('J',[x],[ca.jacobian(f(x),x)])
print(J_fd(2))

# Analytic
f_analytic = PositionCallback_Jac('f_analytic', model, 'pelvis')
J_analytic = ca.Function('J',[x],[ca.jacobian(f_analytic(x),x)])
print(J_analytic(2))

import pdb; pdb.set_trace()