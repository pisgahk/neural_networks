import math
import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return 3 * x**2 - 4 * x + 5


print(f(3.0))

xs = np.arange(-5, 5, 0.25)
ys = f(xs)

plt.plot(xs, ys)
plt.show()


# ---[Calculating the slope]--------------------------------------------------------------------

h = 0.00000001
x = 2 / 3

print((f(x + h) - f(x)) / h)

# ---[Testing differentiation with other values.]-----------------------------------------------

a = 2.0
b = -3.0
c = 10.0

y = a * b + c
print(y)

h = 0.000000001
c += h
y1 = a * b + c

grad = (y1 - y) / h
f"Slope is: {grad}"


# ---[Creating the Value class, I do not know why but I am yet to find out.]--------------------


class Value:
    def __init__(self, data, _children=(), _op="", label="") -> None:
        self.data = data
        self.grad = 0.0

        # For implementing the backpropagation automatically.
        self._backward = lambda: None

        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self) -> str:
        return f"Value(date={self.data})"

    def __add__(self, other):

        # solving the `a + 1` problem
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():  # Defining a closure to determine the gradient.
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other):  # other * self
        # solves the problem of `2 * a` just incase `a * 2` is the exact same thing humanly
        return self * other

    def __truediv__(self, other):  # a/b
        return self * other**1

    def __neg__(self):  # -self
        return self * -1

    def __sub__(self, other):  # a -b
        other = other if isinstance(other, Value) else Value(other)
        return self + (-other)

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers"
        out = Value(self.data**other, (self,), f"**{other}")

        def _backward():
            self.grad += (other * (self.data ** (other - 1))) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value(t, (self,), "tanh")

        def _backward():
            # Gradient here has to be multiplied because of the chain rule.
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()

        # Using topological sort, you better read your DSA.
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()


a = Value(2.0, label="a")
b = Value(-3.0, label="b")
c = Value(10.0, label="c")

# a, b
# a + b
e = b * c
e.label = "e"

d = a + e
d.label = "d"

d

g = Value(-2.0, label="g")
L = g * d
L.label = "L"
L._prev
L._op
L.backward()

# ---[Visualising the Value data structure]-------------------------------------------------------

from graphviz import Digraph


def trace(root):
    # Builds a set of all nodes and edges in a graph
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges


def draw_dot(root):
    dot = Digraph(format="svg", graph_attr={"rankdir": "LR", "splines": "false"})  # LR = Left to Right.

    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))

        # Create a rectangular `record` node for each value
        dot.node(
            name=uid,
            label="{ %s | data: %s | grad %.4f }" % (n.label, n.data, n.grad),
            shape="record",
        )

        if n._op:
            # Create an op node if this value is the result of an operation
            dot.node(name=uid + n._op, label=n._op)
            # Connect op node to value node
            dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        if n2._op:  # Only connect if n2 has an operation
            dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    # return dot
    return dot.render("graph_output", format="svg", view=True)


draw_dot(L)

# ---[From calculating the gradient below, these are the gradients.]----------------------------------

L.grad = 1
d.grad = -2
g.grad = -28
a.grad = -2
e.grad = -2
b.grad = -20
c.grad = 6

# ---[let us try lowering the value of L]-------------------------------------------------------------
# Essentially we are trying to lower the value of L.data, it gives a better prediction when the value is at minimum/zero.
# Not sure though, i think that it is L.grad that we should be reducing instead of L.data
a.data -= 0.01 * a.grad
b.data -= 0.01 * b.grad
c.data -= 0.01 * c.grad
g.data -= 0.01 * g.grad


e = b * c
d = e + a
g = Value(-2.0)
L = g * d
L = L.data
L


# ---[Manually Backpropagation]------------------------------------------------------------------------
def lol():

    h = 0.0000001

    a = Value(2.0, label="a")
    b = Value(-3.0, label="b")
    c = Value(10.0, label="c")
    e = b * c
    e.label = "e"
    d = e + a
    d.label = "d"
    g = Value(-2.0, label="g")
    L = g * d
    L1 = L.data

    a = Value(2.0, label="a")
    b = Value(-3.0, label="b")
    b.data += h
    c = Value(10.0, label="c")
    e = b * c
    e.label = "e"
    d = e + a
    d.label = "d"
    g = Value(-2.0, label="g")
    L = g * d
    L2 = L.data

    print((L2 - L1) / h)


lol()


# ---[Trying with a replica of a neural network]------------------------------------------------------
import math
import numpy as np
import matplotlib.pyplot as plt

plt.plot(np.arange(-5, 5, 0.2), np.tanh(np.arange(-5, 5, 0.2)))
plt.grid()
# plt.show()


# Inputs
x1 = Value(2.0, label="x1")
x2 = Value(0.0, label="x2")
# Weights
w1 = Value(-3.0, label="w1")
w2 = Value(1.0, label="w2")
# Bias of the neuron.
b = Value(6.8813735870195432, label="b")

x1w1 = x1 * w1
x1w1.label = "x1w1"

x2w2 = x2 * w2
x2w2.label = "x2w2"

x1w1x2w2 = x1w1 + x2w2
x1w1x2w2.label = "x1w1x2w2"

n = x1w1x2w2 + b
n.label = "n"

o = n.tanh()
o.label = "o"

o.grad = 1.0
o.backward()
# n._backward()
# x1w1x2w2._backward()
# x1w1._backward()
# x2w2._backward()

draw_dot(o)


# NOTE: WRITE TEST CASES WHEN:
# A VARIABLE IS ADDED TO ITSELF i.e `a + a`
# A VARIABLE IS USED MORE THAN ONCE.
# ADDING A `Value` TO AN INTEGER i.e `a + 1`


# ---[Writing in Pytorch]------------------------------------------------------------------------

import random
import torch

x1 = torch.Tensor([2.0]).double()
x1.requires_grad = True
x2 = torch.Tensor([0.0]).double()
x2.requires_grad = True

w1 = torch.Tensor([-3.0]).double()
w1.requires_grad = True
w2 = torch.Tensor([1.0]).double()
w2.requires_grad = True

b = torch.Tensor([6.8813735870195432]).double()
b.requires_grad = True

n = w1 * x1 + w2 * x2 + b
o = torch.tanh(n)

print(o.data.item())
o.backward()

print("---------------------------")
print(f"x2: {x2.grad}")
print(f"w2: {w2.grad}")
print(f"x1: {x1.grad}")
print(f"w1: {w1.grad}")


# ---[Building the neural network]---------------------------------------------------------------------
import random


class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        # w*x + b
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out

    # According to what i see on the digraph, even the input has gradients which should not happen since  no matter
    # what the input is it should not have a gradient(that influences how it affects the network)
    # Hence creation of this `parameters` method that some how looks for all the parameters in the network
    # and sees what effect they have on the overall network.

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    # def __call__(self, x):
    #     outs = [n(x) for n in self.neurons]
    #     return outs[0] if len(outs) == 1 else outs

    def __call__(self, x) -> Value | list[Value]:
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
        # params = []
        # for neuron in self.neurons:
        #     ps = neuron.parameters()
        #     params.extend(ps)
        # return params


class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(nouts))]

    # def __call__(self, x):
    #     for layer in self.layers:
    #         x = layer(x)
    #     return x

    def __call__(self, x) -> Value:
        for layer in self.layers:
            x = layer(x)
        return x  # type: ignore

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


x = [2.0, 3.0, -1.0]
n = MLP(3, [4, 4, 1])  # 2 dimensional neuron since there are two input '2.0', '3.0'
n(x)

# draw_dot(n(x))

n.parameters()  # These are all the parameters (weights and biases) in the network.
len(
    n.parameters()
)  # This network has 41 parameters in total(According to the current output)

# ---[Testing the MLP]------------------------------------------------------------------------------

xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
ys = [1.0, -1.0, -1.0, 1.0]  # desired outputs.
ypred = [n(x) for x in xs]
ypred

# for i in ys:
#     ys = Value(i)


# ---[Loss Calculation]-----------------------------------------------------------------------------
loss = sum(((yout - ygt) ** 2 for ygt, yout in zip(ys, ypred)), Value(0.0))
loss
loss.backward()
n.layers[0].neurons[0].w[0].grad
draw_dot(loss)

# ---[Gradient Descend]----------------------------------------------------------------------------------------
for p in n.parameters():
    p.data += -0.01 + p.grad

# After calculating the descend, we need to forward pass the network and check the loss to see if it lowering.
# Low loss means that out predictions are matching the targets.

ypred = [n(x) for x in xs]  # Forward pass.

loss = sum(((yout - ygt) ** 2 for ygt, yout in zip(ys, ypred)), Value(0.0))
loss
ypred

# ---[More organized way of calculating the `Loss` and `Gradient descent`.]---------------------------------------

for k in range(1000):

    # Forward Pass.
    ypred = [n(x) for x in xs]
    loss = sum(((yout - ygt) ** 2 for ygt, yout in zip(ys, ypred)), Value(0.0))

    # Backward Pass.
    for p in n.parameters():  # We had to zero grad before backpropagation
        p.grad = 0.0
    loss.backward()

    # Update
    for p in n.parameters():
        p.data += -0.1 * p.grad

    print(k, loss.data)
ypred

# ---[testing]------------------------------------------------------------------------

# this is the loss of the each of the last four neurons in the last layer
[(yout - ygt) ** 2 for ygt, yout in zip(ys, ypred)]
