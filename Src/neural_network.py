import numpy as np

# From TP5 - Machine Learning Course PolyAI A20

def homogeneous(X):
    return np.insert(X, 0, values=1, axis=-1)

# Default Layer Class
class Layer:
    def __init__(self, input_size=None, output_size=None):
        self.input_size = input_size
        self.output_size = output_size
        self.prev = None
        self.next = None
    def __call__(self, callable_graph):
        if isinstance(callable_graph, Layer):
            self.prev = callable_graph
            self.input_size = callable_graph.output_size
            callable_graph.next = self
            if self.output_size is None:
                self.output_size = self.input_size

            return self
        return self.predict(callable_graph)
    
    def predict(self, X):
        return self.forward(X) if self.next is None else self.next.predict(self.forward(X))
        
    def backprop(self, Y):
        return self.backward(Y) if self.prev is None else self.prev.backprop(self.backward(Y))
        
    def str_chain(self):
        return str(self) if self.next is None else str(self) + " -> " + self.next.str_chain()

# Fully Connected Layer Class
class Dense(Layer):
    def __init__(self, units, input_size=None):
        Layer.__init__(self, input_size, units)
        self.W = None
        
    def init_weights(self):
        self.W = np.random.normal(0, scale=2/(self.input_size), size=(self.output_size, self.input_size + 1))
        
    # Forward pass using X
    def forward(self, X):
        if self.W is None:
            self.init_weights()

        Xh = homogeneous(X) # Homogeneous coordinates
        self.last_input = Xh # Save inputs for backpropagation

        #Compute forward pass result
        return Xh.dot(self.W.T)
    
    # Backward pass using Y
    def backward(self, Y):
        #Compute gradient
        input_3d = np.swapaxes(np.expand_dims(self.last_input, axis=0), 0, 1)
        y_t_3d = np.transpose(np.swapaxes(np.expand_dims(Y, axis=0), 0, 1), (0, 2, 1))
        self.grad = np.mean(y_t_3d * input_3d, axis=0)
        
        #Compute backward pass result, don't forget to remove the bias term! Bias does not need to be backpropagated.
        back_error = Y.dot(self.W)

        return np.delete(back_error, 0, axis=-1)
        
    def __str__(self):
        return "Dense(" + str(self.input_size) + ", " + str(self.output_size) + ")"

class ReLU(Layer):
    def forward(self, X):
        self.last_input = X # Save inputs for backpropagation
        return np.maximum(X, 0)

    def backward(self, Y):
        return Y * np.maximum(np.sign(self.last_input), 0)

    def __str__(self):
        return "ReLU(" + str(self.input_size) + ")"

class TanH(Layer):
    def forward(self, X):
        self.last_input = X # Save inputs for backpropagation
        ep = np.exp(X)
        em = np.exp(-X)
        return (ep - em) / (ep + em)

    def backward(self, Y):
        Xf = self.forward(self.last_input)
        return Y * (1 - Xf ** 2)

    def __str__(self):
        return "TanH(" + str(self.input_size) + ")"

class Logistic(Layer):
    def forward(self, X):
        self.last_input = X # Save inputs for backpropagation
        return 1 / (1 + np.exp(-X))

    def backward(self, Y):
        Xf = self.forward(self.last_input)
        return Y * Xf * (1 - Xf)

    def __str__(self):
        return "Logistic(" + str(self.input_size) + ")"
    
def build_mlp():
    inputs = Dense(64, input_size=3072)
    x = inputs
    x = ReLU()(x)
    x = Dense(64)(x)
    x = ReLU()(x)
    x = Dense(20)(x)
    outputs = x
    return outputs