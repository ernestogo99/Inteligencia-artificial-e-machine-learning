import numpy as np


class BaseMLP:
    def __init__(self, hidden_layer_sizes=(10,), learning_rate=0.01,
                 max_iter=200, activation="relu", momentum=0.9,
                 batch_size=32, solver="sgd",
                 grad_clip=1e3, tol=1e-4, verbose=False, weight_decay=0.0):

        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate = float(learning_rate)
        self.max_iter = int(max_iter)
        self.activation_name = activation
        self.momentum = float(momentum)
        self.batch_size = int(batch_size) if batch_size is not None else None
        self.solver = solver
        self.grad_clip = grad_clip
        self.tol = float(tol)
        self.verbose = verbose
        self.weight_decay = float(weight_decay) 

        self.loss_curve_ = []


    def _sigmoid(self, x):
        x = np.array(x, dtype=float)
        out = np.empty_like(x)
        pos = x >= 0
        out[pos] = 1 / (1 + np.exp(-x[pos]))
        expx = np.exp(x[~pos])
        out[~pos] = expx / (1 + expx)
        return out

    def activation(self, x):
        if self.activation_name == "relu":
            return np.maximum(0.0, x)
        if self.activation_name == "tanh":
            return np.tanh(x)
        return self._sigmoid(x)

    def activation_derivative(self, z):
        if self.activation_name == "relu":
            return (z > 0).astype(float)
        if self.activation_name == "tanh":
            t = np.tanh(z)
            return 1 - t * t
        s = self._sigmoid(z)
        return s * (1 - s)


    def init_weights(self, input_dim, output_dim):
        layer_sizes = [input_dim] + list(self.hidden_layer_sizes) + [output_dim]
        self.weights = []
        self.biases = []
        self.vel_W = []
        self.vel_b = []

        for i in range(len(layer_sizes) - 1):
            in_size = layer_sizes[i]
            out_size = layer_sizes[i+1]
            limit = np.sqrt(6 / (in_size + out_size))
            W = np.random.uniform(-limit, limit, (in_size, out_size))
            b = np.zeros((1, out_size))
            self.weights.append(W)
            self.biases.append(b)
            self.vel_W.append(np.zeros_like(W))
            self.vel_b.append(np.zeros_like(b))


    def forward(self, X):
        a = X
        self.z_list = []
        self.a_list = [a]

        for W, b in zip(self.weights[:-1], self.biases[:-1]):
            z = a @ W + b
            a = self.activation(z)
            self.z_list.append(z)
            self.a_list.append(a)
        

        z = a @ self.weights[-1] + self.biases[-1]
        self.z_list.append(z)
        self.a_list.append(z)
        return z



class MLPRegressor(BaseMLP):
    def fit(self, X, y):
        X = np.array(X, float)
        y = np.array(y, float).reshape(-1, 1)

        N = X.shape[0]
        batch = self.batch_size or N

        self.init_weights(X.shape[1], 1)

        prev_loss = np.inf

        for epoch in range(self.max_iter):

         
            perm = np.random.permutation(N)
            Xs, ys = X[perm], y[perm]

            epoch_loss = 0.0
            total_batches = 0
            total_grad_norm = 0.0    

            for i in range(0, N, batch):
                Xb = Xs[i:i+batch]
                yb = ys[i:i+batch]
                bsz = Xb.shape[0]

              
                pred = self.forward(Xb)

             
                loss = np.mean((yb - pred)**2)
                epoch_loss += loss
                total_batches += 1

           
                dZ = (pred - yb) * (2.0 / max(1, bsz))

              
                for j in reversed(range(len(self.weights))):
                    a_prev = self.a_list[j]

                    dW = a_prev.T @ dZ
                    db = np.sum(dZ, axis=0, keepdims=True)
                    if self.weight_decay > 0:
                        dW += self.weight_decay * self.weights[j]
                    grad_norm = np.linalg.norm(dW)
                    total_grad_norm += grad_norm

                    if self.grad_clip:
                        dW = np.clip(dW, -self.grad_clip, self.grad_clip)
                        db = np.clip(db, -self.grad_clip, self.grad_clip)

              
                    self.vel_W[j] = self.momentum * self.vel_W[j] - self.learning_rate * dW
                    self.vel_b[j] = self.momentum * self.vel_b[j] - self.learning_rate * db

                    self.weights[j] += self.vel_W[j]
                    self.biases[j] += self.vel_b[j]

                    if j > 0:
                        dZ = (dZ @ self.weights[j].T) * self.activation_derivative(self.z_list[j-1])

            avg_loss = epoch_loss / max(1, total_batches)
            self.loss_curve_.append(avg_loss)

    
            if abs(prev_loss - avg_loss) < self.tol:
                if self.verbose:
                    print(f"[early stopping] epoch {epoch}, Δloss={abs(prev_loss-avg_loss)} < tol={self.tol}")
                break

            prev_loss = avg_loss


            if total_grad_norm < self.tol:
                if self.verbose:
                    print(f"[early stopping] epoch {epoch}, grad_norm={total_grad_norm} < tol={self.tol}")
                break

      
            if not np.isfinite(avg_loss):
                if self.verbose:
                    print(f"[stop] unstable loss {avg_loss}")
                break

        return self

    def predict(self, X):
        return self.forward(np.array(X, float)).ravel()


class MLPClassifier(BaseMLP):
    def fit(self, X, y):
        X = np.array(X, float)
        y = np.array(y, float).reshape(-1, 1)

        N = X.shape[0]
        batch = self.batch_size or N

        self.init_weights(X.shape[1], 1)

        prev_loss = np.inf

        for epoch in range(self.max_iter):

            perm = np.random.permutation(N)
            Xs, ys = X[perm], y[perm]

            epoch_loss = 0.0
            total_batches = 0
            total_grad_norm = 0.0

            for i in range(0, N, batch):
                Xb = Xs[i:i+batch]
                yb = ys[i:i+batch]
                bsz = Xb.shape[0]

                logits = self.forward(Xb)
                probs = self._sigmoid(logits)

                eps = 1e-12
                loss = -np.mean(yb*np.log(probs+eps) + (1-yb)*np.log(1-probs+eps))
                epoch_loss += loss
                total_batches += 1

                dZ = (probs - yb) / max(1, bsz)

                for j in reversed(range(len(self.weights))):
                    a_prev = self.a_list[j]

                    dW = a_prev.T @ dZ
                    db = np.sum(dZ, axis=0, keepdims=True)
                    if self.weight_decay > 0:
                        dW += self.weight_decay * self.weights[j]
                    total_grad_norm += np.linalg.norm(dW)

              
                    if self.grad_clip:
                        dW = np.clip(dW, -self.grad_clip, self.grad_clip)
                        db = np.clip(db, -self.grad_clip, self.grad_clip)

              
                    self.vel_W[j] = self.momentum * self.vel_W[j] - self.learning_rate * dW
                    self.vel_b[j] = self.momentum * self.vel_b[j] - self.learning_rate * db
                    self.weights[j] += self.vel_W[j]
                    self.biases[j] += self.vel_b[j]

                    if j > 0:
                        dZ = (dZ @ self.weights[j].T) * self.activation_derivative(self.z_list[j-1])

            avg_loss = epoch_loss / max(1, total_batches)
            self.loss_curve_.append(avg_loss)

            if abs(prev_loss - avg_loss) < self.tol:
                if self.verbose:
                    print("[early stop]", epoch)
                break

            prev_loss = avg_loss

            if total_grad_norm < self.tol:
                if self.verbose:
                    print("[early stop grad]", epoch)
                break

            if not np.isfinite(avg_loss):
                break

        return self

    def predict(self, X):
        logits = self.forward(np.array(X, float))
        probs = self._sigmoid(logits)
        return (probs >= 0.5).astype(int).ravel()
