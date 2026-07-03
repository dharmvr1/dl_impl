import numpy as np

def train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray, initial_bias: float, learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):
     
    # Your code 
    n=labels.shape[0]
    mse_values= []
    

    for _ in range(epochs):
        
        z = np.dot(features,initial_weights.T) + initial_bias
        sigmoid = 1/(1+ np.exp(-z))
        sigmoid_derivative = sigmoid * (1-sigmoid)

        loss = np.sum((labels-sigmoid)**2)/n
        mse_values.append(np.round(loss,4))
        error = (labels-sigmoid)


        gradient_weights = 2*(np.dot(np.dot(error,sigmoid_derivative),features))/n
        gradient_bias = 2*np.dot(error,sigmoid_derivative)/n
		
        initial_weights =initial_weights - learning_rate*gradient_weights
        initial_biass =initial_bias - learning_rate*gradient_bias

    updated_weights = initial_weights
    updated_bias = initial_bias


    return updated_weights, updated_bias, mse_values


print(train_neuron(np.array([[1.0, 2.0], [2.0, 1.0], [-1.0, -2.0]]), np.array([1, 0, 0]), np.array([0.1, -0.2]), 0.0, 0.1, 2))