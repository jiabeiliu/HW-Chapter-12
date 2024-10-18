import numpy as np

class SoftmaxActivation:
    def __init__(self):
        pass

    def forward(self, inputs):
        """
        Perform the forward pass through the softmax activation function.
        :param inputs: Raw output scores (logits) from the previous layer, shape (batch_size, num_classes)
        :return: Softmax probabilities, shape (batch_size, num_classes)
        """
        # Shift inputs by subtracting the max to avoid large exponents (numerical stability)
        shifted_inputs = inputs - np.max(inputs, axis=1, keepdims=True)
        
        # Exponentiate the shifted inputs
        exp_values = np.exp(shifted_inputs)
        
        # Normalize by dividing by the sum of exponentiated values
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        
        return probabilities

    def backward(self, dvalues, original_outputs):
        """
        Perform the backward pass through the softmax activation function.
        :param dvalues: Gradient of the loss function with respect to the output of softmax
        :param original_outputs: Output of softmax during the forward pass
        :return: Gradient of the loss function with respect to the input of softmax
        """
        # Initialize the gradient array
        dinputs = np.empty_like(dvalues)
        
        # Loop over each sample in the batch
        for index, (single_output, single_dvalue) in enumerate(zip(original_outputs, dvalues)):
            # Flatten the output array for easy computation
            single_output = single_output.reshape(-1, 1)
            
            # Compute the Jacobian matrix of the softmax function
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            
            # Calculate the gradient of the loss with respect to the inputs of softmax
            dinputs[index] = np.dot(jacobian_matrix, single_dvalue)
        
        return dinputs

# Example usage:
if __name__ == "__main__":
    softmax = SoftmaxActivation()
    
    # Example input (logits from a previous layer)
    inputs = np.array([[2.0, 1.0, 0.1], [1.0, 3.0, 0.5]])
    
    # Forward pass (getting softmax probabilities)
    probabilities = softmax.forward(inputs)
    print("Softmax Probabilities:\n", probabilities)
    
    # Example gradient from loss function during backpropagation
    dvalues = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    
    # Backward pass (computing gradients with respect to inputs)
    gradients = softmax.backward(dvalues, probabilities)
    print("Gradient wrt Inputs:\n", gradients)
