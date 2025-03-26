import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class BoltzmannMachine:
    def __init__(self, visible_units, hidden_units):
        """
        Initialize a Boltzmann Machine with full connectivity
        
        Args:
            visible_units: Number of visible units
            hidden_units: Number of hidden units
        """
        self.visible_units = visible_units
        self.hidden_units = hidden_units
        self.total_units = visible_units + hidden_units
        
        # Initialize symmetric weights (including visible-visible and hidden-hidden connections)
        self.weights = tf.Variable(
            tf.random.normal([self.total_units, self.total_units], stddev=0.1),
            trainable=True
        )
        
        # Make sure self-connections are zero
        self.weights.assign(tf.linalg.set_diag(
            self.weights,
            tf.zeros(self.total_units)
        ))
        
        # Initialize biases
        self.biases = tf.Variable(tf.zeros([self.total_units]), trainable=True)
        
        # Temperature parameters
        self.initial_temp = 10.0
        self.final_temp = 0.1
        
    def sigmoid_with_temp(self, x, temperature):
        """Sigmoid activation with temperature parameter"""
        return tf.nn.sigmoid(x / temperature)
    
    def sample_bernoulli(self, probs):
        """Sample from Bernoulli distribution given probabilities"""
        return tf.nn.relu(tf.sign(probs - tf.random.uniform(tf.shape(probs))))
    
    def energy(self, state):
        """Compute energy of a state"""
        return -tf.reduce_sum(tf.matmul(state, self.weights) * state - tf.reduce_sum(self.biases * state))
    
    def run_to_equilibrium(self, initial_state, steps, temperature):
        """
        Run the network to thermal equilibrium using Gibbs sampling
        
        Args:
            initial_state: Initial state of all units
            steps: Number of Gibbs sampling steps
            temperature: Current temperature
            
        Returns:
            Final state after running to equilibrium
        """
        state = tf.identity(initial_state)
        
        for _ in range(steps):
            # Random order update
            for i in tf.random.shuffle(tf.range(self.total_units)):
                # Compute activation
                activation = tf.reduce_sum(self.weights[i] * state) + self.biases[i]
                prob = self.sigmoid_with_temp(activation, temperature)
                state = tf.tensor_scatter_nd_update(
                    state, 
                    [[i]], 
                    [self.sample_bernoulli(prob)]
                )
        
        return state
    
    def train_step(self, visible_data, learning_rate, k, current_temp):
        """
        Perform one training step using contrastive divergence
        
        Args:
            visible_data: Batch of visible data (shape: [batch_size, visible_units])
            learning_rate: Learning rate
            k: Number of Gibbs sampling steps in negative phase
            current_temp: Current temperature
            
        Returns:
            Reconstruction error
        """
        batch_size = visible_data.shape[0]
        
        # Positive phase (clamped)
        # Initialize hidden units randomly
        hidden_init = tf.random.uniform([batch_size, self.hidden_units], 0, 1)
        hidden_init = self.sample_bernoulli(hidden_init)
        
        # Ensure both tensors are of the same type
        visible_data = tf.cast(visible_data, dtype=tf.float32)
        hidden_init = tf.cast(hidden_init, dtype=tf.float32)
        
        # Combine visible and hidden
        positive_state = tf.concat([visible_data, hidden_init], axis=1)
        
        # Run to equilibrium with visible units clamped
        clamped_state = tf.concat([
            visible_data,  # Visible units clamped
            positive_state[:, self.visible_units:]  # Hidden units free
        ], axis=1)
        
        positive_state = self.run_to_equilibrium(clamped_state, 10, current_temp)
        
        # Negative phase (free-running)
        negative_state = self.run_to_equilibrium(positive_state, k, current_temp)
        
        # Compute statistics
        positive_stats = tf.einsum('bi,bj->ij', positive_state, positive_state) / batch_size
        negative_stats = tf.einsum('bi,bj->ij', negative_state, negative_state) / batch_size
        
        # Update weights (symmetric updates)
        delta_weights = learning_rate * (positive_stats - negative_stats)
        
        # Zero out diagonal (no self-connections)
        delta_weights = tf.linalg.set_diag(delta_weights, tf.zeros(self.total_units))
        
        # Symmetric update (upper and lower triangles)
        self.weights.assign_add((delta_weights + tf.transpose(delta_weights)) / 2)
        
        # Update biases
        delta_biases = learning_rate * tf.reduce_mean(positive_state - negative_state, axis=0)
        self.biases.assign_add(delta_biases)
        
        # Compute reconstruction error
        reconstruction_error = tf.reduce_mean(
            tf.square(visible_data - negative_state[:, :self.visible_units])
        )
        
        return reconstruction_error
    
    def fit(self, data, epochs=5, batch_size=32, k=1, learning_rate=0.01):
        """
        Train the Boltzmann Machine
        
        Args:
            data: Training data (shape: [n_samples, visible_units])
            epochs: Number of training epochs
            batch_size: Batch size
            k: Number of Gibbs steps in negative phase
            learning_rate: Learning rate
        """
        n_samples = data.shape[0]
        errors = []
        
        dataset = tf.data.Dataset.from_tensor_slices(data)
        dataset = dataset.shuffle(n_samples).batch(batch_size)
        
        for epoch in range(epochs):
            # Anneal temperature
            current_temp = max(
                self.initial_temp * (1 - epoch/epochs), 
                self.final_temp
            )
            
            epoch_error = 0
            for batch in dataset:
                error = self.train_step(batch, learning_rate, k, current_temp)
                epoch_error += error
            
            avg_error = epoch_error / (n_samples / batch_size)
            errors.append(avg_error)
            
            print(f"Epoch {epoch+1}/{epochs}, Temp: {current_temp:.2f}, Error: {avg_error:.4f}")
        
        plt.plot(errors)
        plt.title("Training Error")
        plt.xlabel("Epoch")
        plt.ylabel("Reconstruction Error")
        plt.show()
    
    def reconstruct(self, visible_data, steps=100, temperature=1.0):
        """
        Reconstruct visible data by running the network
        
        Args:
            visible_data: Input data (shape: [batch_size, visible_units])
            steps: Number of Gibbs sampling steps
            temperature: Temperature for sampling
            
        Returns:
            Reconstructed visible data
        """
        batch_size = visible_data.shape[0]
        
        # Initialize hidden units randomly
        hidden_init = tf.random.uniform([batch_size, self.hidden_units], 0, 1)
        hidden_init = self.sample_bernoulli(hidden_init)
        
        # Combine visible and hidden
        state = tf.concat([visible_data, hidden_init], axis=1)
        
        # Run to equilibrium
        state = self.run_to_equilibrium(state, steps, temperature)
        
        return state[:, :self.visible_units]

# Load and preprocess data (same as original example)
movies_df = pd.read_csv('./ml-1m/movies.dat', sep='::', header=None, engine='python', encoding='latin-1')
ratings_df = pd.read_csv('./ml-1m/ratings.dat', sep='::', header=None, engine='python', encoding='latin-1')

movies_df.columns = ['MovieID', 'Title', 'Genres']
ratings_df.columns = ['UserID', 'MovieID', 'Rating', 'Timestamp']

# Create user-rating matrix
user_rating_df = ratings_df.pivot(index='UserID', columns='MovieID', values='Rating')
norm_user_rating_df = user_rating_df.fillna(0) / 5.0
trX = norm_user_rating_df.values

# Initialize and train Boltzmann Machine
visible_units = trX.shape[1]
hidden_units = 20  # Can be adjusted

bm = BoltzmannMachine(visible_units, hidden_units)
bm.fit(trX, epochs=50, batch_size=100, k=5, learning_rate=0.001)

# Make recommendations for a user
mock_user_id = 114
input_user = trX[mock_user_id-1].reshape(1, -1)
reconstructed = bm.reconstruct(input_user, steps=100, temperature=0.5)

# Create recommendations dataframe
scored_movies = movies_df[movies_df['MovieID'].isin(user_rating_df.columns)]
scored_movies = scored_movies.assign(RecommendationScore=reconstructed[0])
recommendations = scored_movies.sort_values("RecommendationScore", ascending=False)

# Filter out movies already rated by user
user_movies = ratings_df[ratings_df['UserID'] == mock_user_id]['MovieID']
new_recommendations = recommendations[~recommendations['MovieID'].isin(user_movies)]

print("Top 10 Recommendations:")
print(new_recommendations[['MovieID', 'Title', 'RecommendationScore']].head(10))