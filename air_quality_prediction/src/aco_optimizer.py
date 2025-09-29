"""
Ant Colony Optimization (ACO) for Feature Selection in Air Quality Prediction

This module implements a multi-objective ACO algorithm that optimizes both:
1. Prediction accuracy (minimize RMSE)
2. Feature interpretability (prefer policy-controllable features)

Key Innovation: Bias toward policy-controllable features (PM2.5, NO2, SO2, etc.)
while de-emphasizing weather features (temperature, humidity, wind).
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ACO_FeatureSelection:
    """
    Ant Colony Optimization for Feature Selection with Policy Relevance Bias
    
    This class implements ACO with multi-objective optimization:
    - Primary objective: Minimize prediction error (RMSE)
    - Secondary objective: Maximize policy controllability of selected features
    """
    
    def __init__(self, 
                 n_ants: int = 20,
                 n_iterations: int = 50,
                 alpha: float = 1.0,  # Pheromone importance
                 beta: float = 2.0,   # Heuristic importance
                 rho: float = 0.1,    # Evaporation rate
                 q0: float = 0.9,     # Exploitation probability
                 policy_weight: float = 1.5,  # Weight for policy-controllable features
                 min_features: int = 3,
                 max_features: int = 15):
        """
        Initialize ACO parameters
        
        Args:
            n_ants: Number of ants (solutions) per iteration
            n_iterations: Maximum number of iterations
            alpha: Pheromone importance weight
            beta: Heuristic information weight
            rho: Pheromone evaporation rate
            q0: Exploitation probability (vs exploration)
            policy_weight: Weight multiplier for policy-controllable features
            min_features: Minimum number of features to select
            max_features: Maximum number of features to select
        """
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q0 = q0
        self.policy_weight = policy_weight
        self.min_features = min_features
        self.max_features = max_features
        
        # Policy-controllable features (higher priority)
        self.policy_controllable = [
            'PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'NO', 'NOx', 
            'NH3', 'Benzene', 'Toluene', 'Xylene'
        ]
        
        # Weather/uncontrollable features (lower priority)
        self.uncontrollable = [
            'Temperature', 'Humidity', 'Wind_Speed', 'Wind_Direction',
            'Pressure', 'Precipitation'
        ]
        
        # Initialize tracking variables
        self.best_solution = None
        self.best_fitness = float('inf')
        self.convergence_history = []
        self.feature_selection_history = []
        
    def _calculate_heuristic(self, features: List[str], target_correlation: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate heuristic information for each feature
        
        Args:
            features: List of available features
            target_correlation: Correlation of each feature with target
            
        Returns:
            Dictionary of heuristic values for each feature
        """
        heuristic = {}
        
        for feature in features:
            # Base heuristic from correlation with target
            base_heuristic = abs(target_correlation.get(feature, 0))
            
            # Apply policy relevance bias
            if feature in self.policy_controllable:
                policy_bias = self.policy_weight
            elif feature in self.uncontrollable:
                policy_bias = 1.0 / self.policy_weight  # Reduce priority
            else:
                policy_bias = 1.0  # Neutral
            
            # Combined heuristic
            heuristic[feature] = base_heuristic * policy_bias
            
        return heuristic
    
    def _initialize_pheromones(self, features: List[str]) -> Dict[str, float]:
        """Initialize pheromone trails for all features"""
        initial_pheromone = 1.0
        return {feature: initial_pheromone for feature in features}
    
    def _construct_solution(self, 
                          features: List[str], 
                          pheromones: Dict[str, float],
                          heuristics: Dict[str, float],
                          X_train: pd.DataFrame,
                          y_train: pd.Series) -> Tuple[List[str], float]:
        """
        Construct a solution (feature subset) using ACO
        
        Args:
            features: Available features
            pheromones: Current pheromone levels
            heuristics: Heuristic information
            X_train: Training features
            y_train: Training target
            
        Returns:
            Tuple of (selected_features, fitness_value)
        """
        selected_features = []
        available_features = features.copy()
        
        # Determine number of features to select (random between min and max)
        n_features = np.random.randint(self.min_features, min(self.max_features + 1, len(available_features)))
        
        for _ in range(n_features):
            if not available_features:
                break
                
            # Calculate selection probabilities
            probabilities = []
            for feature in available_features:
                pheromone = pheromones[feature]
                heuristic = heuristics[feature]
                
                # Probability calculation
                prob = (pheromone ** self.alpha) * (heuristic ** self.beta)
                probabilities.append(prob)
            
            # Normalize probabilities
            total_prob = sum(probabilities)
            if total_prob == 0:
                probabilities = [1.0 / len(available_features)] * len(available_features)
            else:
                probabilities = [p / total_prob for p in probabilities]
            
            # Select feature (exploitation vs exploration)
            if np.random.random() < self.q0:
                # Exploitation: select best feature
                selected_idx = np.argmax(probabilities)
            else:
                # Exploration: probabilistic selection
                selected_idx = np.random.choice(len(available_features), p=probabilities)
            
            selected_feature = available_features[selected_idx]
            selected_features.append(selected_feature)
            available_features.remove(selected_feature)
        
        # Calculate fitness for this solution
        fitness = self._evaluate_solution(selected_features, X_train, y_train)
        
        return selected_features, fitness
    
    def _evaluate_solution(self, 
                         selected_features: List[str], 
                         X_train: pd.DataFrame,
                         y_train: pd.Series) -> float:
        """
        Evaluate a solution using Decision Tree and multi-objective fitness
        
        Args:
            selected_features: List of selected features
            X_train: Training features
            y_train: Training target
            
        Returns:
            Fitness value (lower is better)
        """
        if len(selected_features) < self.min_features:
            return float('inf')
        
        try:
            # Prepare data
            X_selected = X_train[selected_features]
            
            # Handle missing values
            X_selected = X_selected.fillna(X_selected.mean())
            
            # Train Decision Tree
            dt = DecisionTreeRegressor(
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42
            )
            
            dt.fit(X_selected, y_train)
            
            # Predict and calculate RMSE
            y_pred = dt.predict(X_selected)
            rmse = np.sqrt(mean_squared_error(y_train, y_pred))
            
            # Multi-objective fitness: RMSE + feature count penalty
            feature_penalty = len(selected_features) * 0.01  # Small penalty for more features
            
            # Policy relevance bonus (negative penalty for policy-controllable features)
            policy_bonus = 0
            controllable_count = sum(1 for f in selected_features if f in self.policy_controllable)
            uncontrollable_count = sum(1 for f in selected_features if f in self.uncontrollable)
            
            if controllable_count > 0:
                policy_bonus = -0.05 * controllable_count  # Bonus for policy-controllable features
            
            fitness = rmse + feature_penalty + policy_bonus
            
            return fitness
            
        except Exception as e:
            logger.warning(f"Error evaluating solution: {e}")
            return float('inf')
    
    def _update_pheromones(self, 
                          pheromones: Dict[str, float],
                          solutions: List[Tuple[List[str], float]]) -> Dict[str, float]:
        """
        Update pheromone trails based on solution quality
        
        Args:
            pheromones: Current pheromone levels
            solutions: List of (feature_subset, fitness) tuples
            
        Returns:
            Updated pheromone levels
        """
        # Evaporate pheromones
        for feature in pheromones:
            pheromones[feature] *= (1 - self.rho)
        
        # Deposit pheromones based on solution quality
        for selected_features, fitness in solutions:
            if fitness == float('inf'):
                continue
                
            # Pheromone deposit amount (inverse of fitness)
            deposit = 1.0 / (1.0 + fitness)
            
            for feature in selected_features:
                pheromones[feature] += deposit
        
        return pheromones
    
    def optimize(self, 
                X: pd.DataFrame, 
                y: pd.Series,
                test_size: float = 0.2,
                random_state: int = 42) -> Dict:
        """
        Run ACO optimization to find best feature subset
        
        Args:
            X: Feature matrix
            y: Target variable
            test_size: Fraction of data to use for testing
            random_state: Random seed
            
        Returns:
            Dictionary containing optimization results
        """
        logger.info("Starting ACO feature selection optimization")
        
        # Prepare data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Get available features (exclude non-numeric columns)
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        logger.info(f"Available features for selection: {len(numeric_features)}")
        
        # Calculate target correlations for heuristic
        target_correlations = {}
        for feature in numeric_features:
            if feature in X_train.columns:
                corr = X_train[feature].corr(y_train)
                target_correlations[feature] = corr if not np.isnan(corr) else 0
        
        # Initialize pheromones and heuristics
        pheromones = self._initialize_pheromones(numeric_features)
        heuristics = self._calculate_heuristic(numeric_features, target_correlations)
        
        # Track convergence
        no_improvement_count = 0
        best_iteration = 0
        
        for iteration in range(self.n_iterations):
            logger.info(f"Iteration {iteration + 1}/{self.n_iterations}")
            
            # Generate solutions using ants
            solutions = []
            for ant in range(self.n_ants):
                selected_features, fitness = self._construct_solution(
                    numeric_features, pheromones, heuristics, X_train, y_train
                )
                solutions.append((selected_features, fitness))
            
            # Find best solution in this iteration
            iteration_best = min(solutions, key=lambda x: x[1])
            iteration_features, iteration_fitness = iteration_best
            
            # Update global best
            if iteration_fitness < self.best_fitness:
                self.best_fitness = iteration_fitness
                self.best_solution = iteration_features.copy()
                best_iteration = iteration
                no_improvement_count = 0
                logger.info(f"New best solution found: {len(self.best_solution)} features, fitness: {self.best_fitness:.4f}")
            else:
                no_improvement_count += 1
            
            # Update pheromones
            pheromones = self._update_pheromones(pheromones, solutions)
            
            # Track convergence
            self.convergence_history.append({
                'iteration': iteration + 1,
                'best_fitness': self.best_fitness,
                'iteration_fitness': iteration_fitness,
                'n_features': len(iteration_features),
                'pheromone_mean': np.mean(list(pheromones.values()))
            })
            
            self.feature_selection_history.append({
                'iteration': iteration + 1,
                'selected_features': iteration_features.copy()
            })
            
            # Early stopping
            if no_improvement_count >= 10:
                logger.info(f"Early stopping: No improvement for {no_improvement_count} iterations")
                break
        
        # Final evaluation on test set
        if self.best_solution:
            X_test_selected = X_test[self.best_solution].fillna(X_test[self.best_solution].mean())
            
            dt_final = DecisionTreeRegressor(
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42
            )
            
            X_train_selected = X_train[self.best_solution].fillna(X_train[self.best_solution].mean())
            dt_final.fit(X_train_selected, y_train)
            
            y_pred = dt_final.predict(X_test_selected)
            
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            test_mae = mean_absolute_error(y_test, y_pred)
            test_r2 = r2_score(y_test, y_pred)
            
            logger.info(f"Optimization completed. Best solution: {len(self.best_solution)} features")
            logger.info(f"Test RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, R²: {test_r2:.4f}")
            
            return {
                'best_features': self.best_solution,
                'best_fitness': self.best_fitness,
                'test_rmse': test_rmse,
                'test_mae': test_mae,
                'test_r2': test_r2,
                'convergence_history': self.convergence_history,
                'feature_selection_history': self.feature_selection_history,
                'best_iteration': best_iteration,
                'total_iterations': iteration + 1,
                'policy_controllable_selected': [f for f in self.best_solution if f in self.policy_controllable],
                'uncontrollable_selected': [f for f in self.best_solution if f in self.uncontrollable]
            }
        else:
            logger.error("No valid solution found")
            return None
    
    def plot_convergence(self, save_path: str = None):
        """Plot convergence history"""
        if not self.convergence_history:
            logger.warning("No convergence history available")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot fitness convergence
        iterations = [h['iteration'] for h in self.convergence_history]
        best_fitness = [h['best_fitness'] for h in self.convergence_history]
        iteration_fitness = [h['iteration_fitness'] for h in self.convergence_history]
        
        ax1.plot(iterations, best_fitness, 'b-', label='Best Fitness', linewidth=2)
        ax1.plot(iterations, iteration_fitness, 'r--', alpha=0.7, label='Iteration Fitness')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Fitness (Lower is Better)')
        ax1.set_title('ACO Convergence: Fitness Evolution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot pheromone evolution
        pheromone_mean = [h['pheromone_mean'] for h in self.convergence_history]
        ax2.plot(iterations, pheromone_mean, 'g-', linewidth=2)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Mean Pheromone Level')
        ax2.set_title('ACO Convergence: Pheromone Evolution')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Convergence plot saved to: {save_path}")
        
        plt.show()
    
    def plot_feature_selection(self, save_path: str = None):
        """Plot feature selection analysis"""
        if not self.best_solution:
            logger.warning("No solution available for plotting")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Policy relevance analysis
        controllable_selected = [f for f in self.best_solution if f in self.policy_controllable]
        uncontrollable_selected = [f for f in self.best_solution if f in self.uncontrollable]
        
        labels = ['Policy-Controllable', 'Uncontrollable']
        sizes = [len(controllable_selected), len(uncontrollable_selected)]
        colors = ['#2E8B57', '#DC143C']
        
        ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Selected Features by Policy Relevance')
        
        # Feature count over iterations
        iterations = [h['iteration'] for h in self.convergence_history]
        n_features = [h['n_features'] for h in self.convergence_history]
        
        ax2.plot(iterations, n_features, 'b-', linewidth=2)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Number of Selected Features')
        ax2.set_title('Feature Count Evolution')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature selection plot saved to: {save_path}")
        
        plt.show()
    
    def save_results(self, output_dir: str):
        """Save optimization results to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save best features
        features_file = os.path.join(output_dir, 'best_features.txt')
        with open(features_file, 'w') as f:
            f.write("ACO Feature Selection Results\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Best Features ({len(self.best_solution)}):\n")
            for i, feature in enumerate(self.best_solution, 1):
                f.write(f"{i:2d}. {feature}\n")
            
            f.write(f"\nPolicy-Controllable Features:\n")
            for feature in self.best_solution:
                if feature in self.policy_controllable:
                    f.write(f"  - {feature}\n")
            
            f.write(f"\nUncontrollable Features:\n")
            for feature in self.best_solution:
                if feature in self.uncontrollable:
                    f.write(f"  - {feature}\n")
        
        # Save convergence history
        convergence_file = os.path.join(output_dir, 'convergence_history.csv')
        pd.DataFrame(self.convergence_history).to_csv(convergence_file, index=False)
        
        logger.info(f"Results saved to: {output_dir}")

