
import numpy as np
import pandas as pd
import os
import pickle
import json
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.regularizers import l1_l2
import warnings
warnings.filterwarnings('ignore')

class AdaptiveWindowAgent:
    """Agent that uses Enhanced Feature Engineering MLP to predict optimal window size"""
    
    def __init__(self, output_dir='./mlp_checkpoints/'):
        self.output_dir = output_dir
        self.y_scaler = StandardScaler()
        self.x_scaler = StandardScaler()
        self.model = None
        self.selector = None
        
        # Create directory structure
        self.enhanced_dir = f"{output_dir}enhanced_mlp/"
        self.checkpoint_dir = f"{self.enhanced_dir}checkpoints/"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Model parameters
        self.min_window = 5
        self.max_window = 50
        self.default_window = 20
        self.is_trained = False
        
        # Checkpoint files
        self.checkpoint_files = {
            'model': f"{self.enhanced_dir}enhanced_feature_model.keras",
            'scalers': f"{self.checkpoint_dir}scalers.pkl",
            'selector': f"{self.checkpoint_dir}selector.pkl",
            'training_state': f"{self.checkpoint_dir}training_state.json"
        }
        
        # Try to load existing model
        self.load_trained_model()
        
    def load_trained_model(self):
        """Load pre-trained model and scalers if available"""
        try:
            if os.path.exists(self.checkpoint_files['model']):
                self.model = load_model(self.checkpoint_files['model'])
                
                if os.path.exists(self.checkpoint_files['scalers']):
                    with open(self.checkpoint_files['scalers'], 'rb') as f:
                        scalers = pickle.load(f)
                    self.x_scaler = scalers['x_scaler']
                    self.y_scaler = scalers['y_scaler']
                
                if os.path.exists(self.checkpoint_files['selector']):
                    with open(self.checkpoint_files['selector'], 'rb') as f:
                        self.selector = pickle.load(f)
                
                self.is_trained = True
                print("✅ Loaded pre-trained Enhanced MLP model for window prediction")
            else:
                print("📝 No pre-trained model found, will use lightweight fallback")
        except Exception as e:
            print(f"⚠️ Could not load pre-trained model: {e}")
            self.is_trained = False
    
    def create_enhanced_features(self, data_matrix: np.ndarray) -> np.ndarray:
        """Create enhanced features from time series data matrix"""
        if data_matrix.shape[0] < 5:
            return data_matrix
        
        try:
            enhanced_features = []
            
            # 1. Original features
            enhanced_features.append(data_matrix)
            
            # 2. Best interaction (Feature 0 × Feature 1 for simplicity)
            if data_matrix.shape[1] > 1:
                best_interaction = data_matrix[:, 0] * data_matrix[:, 1]
                enhanced_features.append(best_interaction.reshape(-1, 1))
            
            # 3. Top feature interactions (limited for real-time)
            interaction_features = []
            top_features = min(5, data_matrix.shape[1])
            
            for i in range(top_features):
                for j in range(i+1, top_features):
                    if len(interaction_features) < 10:  # Limit for performance
                        # Multiplication
                        mult_feat = data_matrix[:, i] * data_matrix[:, j]
                        interaction_features.append(mult_feat)
                        
                        # Safe division
                        if np.all(np.abs(data_matrix[:, j]) > 1e-8):
                            div_feat = data_matrix[:, i] / (data_matrix[:, j] + 1e-8)
                            interaction_features.append(div_feat)
            
            if interaction_features:
                interaction_matrix = np.column_stack(interaction_features)
                enhanced_features.append(interaction_matrix)
            
            # 4. Polynomial features (limited)
            poly_features = []
            for i in range(min(3, data_matrix.shape[1])):
                # Quadratic terms
                quad_feat = data_matrix[:, i] ** 2
                poly_features.append(quad_feat)
            
            if poly_features:
                poly_matrix = np.column_stack(poly_features)
                enhanced_features.append(poly_matrix)
            
            # 5. Statistical features
            stat_features = []
            if data_matrix.shape[1] >= 3:
                mean_feat = np.mean(data_matrix, axis=1)
                std_feat = np.std(data_matrix, axis=1)
                stat_features.extend([mean_feat, std_feat])
            
            if stat_features:
                stat_matrix = np.column_stack(stat_features)
                enhanced_features.append(stat_matrix)
            
            # Combine all features
            X_enhanced = np.hstack(enhanced_features)
            return X_enhanced
            
        except Exception as e:
            print(f"⚠️ Feature enhancement failed: {e}, using original features")
            return data_matrix
    
    def extract_features_from_df(self, df: pd.DataFrame) -> np.ndarray:
        """Extract features from DataFrame for window prediction"""
        if len(df) < 5:
            # Return minimal features for new streams
            return np.array([[0.1, 0.1, 0.1, 0.1, 0.1]])
        
        try:
            # Get numeric columns (exclude timestamp)
            numeric_cols = [col for col in df.columns if col != 'timestamp']
            if not numeric_cols:
                return np.array([[0.1, 0.1, 0.1, 0.1, 0.1]])
            
            # Take recent data
            recent_data = df[numeric_cols].tail(min(20, len(df))).dropna()
            if len(recent_data) < 3:
                return np.array([[0.1, 0.1, 0.1, 0.1, 0.1]])
            
            # Convert to matrix and create enhanced features
            data_matrix = recent_data.values
            enhanced_features = self.create_enhanced_features(data_matrix)
            
            # Take the last row as current features
            current_features = enhanced_features[-1:, :]
            
            return current_features
            
        except Exception as e:
            print(f"⚠️ Feature extraction failed: {e}")
            return np.array([[0.1, 0.1, 0.1, 0.1, 0.1]])
    
    def build_lightweight_mlp(self, input_dim: int):
        """Build lightweight MLP for real-time inference"""
        model = Sequential([
            Dense(512, input_dim=input_dim, activation='relu', 
                  kernel_regularizer=l1_l2(0.0001, 0.001)),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(128, activation='relu'),
            Dropout(0.2),
            
            Dense(64, activation='relu'),
            Dropout(0.1),
            
            Dense(32, activation='relu'),
            Dense(1, activation='linear')
        ])
        
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        model.compile(loss='mae', optimizer=optimizer, metrics=['mse', 'mae'])
        
        return model
    
    def train_fallback_model(self, df: pd.DataFrame):
        """Train a quick fallback model if no pre-trained model available"""
        if self.is_trained or len(df) < 20:
            return
        
        try:
            print("🚀 Training lightweight window prediction model...")
            
            # Generate synthetic training data based on current stream characteristics
            features_sample = self.extract_features_from_df(df)
            input_dim = features_sample.shape[1]
            
            # Create synthetic dataset
            X_train = []
            y_train = []
            
            for _ in range(500):  # Smaller dataset for speed
                # Generate synthetic features similar to current data
                base_features = np.random.normal(0, 1, input_dim)
                noise = np.random.normal(0, 0.1, input_dim)
                synthetic_features = base_features + noise
                
                # Heuristic window size based on feature characteristics
                volatility = np.std(synthetic_features)
                mean_abs = np.mean(np.abs(synthetic_features))
                
                optimal_window = max(self.min_window, 
                                   min(self.max_window, 
                                       int(self.min_window + volatility * 5 + mean_abs * 3)))
                
                X_train.append(synthetic_features)
                y_train.append(optimal_window)
            
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            
            # Split and scale
            X_train_scaled = self.x_scaler.fit_transform(X_train)
            y_train_scaled = self.y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
            
            # Build and train model
            self.model = self.build_lightweight_mlp(input_dim)
            
            self.model.fit(
                X_train_scaled, y_train_scaled,
                epochs=100,
                batch_size=32,
                verbose=0,
                validation_split=0.2,
                callbacks=[EarlyStopping(patience=10, restore_best_weights=True)]
            )
            
            # Save the model and scalers
            self.model.save(self.checkpoint_files['model'])
            with open(self.checkpoint_files['scalers'], 'wb') as f:
                pickle.dump({'x_scaler': self.x_scaler, 'y_scaler': self.y_scaler}, f)
            
            self.is_trained = True
            print("✅ Lightweight model trained and saved")
            
        except Exception as e:
            print(f"⚠️ Fallback model training failed: {e}")
    
    def predict_window(self, df: pd.DataFrame) -> int:
        """Predict optimal window size using Enhanced MLP or fallback"""
        try:
            # Train fallback model if no model is available
            if not self.is_trained:
                self.train_fallback_model(df)
            
            # If still no model, return default
            if not self.is_trained or self.model is None:
                return self.default_window
            
            # Extract and enhance features
            features = self.extract_features_from_df(df)
            
            # Apply feature selection if available
            if self.selector is not None:
                try:
                    features = self.selector.transform(features)
                except:
                    pass  # Continue without selection if it fails
            
            # Scale features
            features_scaled = self.x_scaler.transform(features)
            
            # Predict
            prediction_scaled = self.model.predict(features_scaled, verbose=0)
            prediction = self.y_scaler.inverse_transform(prediction_scaled.reshape(-1, 1))[0, 0]
            
            # Clamp to valid range
            window_size = int(max(self.min_window, min(self.max_window, prediction)))
            
            return window_size
            
        except Exception as e:
            print(f"⚠️ Window prediction failed: {e}, using default")
            return self.default_window
    
    def get_model_info(self) -> dict:
        """Get information about the current model"""
        return {
            'is_trained': self.is_trained,
            'model_available': self.model is not None,
            'has_selector': self.selector is not None,
            'model_path': self.checkpoint_files['model'],
            'model_exists': os.path.exists(self.checkpoint_files['model'])
        }
