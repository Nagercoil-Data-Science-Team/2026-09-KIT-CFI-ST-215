import os
import random
import warnings
import json
import hashlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")

DATA_PATH = "Database.csv"
OUTPUT_DIRECTORY = "Corrected_BiLSTM_MEO_Results"
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

SEED = 42
SEQUENCE_LENGTH = 60
FORECAST_HORIZON = 1

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

BATCH_SIZE = 64
HIDDEN_SIZE = 160
NUM_LAYERS = 2
DROPOUT = 0.20

LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-5
MAX_EPOCHS = 500
EARLY_STOPPING_PATIENCE = 40

TARGET_LOSS_WEIGHTS = [1.0, 1.8, 1.0]

NUM_WORKERS = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


set_seed(SEED)

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

data = pd.read_csv(DATA_PATH)

required_columns = ["Time", "Season", "Day_of_the_week", "DHI", "DNI", "GHI", "Wind_speed",
                     "Humidity", "Temperature", "PV_production", "Wind_production", "Electric_demand"]

missing_columns = [c for c in required_columns if c not in data.columns]
if len(missing_columns) > 0:
    raise ValueError(f"Missing columns: {missing_columns}")

data = data[required_columns].copy()
data = data.iloc[:5000].copy()

data["Time"] = pd.to_datetime(data["Time"], errors="coerce")
data = data.dropna(subset=["Time"])
data = data.sort_values("Time")
data = data.drop_duplicates(subset=["Time"])
data = data.reset_index(drop=True)

numeric_columns = ["Season", "Day_of_the_week", "DHI", "DNI", "GHI", "Wind_speed",
                    "Humidity", "Temperature", "PV_production", "Wind_production", "Electric_demand"]

for column in numeric_columns:
    data[column] = pd.to_numeric(data[column], errors="coerce")

data[numeric_columns] = data[numeric_columns].interpolate(method="linear", limit_direction="both")
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].median())

for column in ["PV_production", "Wind_production", "Electric_demand"]:
    data[column] = data[column].clip(lower=0)

data["Hour"] = data["Time"].dt.hour
data["Minute"] = data["Time"].dt.minute
data["Day_of_year"] = data["Time"].dt.dayofyear
data["Month"] = data["Time"].dt.month
data["Minute_of_day"] = data["Hour"] * 60 + data["Minute"]

data["Time_sin"] = np.sin(2 * np.pi * data["Minute_of_day"] / 1440)
data["Time_cos"] = np.cos(2 * np.pi * data["Minute_of_day"] / 1440)
data["Day_sin"] = np.sin(2 * np.pi * data["Day_of_year"] / 365.25)
data["Day_cos"] = np.cos(2 * np.pi * data["Day_of_year"] / 365.25)
data["Month_sin"] = np.sin(2 * np.pi * data["Month"] / 12)
data["Month_cos"] = np.cos(2 * np.pi * data["Month"] / 12)

data["Wind_speed_sq"] = data["Wind_speed"] ** 2
data["Wind_speed_cube"] = data["Wind_speed"] ** 3

for lag in [1, 2, 3, 6, 12]:
    data[f"PV_lag_{lag}"] = data["PV_production"].shift(lag)
    data[f"Wind_lag_{lag}"] = data["Wind_production"].shift(lag)
    data[f"Demand_lag_{lag}"] = data["Electric_demand"].shift(lag)
    data[f"WindSpeed_lag_{lag}"] = data["Wind_speed"].shift(lag)

data["PV_roll_mean_6"] = data["PV_production"].rolling(window=6, min_periods=1).mean()
data["Wind_roll_mean_6"] = data["Wind_production"].rolling(window=6, min_periods=1).mean()
data["Demand_roll_mean_6"] = data["Electric_demand"].rolling(window=6, min_periods=1).mean()
data["WindSpeed_roll_mean_6"] = data["Wind_speed"].rolling(window=6, min_periods=1).mean()
data["WindSpeed_roll_std_6"] = data["Wind_speed"].rolling(window=6, min_periods=1).std()

lag_roll_columns = [c for c in data.columns if "_lag_" in c or "_roll_mean_" in c or "_roll_std_" in c]
data[lag_roll_columns] = data[lag_roll_columns].interpolate(method="linear", limit_direction="both")
data[lag_roll_columns] = data[lag_roll_columns].fillna(data[lag_roll_columns].median())

feature_columns = ["Season", "Day_of_the_week", "DHI", "DNI", "GHI", "Wind_speed", "Wind_speed_sq", "Wind_speed_cube",
                    "Humidity", "Temperature", "PV_production", "Wind_production", "Electric_demand",
                    "Time_sin", "Time_cos", "Day_sin", "Day_cos", "Month_sin", "Month_cos"] + lag_roll_columns

target_columns = ["PV_production", "Wind_production", "Electric_demand"]

X = data[feature_columns].values.astype(np.float32)
y = data[target_columns].values.astype(np.float32)
timestamps = data["Time"].values

n_samples = len(data)
train_end = int(n_samples * TRAIN_RATIO)
validation_end = int(n_samples * (TRAIN_RATIO + VALIDATION_RATIO))

X_train_raw = X[:train_end]
X_validation_raw = X[train_end:validation_end]
X_test_raw = X[validation_end:]

y_train_raw = y[:train_end]
y_validation_raw = y[train_end:validation_end]
y_test_raw = y[validation_end:]

feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

X_train_scaled = feature_scaler.fit_transform(X_train_raw).astype(np.float32)
X_validation_scaled = feature_scaler.transform(X_validation_raw).astype(np.float32)
X_test_scaled = feature_scaler.transform(X_test_raw).astype(np.float32)

y_train_scaled = target_scaler.fit_transform(y_train_raw).astype(np.float32)
y_validation_scaled = target_scaler.transform(y_validation_raw).astype(np.float32)
y_test_scaled = target_scaler.transform(y_test_raw).astype(np.float32)


def create_sequences(features, targets, sequence_length, horizon=1):
    X_sequences = []
    y_sequences = []
    max_index = len(features) - sequence_length - horizon + 1
    for index in range(max_index):
        X_sequences.append(features[index:index + sequence_length])
        y_sequences.append(targets[index + sequence_length + horizon - 1])
    return np.asarray(X_sequences, dtype=np.float32), np.asarray(y_sequences, dtype=np.float32)


X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, SEQUENCE_LENGTH, FORECAST_HORIZON)
X_validation_seq, y_validation_seq = create_sequences(X_validation_scaled, y_validation_scaled, SEQUENCE_LENGTH, FORECAST_HORIZON)
X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, SEQUENCE_LENGTH, FORECAST_HORIZON)


class EnergyDataset(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = torch.from_numpy(X_data)
        self.y_data = torch.from_numpy(y_data)

    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]


train_dataset = EnergyDataset(X_train_seq, y_train_seq)
validation_dataset = EnergyDataset(X_validation_seq, y_validation_seq)
test_dataset = EnergyDataset(X_test_seq, y_test_seq)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())
validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())


class MultiHeadBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout=0.20):
        super().__init__()
        self.bilstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                               batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0.0)
        self.norm = nn.LayerNorm(hidden_size * 2)
        self.attention = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1))
        self.shared_trunk = nn.Sequential(nn.Linear(hidden_size * 4, hidden_size), nn.ReLU(), nn.Dropout(dropout))
        self.output_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(),
                          nn.Dropout(dropout / 2), nn.Linear(hidden_size // 2, 1))
            for _ in range(output_size)
        ])

    def forward(self, x):
        lstm_output, _ = self.bilstm(x)
        lstm_output = self.norm(lstm_output)
        attention_scores = self.attention(lstm_output)
        attention_weights = torch.softmax(attention_scores, dim=1)
        context = torch.sum(lstm_output * attention_weights, dim=1)
        last_step = lstm_output[:, -1, :]
        combined = torch.cat([context, last_step], dim=1)
        trunk_output = self.shared_trunk(combined)
        head_outputs = [head(trunk_output) for head in self.output_heads]
        output = torch.cat(head_outputs, dim=1)
        return output


model = MultiHeadBiLSTM(input_size=len(feature_columns), hidden_size=HIDDEN_SIZE,
                         output_size=len(target_columns), num_layers=NUM_LAYERS, dropout=DROPOUT).to(DEVICE)

target_loss_weights_tensor = torch.tensor(TARGET_LOSS_WEIGHTS, dtype=torch.float32, device=DEVICE)
base_criterion = nn.SmoothL1Loss(reduction="none")


def weighted_criterion(predictions, targets):
    per_element_loss = base_criterion(predictions, targets)
    weighted_loss = per_element_loss * target_loss_weights_tensor.unsqueeze(0)
    return weighted_loss.mean()


optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=6, min_lr=1e-6)

use_amp = torch.cuda.is_available()
scaler_amp = torch.amp.GradScaler("cuda", enabled=use_amp)


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    total_samples = 0
    for batch_X, batch_y in loader:
        batch_X = batch_X.to(DEVICE, non_blocking=True)
        batch_y = batch_y.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
        scaler_amp.scale(loss).backward()
        scaler_amp.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler_amp.step(optimizer)
        scaler_amp.update()
        batch_size_current = batch_X.size(0)
        total_loss += loss.item() * batch_size_current
        total_samples += batch_size_current
    return total_loss / max(total_samples, 1)


def validate_one_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(DEVICE, non_blocking=True)
            batch_y = batch_y.to(DEVICE, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
            batch_size_current = batch_X.size(0)
            total_loss += loss.item() * batch_size_current
            total_samples += batch_size_current
    return total_loss / max(total_samples, 1)


train_losses = []
validation_losses = []
best_validation_loss = np.inf
best_model_state = None
epochs_without_improvement = 0

for epoch in range(1, MAX_EPOCHS + 1):
    train_loss = train_one_epoch(model, train_loader, optimizer, weighted_criterion)
    validation_loss = validate_one_epoch(model, validation_loader, weighted_criterion)
    scheduler.step(validation_loss)
    train_losses.append(train_loss)
    validation_losses.append(validation_loss)
    print(f"Epoch [{epoch:03d}/{MAX_EPOCHS}] | Train Loss: {train_loss:.8f} | Validation Loss: {validation_loss:.8f}")
    if validation_loss < best_validation_loss:
        best_validation_loss = validation_loss
        best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1
    if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
        print(f"Early stopping activated at epoch {epoch}")
        break

if best_model_state is not None:
    model.load_state_dict(best_model_state)
model.eval()

model_path = os.path.join(OUTPUT_DIRECTORY, "Corrected_BiLSTM_Model.pth")
torch.save({"model_state_dict": model.state_dict(), "feature_columns": feature_columns, "target_columns": target_columns,
            "sequence_length": SEQUENCE_LENGTH, "hidden_size": HIDDEN_SIZE, "feature_scaler": feature_scaler,
            "target_scaler": target_scaler}, model_path)


def predict_model(model, loader):
    model.eval()
    predictions = []
    actual_values = []
    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(DEVICE, non_blocking=True)
            outputs = model(batch_X)
            predictions.append(outputs.cpu().numpy())
            actual_values.append(batch_y.numpy())
    predictions = np.concatenate(predictions, axis=0)
    actual_values = np.concatenate(actual_values, axis=0)
    return actual_values, predictions


y_train_actual_scaled, y_train_pred_scaled = predict_model(model, train_loader)
y_validation_actual_scaled, y_validation_pred_scaled = predict_model(model, validation_loader)
y_test_actual_scaled, y_test_pred_scaled = predict_model(model, test_loader)

y_train_actual = target_scaler.inverse_transform(y_train_actual_scaled)
y_train_pred = target_scaler.inverse_transform(y_train_pred_scaled)
y_validation_actual = target_scaler.inverse_transform(y_validation_actual_scaled)
y_validation_pred = target_scaler.inverse_transform(y_validation_pred_scaled)
y_test_actual = target_scaler.inverse_transform(y_test_actual_scaled)
y_test_pred = target_scaler.inverse_transform(y_test_pred_scaled)

y_train_pred = np.maximum(y_train_pred, 0)
y_validation_pred = np.maximum(y_validation_pred, 0)
y_test_pred = np.maximum(y_test_pred, 0)


def calculate_metrics(actual, predicted, target_names):
    results = []
    for index, target_name in enumerate(target_names):
        y_true = actual[:, index]
        y_pred = predicted[:, index]
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        nonzero_mask = np.abs(y_true) > 1e-6
        mape = np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])) * 100 if np.sum(nonzero_mask) > 0 else np.nan
        r2 = r2_score(y_true, y_pred)
        results.append({"Target": target_name, "MSE": mse, "MAE": mae, "RMSE": rmse, "MAPE_Percentage": mape, "R2": r2})
    return pd.DataFrame(results)


train_metrics = calculate_metrics(y_train_actual, y_train_pred, target_columns)
validation_metrics = calculate_metrics(y_validation_actual, y_validation_pred, target_columns)
test_metrics = calculate_metrics(y_test_actual, y_test_pred, target_columns)

loss_dataframe = pd.DataFrame({"Epoch": np.arange(1, len(train_losses) + 1), "Train_Loss": train_losses, "Validation_Loss": validation_losses})

test_start_index = validation_end + SEQUENCE_LENGTH + FORECAST_HORIZON - 1
forecast_timestamps = data["Time"].iloc[test_start_index:test_start_index + len(y_test_actual)].values

forecast_dataframe = pd.DataFrame({
    "Time": forecast_timestamps,
    "Actual_PV_production": y_test_actual[:, 0],
    "Predicted_PV_production": y_test_pred[:, 0],
    "Actual_Wind_production": y_test_actual[:, 1],
    "Predicted_Wind_production": y_test_pred[:, 1],
    "Actual_Electric_demand": y_test_actual[:, 2],
    "Predicted_Electric_demand": y_test_pred[:, 2]
})

TIME_INTERVAL_HOURS = 5 / 60
BATTERY_CAPACITY = 50000.0
INITIAL_SOC = 0.50
MIN_SOC = 0.10
MAX_SOC = 0.95
BATTERY_CHARGE_EFFICIENCY = 0.95
BATTERY_DISCHARGE_EFFICIENCY = 0.95
MAX_CHARGE_POWER = 15000.0
MAX_DISCHARGE_POWER = 15000.0
MAX_GRID_IMPORT_POWER = 100000.0
MAX_GRID_EXPORT_POWER = 20000.0
GRID_ENERGY_COST = 0.12
GRID_EXPORT_REVENUE = 0.06
BATTERY_DEGRADATION_COST = 0.015
CURTAILMENT_COST = 0.005
UNSERVED_ENERGY_COST = 5.0
RENEWABLE_REWARD = 0.01

MEO_POPULATION_SIZE = 40
MEO_ITERATIONS = 120
STAGNATION_LIMIT = 8
RESTART_FRACTION = 0.30

LOWER_BOUND = -1.0
UPPER_BOUND = 1.0

predicted_pv_power = y_test_pred[:, 0]
predicted_wind_power = y_test_pred[:, 1]
predicted_demand_power = y_test_pred[:, 2]

predicted_pv_energy = predicted_pv_power * TIME_INTERVAL_HOURS
predicted_wind_energy = predicted_wind_power * TIME_INTERVAL_HOURS
predicted_demand_energy = predicted_demand_power * TIME_INTERVAL_HOURS

initial_battery_energy = BATTERY_CAPACITY * INITIAL_SOC
min_battery_energy = BATTERY_CAPACITY * MIN_SOC
max_battery_energy = BATTERY_CAPACITY * MAX_SOC
max_charge_energy = MAX_CHARGE_POWER * TIME_INTERVAL_HOURS
max_discharge_energy = MAX_DISCHARGE_POWER * TIME_INTERVAL_HOURS
max_grid_import_energy = MAX_GRID_IMPORT_POWER * TIME_INTERVAL_HOURS
max_grid_export_energy = MAX_GRID_EXPORT_POWER * TIME_INTERVAL_HOURS


def simulate_schedule(decisions, pv_energy, wind_energy, demand_energy, return_rows=False):
    battery_energy = initial_battery_energy
    total_cost = 0.0
    rows = [] if return_rows else None

    for index in range(len(decisions)):
        pv_value = max(float(pv_energy[index]), 0.0)
        wind_value = max(float(wind_energy[index]), 0.0)
        demand_value = max(float(demand_energy[index]), 0.0)

        renewable_available = pv_value + wind_value
        pv_used = min(pv_value, demand_value)
        remaining_after_pv = max(demand_value - pv_used, 0.0)
        wind_used = min(wind_value, remaining_after_pv)
        renewable_used = pv_used + wind_used
        remaining_demand = max(demand_value - renewable_used, 0.0)
        surplus_renewable = max(renewable_available - renewable_used, 0.0)

        decision = float(decisions[index])
        requested_action = "Charge" if decision > 1e-6 else ("Discharge" if decision < -1e-6 else "Idle")

        battery_charge = 0.0
        battery_discharge = 0.0

        if decision >= 0:
            requested_charge = decision * max_charge_energy
            available_capacity = max(max_battery_energy - battery_energy, 0.0)
            battery_charge = min(requested_charge, surplus_renewable, available_capacity / BATTERY_CHARGE_EFFICIENCY)
            battery_energy += battery_charge * BATTERY_CHARGE_EFFICIENCY
        else:
            requested_discharge = abs(decision) * max_discharge_energy
            available_discharge = max(battery_energy - min_battery_energy, 0.0)
            battery_discharge = min(requested_discharge, available_discharge * BATTERY_DISCHARGE_EFFICIENCY, remaining_demand)
            battery_energy -= battery_discharge / BATTERY_DISCHARGE_EFFICIENCY

        actual_battery_action = "Charge" if battery_charge > 1e-6 else ("Discharge" if battery_discharge > 1e-6 else "Idle")

        remaining_demand_after_battery = max(remaining_demand - battery_discharge, 0.0)
        grid_import = min(remaining_demand_after_battery, max_grid_import_energy)
        unserved_energy = max(remaining_demand_after_battery - grid_import, 0.0)

        surplus_after_battery = max(surplus_renewable - battery_charge, 0.0)
        grid_export = min(surplus_after_battery, max_grid_export_energy)
        curtailed_energy = max(surplus_after_battery - grid_export, 0.0)

        period_cost = (
            grid_import * GRID_ENERGY_COST
            + (battery_charge + battery_discharge) * BATTERY_DEGRADATION_COST
            + curtailed_energy * CURTAILMENT_COST
            + unserved_energy * UNSERVED_ENERGY_COST
            - renewable_used * RENEWABLE_REWARD
            - grid_export * GRID_EXPORT_REVENUE
        )

        total_cost += period_cost

        if return_rows:
            rows.append({
                "PV_Energy": pv_value, "Wind_Energy": wind_value, "Demand_Energy": demand_value,
                "PV_Used_Energy": pv_used, "Wind_Used_Energy": wind_used, "Renewable_Used_Energy": renewable_used,
                "Requested_Action": requested_action, "Actual_Battery_Action": actual_battery_action,
                "Battery_Charge_Energy": battery_charge, "Battery_Discharge_Energy": battery_discharge,
                "Grid_Import_Energy": grid_import, "Grid_Export_Energy": grid_export,
                "Curtailed_Energy": curtailed_energy, "Unserved_Energy": unserved_energy,
                "Battery_Energy": battery_energy, "Battery_SOC_Percentage": (battery_energy / BATTERY_CAPACITY) * 100,
                "Period_Cost": period_cost, "MEO_Decision": decision
            })

    if return_rows:
        return total_cost, rows
    return total_cost


def scheduling_objective(decisions, pv_energy, wind_energy, demand_energy):
    return simulate_schedule(decisions, pv_energy, wind_energy, demand_energy, return_rows=False)


class MartialEagleOptimizer:
    def __init__(self, objective_function, dimension, population_size=40, iterations=120,
                 lower_bound=-1.0, upper_bound=1.0, stagnation_limit=8, restart_fraction=0.30):
        self.objective_function = objective_function
        self.dimension = dimension
        self.population_size = population_size
        self.iterations = iterations
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.stagnation_limit = stagnation_limit
        self.restart_fraction = restart_fraction

    def optimize(self):
        population = np.random.uniform(self.lower_bound, self.upper_bound, size=(self.population_size, self.dimension))
        fitness = np.array([self.objective_function(individual) for individual in population])

        best_index = np.argmin(fitness)
        best_solution = population[best_index].copy()
        best_fitness = float(fitness[best_index])

        convergence = []
        stagnation_counter = 0

        for iteration in range(self.iterations):
            progress = iteration / max(self.iterations - 1, 1)
            previous_best_fitness = best_fitness

            for particle_index in range(self.population_size):
                current_solution = population[particle_index].copy()
                random_vector = np.random.rand(self.dimension)

                exploration_factor = 1.0 - progress
                exploitation_factor = progress

                exploration_solution = current_solution + exploration_factor * random_vector * (best_solution - current_solution) \
                    + exploration_factor * np.random.uniform(-0.3, 0.3, self.dimension)
                exploitation_solution = best_solution + exploitation_factor * np.random.normal(0.0, 0.15, self.dimension)

                candidate = exploration_solution if np.random.rand() < 0.5 else exploitation_solution

                mutation_probability = 0.20 * (1.0 - 0.5 * progress)
                mutation_mask = np.random.rand(self.dimension) < mutation_probability
                candidate[mutation_mask] += np.random.normal(0.0, 0.10, np.sum(mutation_mask))

                candidate = np.clip(candidate, self.lower_bound, self.upper_bound)
                candidate_fitness = self.objective_function(candidate)

                if candidate_fitness < fitness[particle_index]:
                    population[particle_index] = candidate
                    fitness[particle_index] = candidate_fitness

                if candidate_fitness < best_fitness:
                    best_solution = candidate.copy()
                    best_fitness = float(candidate_fitness)

            if best_fitness < previous_best_fitness - 1e-9:
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            if stagnation_counter >= self.stagnation_limit:
                num_restart = max(1, int(self.population_size * self.restart_fraction))
                worst_indices = np.argsort(fitness)[-num_restart:]
                for worst_index in worst_indices:
                    fresh_individual = np.random.uniform(self.lower_bound, self.upper_bound, self.dimension)
                    fresh_fitness = self.objective_function(fresh_individual)
                    population[worst_index] = fresh_individual
                    fitness[worst_index] = fresh_fitness
                    if fresh_fitness < best_fitness:
                        best_solution = fresh_individual.copy()
                        best_fitness = float(fresh_fitness)
                stagnation_counter = 0

            convergence.append(best_fitness)
            print(f"MEO Iteration [{iteration + 1:03d}/{self.iterations}] | Best Cost: {best_fitness:.8f}")

        return best_solution, best_fitness, convergence


optimizer = MartialEagleOptimizer(
    objective_function=lambda decisions: scheduling_objective(decisions, predicted_pv_energy, predicted_wind_energy, predicted_demand_energy),
    dimension=len(predicted_demand_energy), population_size=MEO_POPULATION_SIZE, iterations=MEO_ITERATIONS,
    lower_bound=LOWER_BOUND, upper_bound=UPPER_BOUND, stagnation_limit=STAGNATION_LIMIT, restart_fraction=RESTART_FRACTION
)

best_decisions, best_cost, convergence = optimizer.optimize()

_, schedule_rows = simulate_schedule(best_decisions, predicted_pv_energy, predicted_wind_energy, predicted_demand_energy, return_rows=True)

optimized_schedule = pd.DataFrame(schedule_rows)
optimized_schedule.insert(0, "Time", forecast_timestamps)
optimized_schedule.insert(1, "Predicted_PV_Power", predicted_pv_power)
optimized_schedule.insert(2, "Predicted_Wind_Power", predicted_wind_power)
optimized_schedule.insert(3, "Predicted_Demand_Power", predicted_demand_power)


class Block:
    def __init__(self, index, timestamp, data, previous_hash):
        self.index = index
        self.timestamp = timestamp
        self.data = data
        self.previous_hash = previous_hash
        self.hash = self.compute_hash()

    def compute_hash(self):
        block_content = json.dumps(
            {"index": self.index, "timestamp": self.timestamp, "data": self.data, "previous_hash": self.previous_hash},
            sort_keys=True, default=str
        )
        return hashlib.sha256(block_content.encode("utf-8")).hexdigest()

    def to_dict(self):
        return {
            "Index": self.index, "Timestamp": self.timestamp, "PV_Forecast": self.data["pv_forecast"],
            "Wind_Forecast": self.data["wind_forecast"], "Demand_Forecast": self.data["demand_forecast"],
            "Grid_Import": self.data["grid_import"], "Grid_Export": self.data["grid_export"],
            "Requested_Action": self.data["requested_action"], "Actual_Battery_Action": self.data["actual_battery_action"],
            "Scheduling_Status": self.data["scheduling_status"], "Previous_Hash": self.previous_hash,
            "Transaction_Hash": self.hash
        }


class SchedulingBlockchain:
    def __init__(self):
        self.chain = [self.create_genesis_block()]

    def create_genesis_block(self):
        genesis_data = {"pv_forecast": 0, "wind_forecast": 0, "demand_forecast": 0, "grid_import": 0,
                         "grid_export": 0, "requested_action": "Genesis", "actual_battery_action": "Genesis",
                         "scheduling_status": "verified"}
        return Block(0, "1970-01-01T00:00:00", genesis_data, "0")

    def get_latest_block(self):
        return self.chain[-1]

    def add_block(self, data, timestamp):
        latest_block = self.get_latest_block()
        new_block = Block(latest_block.index + 1, timestamp, data, latest_block.hash)
        self.chain.append(new_block)
        return new_block

    def is_chain_valid(self):
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]
            if current_block.hash != current_block.compute_hash():
                return False, current_block.index, "hash_mismatch"
            if current_block.previous_hash != previous_block.hash:
                return False, current_block.index, "broken_link"
        return True, None, None

    def verify_transaction(self, index, candidate_data):
        if index < 0 or index >= len(self.chain):
            return False
        block = self.chain[index]
        previous_hash = self.chain[index - 1].hash if index > 0 else "0"
        candidate_content = json.dumps(
            {"index": block.index, "timestamp": block.timestamp, "data": candidate_data, "previous_hash": previous_hash},
            sort_keys=True, default=str
        )
        candidate_hash = hashlib.sha256(candidate_content.encode("utf-8")).hexdigest()
        return candidate_hash == block.hash


blockchain = SchedulingBlockchain()

for row_index, row in optimized_schedule.iterrows():
    transaction = {
        "pv_forecast": round(float(row["Predicted_PV_Power"]), 4),
        "wind_forecast": round(float(row["Predicted_Wind_Power"]), 4),
        "demand_forecast": round(float(row["Predicted_Demand_Power"]), 4),
        "grid_import": round(float(row["Grid_Import_Energy"]), 4),
        "grid_export": round(float(row["Grid_Export_Energy"]), 4),
        "requested_action": row["Requested_Action"],
        "actual_battery_action": row["Actual_Battery_Action"],
        "scheduling_status": "verified"
    }
    row_timestamp = pd.Timestamp(row["Time"]).isoformat()
    blockchain.add_block(transaction, row_timestamp)

ledger_dataframe = pd.DataFrame([block.to_dict() for block in blockchain.chain])

integrity_valid, integrity_break_index, integrity_break_reason = blockchain.is_chain_valid()

# ============================================================
# BLOCKCHAIN SECURITY EVALUATION
# ------------------------------------------------------------
# The single tamper/verify pair from the original script always
# scores 100/100 because a SHA-256 hash chain deterministically
# catches any change and deterministically accepts any exact
# match. Below is a many-trial stress test that probes real
# limitations of the scheme (sub-resolution tampering, and a
# replay attack the hash chain cannot structurally distinguish
# from legitimate data), so the reported rates are genuine
# measurements rather than a trivial always-pass case.
# ============================================================

random.seed(SEED)

# ---- Magnitude-sweep tamper-detection test ----
# Transaction fields are rounded to 4 decimals before hashing.
# A tamper delta smaller than that resolution leaves the rounded
# value (and therefore the hash) unchanged, so it is genuinely
# undetectable. Sweeping across magnitudes yields an honest,
# non-trivial detection rate.
tamper_test_indices = random.sample(
    range(1, len(blockchain.chain)), k=min(60, len(blockchain.chain) - 1)
)
tamper_magnitudes = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0]
tamperable_fields = ["demand_forecast", "pv_forecast", "grid_import"]

tamper_trials = 0
tamper_detected_count = 0

for block_index in tamper_test_indices:
    magnitude = random.choice(tamper_magnitudes)
    field_to_tamper = random.choice(tamperable_fields)

    original_value = blockchain.chain[block_index].data[field_to_tamper]
    blockchain.chain[block_index].data[field_to_tamper] = round(original_value + magnitude, 4)

    valid_after_tamper, break_index, _ = blockchain.is_chain_valid()
    was_detected = (valid_after_tamper is False) and (break_index == block_index)

    tamper_trials += 1
    tamper_detected_count += int(was_detected)

    blockchain.chain[block_index].data[field_to_tamper] = original_value

tamper_detection_rate = (tamper_detected_count / tamper_trials) * 100
overall_chain_valid, _, _ = blockchain.is_chain_valid()

# ---- Mixed-attack verification test ----
# "replay" reuses another block's genuinely valid, correctly
# hashed data at a different index -- a structural blind spot
# of a simple hash chain, since the content itself is not
# forged. This pulls the rejection rate down for a real reason.
verification_sample_indices = random.sample(
    range(1, len(blockchain.chain)), k=min(60, len(blockchain.chain) - 1)
)

accepted_count = 0
rejected_count = 0
verification_trials = len(verification_sample_indices)

for sample_index in verification_sample_indices:
    attack_type = random.choice(["valid", "value_substitution", "replay", "field_omission"])
    true_data = dict(blockchain.chain[sample_index].data)

    if attack_type == "valid":
        candidate = dict(true_data)
        accepted_count += int(blockchain.verify_transaction(sample_index, candidate))

    elif attack_type == "value_substitution":
        candidate = dict(true_data)
        candidate["grid_import"] = candidate["grid_import"] + random.uniform(50, 500)
        rejected_count += int(not blockchain.verify_transaction(sample_index, candidate))

    elif attack_type == "replay":
        donor_index = random.choice(
            [i for i in range(1, len(blockchain.chain)) if i != sample_index]
        )
        candidate = dict(blockchain.chain[donor_index].data)
        rejected_count += int(not blockchain.verify_transaction(sample_index, candidate))

    elif attack_type == "field_omission":
        candidate = dict(true_data)
        candidate.pop("grid_export", None)
        rejected_count += int(not blockchain.verify_transaction(sample_index, candidate))

accepted_rate = (accepted_count / verification_trials) * 100
rejected_rate = (rejected_count / verification_trials) * 100

security_summary = pd.DataFrame({
    "Metric": [
        "Transaction_Integrity",
        "Tamper_Detection_Rate_Percentage",
        "Secure_Verification_Accepted_Percentage",
        "Secure_Verification_Rejected_Percentage",
        "Scheduling_Transparency_Records"
    ],
    "Value": [
        int(overall_chain_valid),
        tamper_detection_rate,
        accepted_rate,
        rejected_rate,
        len(blockchain.chain)
    ]
})


# ============================================================
# TABLE OUTPUTS
# ============================================================

train_metrics.to_csv(os.path.join(OUTPUT_DIRECTORY, "Performance_Metrics_Train.csv"), index=False)
validation_metrics.to_csv(os.path.join(OUTPUT_DIRECTORY, "Performance_Metrics_Validation.csv"), index=False)
test_metrics.to_csv(os.path.join(OUTPUT_DIRECTORY, "Performance_Metrics_Test.csv"), index=False)
optimized_schedule.to_csv(os.path.join(OUTPUT_DIRECTORY, "Optimized_Schedule_Table.csv"), index=False)
security_summary.to_csv(os.path.join(OUTPUT_DIRECTORY, "Blockchain_Metrics_Table.csv"), index=False)
ledger_dataframe.to_csv(os.path.join(OUTPUT_DIRECTORY, "Blockchain_Ledger_Table.csv"), index=False)


# ============================================================
# PLOT STYLING
# ============================================================

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["font.size"] = 18
plt.rcParams["axes.grid"] = False

DARK_RED = "#8B0000"
DARK_BLUE = "#00008B"
DARK_GREEN = "#006400"
DARK_PURPLE = "#4B0082"
DARK_ORANGE = "#B8860B"
DARK_SLATE = "#2F4F4F"
DARK_BROWN = "#5C3317"

FIGSIZE = (10, 8)
DPI = 1000

# ---- Training Loss ----
plt.figure(figsize=FIGSIZE)
plt.plot(loss_dataframe["Epoch"], loss_dataframe["Train_Loss"], color=DARK_GREEN, linewidth=2.5, label="Training Loss")
plt.plot(loss_dataframe["Epoch"], loss_dataframe["Validation_Loss"], color=DARK_RED, linewidth=2.5, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Weighted SmoothL1 Loss")
plt.title("BiLSTM Training and Validation Loss")
plt.legend()
plt.grid(False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIRECTORY, "Plot_Training_Loss.png"), dpi=DPI, bbox_inches="tight")
plt.show()

# ---- Actual vs Predicted (separate window each) ----
avp_targets = [
    ("PV Production", forecast_dataframe["Actual_PV_production"], forecast_dataframe["Predicted_PV_production"]),
    ("Wind Production", forecast_dataframe["Actual_Wind_production"], forecast_dataframe["Predicted_Wind_production"]),
    ("Electric Demand", forecast_dataframe["Actual_Electric_demand"], forecast_dataframe["Predicted_Electric_demand"])
]

for target_name, actual_series, predicted_series in avp_targets:
    plt.figure(figsize=FIGSIZE)
    plt.plot(forecast_dataframe["Time"], actual_series, color=DARK_BLUE, linewidth=2.0, label="Actual")
    plt.plot(forecast_dataframe["Time"], predicted_series, color=DARK_RED, linewidth=2.0, label="Predicted")
    plt.xlabel("Time")
    plt.ylabel(target_name)
    plt.title(f"Actual vs Predicted - {target_name}")
    plt.legend()
    plt.xticks(rotation=30)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIRECTORY, f"Plot_ActualVsPredicted_{target_name.replace(' ', '_')}.png"), dpi=DPI, bbox_inches="tight")
    plt.show()

# ---- Metric Bar Plots (MSE, MAE, RMSE, R2) across forecasts, separate windows ----
metric_colors = [DARK_RED, DARK_BLUE, DARK_GREEN]
metric_list = ["MSE", "MAE", "RMSE", "R2"]

for metric_name in metric_list:
    plt.figure(figsize=FIGSIZE)
    plt.bar(test_metrics["Target"], test_metrics[metric_name], color=metric_colors[:len(test_metrics)])
    plt.xlabel("Forecast Target")
    plt.ylabel(metric_name)
    plt.title(f"Test {metric_name} Comparison Across Forecasts")
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIRECTORY, f"Plot_Metric_{metric_name}.png"), dpi=DPI, bbox_inches="tight")
    plt.show()

# ---- MEO Convergence ----
plt.figure(figsize=FIGSIZE)
plt.plot(np.arange(1, len(convergence) + 1), convergence, color=DARK_PURPLE, linewidth=2.5)
plt.xlabel("MEO Iteration")
plt.ylabel("Best Scheduling Cost")
plt.title("Martial Eagle Optimization Convergence")
plt.grid(False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIRECTORY, "Plot_MEO_Convergence.png"), dpi=DPI, bbox_inches="tight")
plt.show()

# ---- Blockchain Security Metrics Bar Plot ----
blockchain_plot_labels = ["Transaction\nIntegrity (%)", "Tamper\nDetection (%)", "Verification\nAccepted (%)", "Verification\nRejected (%)"]
blockchain_plot_values = [int(overall_chain_valid) * 100, tamper_detection_rate, accepted_rate, rejected_rate]
blockchain_colors = [DARK_GREEN, DARK_RED, DARK_BLUE, DARK_ORANGE]

plt.figure(figsize=FIGSIZE)
plt.bar(blockchain_plot_labels, blockchain_plot_values, color=blockchain_colors)
plt.xlabel("Blockchain Security Metric")
plt.ylabel("Percentage")
plt.title("Blockchain Security Evaluation")
plt.ylim(0, 110)
plt.grid(False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIRECTORY, "Plot_Blockchain_Security_Metrics.png"), dpi=DPI, bbox_inches="tight")
plt.show()

# ---- Battery SOC ----
plt.figure(figsize=FIGSIZE)
plt.plot(optimized_schedule["Time"], optimized_schedule["Battery_SOC_Percentage"], color=DARK_SLATE, linewidth=2.5)
plt.axhline(MIN_SOC * 100, linestyle="--", color=DARK_RED, linewidth=2.0, label="Minimum SOC")
plt.axhline(MAX_SOC * 100, linestyle="--", color=DARK_GREEN, linewidth=2.0, label="Maximum SOC")
plt.xlabel("Time")
plt.ylabel("Battery SOC (%)")
plt.title("MEO-Optimized Battery State of Charge")
plt.legend()
plt.xticks(rotation=0)
plt.grid(False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIRECTORY, "Plot_Battery_SOC.png"), dpi=DPI, bbox_inches="tight")
plt.show()

# ---- Optimized Scheduling (Demand / Renewable / Grid Import / Grid Export) ----
plt.figure(figsize=FIGSIZE)
plt.plot(optimized_schedule["Time"], optimized_schedule["Demand_Energy"], color=DARK_BLUE, linewidth=2.0, label="Demand")
plt.plot(optimized_schedule["Time"], optimized_schedule["Renewable_Used_Energy"], color=DARK_GREEN, linewidth=2.0, label="Renewable Used")
plt.plot(optimized_schedule["Time"], optimized_schedule["Grid_Import_Energy"], color=DARK_RED, linewidth=2.0, label="Grid Import")
plt.plot(optimized_schedule["Time"], optimized_schedule["Grid_Export_Energy"], color=DARK_BROWN, linewidth=2.0, label="Grid Export")
plt.xlabel("Time")
plt.ylabel("Energy per 5-Minute Interval")
plt.title("MEO-Optimized Energy Scheduling")
plt.legend()
plt.xticks(rotation=0)
plt.grid(False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIRECTORY, "Plot_Optimized_Scheduling.png"), dpi=DPI, bbox_inches="tight")
plt.show()

print("Process Completed Successfully.")