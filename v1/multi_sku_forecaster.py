import pandas as pd
import numpy as np
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder

INPUT_FILE = "../demand_data.csv"
ACTUAL_FILE = "../actual_data.csv"
OUTPUT_FILE = "forecast_output.csv"
FORECAST_WEEKS = 26 
INTERMITTENT_THRESHOLD = 0.6  

def load_data(filepath):
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
    df = df.sort_values(['sku_id', 'date']).reset_index(drop=True)
    return df

def aggregate_to_weekly(df):
    print("Aggregating daily data to weekly...")
    
    weekly_df = df.groupby(
        ['sku_id', pd.Grouper(key='date', freq='W')]
    )['demand'].sum().reset_index()
    
    weekly_df = weekly_df.sort_values(['sku_id', 'date']).reset_index(drop=True)
    
    print(f"  Daily records: {len(df)} → Weekly records: {len(weekly_df)}")
    
    return weekly_df

def preprocess_data(df):
    processed = []
    
    for sku in df['sku_id'].unique():
        sku_data = df[df['sku_id'] == sku].copy()
        
        min_date = sku_data['date'].min()
        max_date = sku_data['date'].max()
        all_weeks = pd.date_range(min_date, max_date, freq='W')
        
        sku_data = sku_data.set_index('date').reindex(all_weeks)
        sku_data['sku_id'] = sku
        sku_data['demand'] = sku_data['demand'].fillna(0)
        sku_data = sku_data.reset_index().rename(columns={'index': 'date'})
        
        processed.append(sku_data)
    
    return pd.concat(processed, ignore_index=True)

def classify_skus(df):
    classification = {}
    zero_ratios = {}
    
    for sku in df['sku_id'].unique():
        sku_data = df[df['sku_id'] == sku]['demand']
        zero_ratio = (sku_data == 0).sum() / len(sku_data)
        zero_ratios[sku] = zero_ratio
        
        if zero_ratio > INTERMITTENT_THRESHOLD:
            classification[sku] = 'intermittent'
        else:
            classification[sku] = 'smooth'
    
    return classification, zero_ratios

def calculate_sku_stats(df):
    sku_stats = {}
    
    for sku in df['sku_id'].unique():
        sku_demand = df[df['sku_id'] == sku]['demand']
        
        sku_stats[sku] = {
            'avg_demand': sku_demand.mean(),
            'std_demand': sku_demand.std() if len(sku_demand) > 1 else 0,
            'zero_ratio': (sku_demand == 0).sum() / len(sku_demand)
        }
    
    return sku_stats

def create_features(df, sku_stats):
    df = df.copy()
    
    le = LabelEncoder()
    df['sku_encoded'] = le.fit_transform(df['sku_id'])
    
    sku_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    
    df['week_of_year'] = df['date'].dt.isocalendar().week
    df['month'] = df['date'].dt.month
    
    df['week_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
    df['week_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)
    
    df['trend'] = df.groupby('sku_id').cumcount()
    
    featured = []
    for sku in df['sku_id'].unique():
        sku_df = df[df['sku_id'] == sku].copy()
        
        sku_df['lag_1'] = sku_df['demand'].shift(1)
        sku_df['lag_2'] = sku_df['demand'].shift(2)
        sku_df['lag_4'] = sku_df['demand'].shift(4)
        
        sku_df['rolling_mean_2'] = sku_df['demand'].shift(1).rolling(2, min_periods=1).mean()
        sku_df['rolling_mean_4'] = sku_df['demand'].shift(1).rolling(4, min_periods=1).mean()
        
        sku_df['sku_avg_demand'] = sku_stats[sku]['avg_demand']
        sku_df['sku_std_demand'] = sku_stats[sku]['std_demand']
        sku_df['sku_zero_ratio'] = sku_stats[sku]['zero_ratio']
        
        featured.append(sku_df)
    
    result = pd.concat(featured, ignore_index=True)
    result = result.fillna(0)
    
    return result, le, sku_mapping

def train_global_model(train_df, quantile):
    features = [
        'sku_encoded', 'week_of_year', 'month', 'week_sin', 'week_cos', 'trend',
        'lag_1', 'lag_2', 'lag_4',
        'rolling_mean_2', 'rolling_mean_4',
        'sku_avg_demand', 'sku_std_demand', 'sku_zero_ratio'
    ]
    
    train_df_clean = train_df.dropna(subset=['lag_4']).copy()
    
    if len(train_df_clean) < 10:
        return None
    
    train_df_clean = train_df_clean.sort_values('date')
    split_index = int(len(train_df_clean) * 0.9)
    train_data = train_df_clean.iloc[:split_index]
    val_data = train_df_clean.iloc[split_index:]
    
    X_train = train_data[features]
    y_train = train_data['demand']
    
    X_val = val_data[features]
    y_val = val_data['demand']
    
    weights = np.where(y_train > 0, 2.0, 1.0)
    
    model = lgb.LGBMRegressor(
        objective='quantile',
        alpha=quantile,
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        verbosity=-1
    )
    
    model.fit(
        X_train, y_train,
        sample_weight=weights,
        eval_set=[(X_val, y_val)] if len(val_data) > 0 else None,
        callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)] if len(val_data) > 0 else None
    )
    
    return model

def forecast_smooth_sku_global(sku_data, sku_stats, sku_encoded, forecast_weeks, 
                                model_p10, model_p50, model_p90):
    history = list(sku_data['demand'].values)
    
    last_date = sku_data['date'].max()
    forecasts = []
    
    sku_id = sku_data['sku_id'].iloc[0]
    sku_avg = sku_stats[sku_id]['avg_demand']
    sku_std = sku_stats[sku_id]['std_demand']
    sku_zero = sku_stats[sku_id]['zero_ratio']
    
    initial_trend = len(sku_data)
    
    for i in range(forecast_weeks):
        future_date = last_date + timedelta(weeks=i+1)
        
        lag_1 = history[-1] if len(history) >= 1 else 0
        lag_2 = history[-2] if len(history) >= 2 else 0
        lag_4 = history[-4] if len(history) >= 4 else 0
        
        rolling_2 = np.mean(history[-2:]) if len(history) >= 2 else np.mean(history)
        rolling_4 = np.mean(history[-4:]) if len(history) >= 4 else np.mean(history)
        
        week_of_year = future_date.isocalendar()[1]
        month = future_date.month
        
        week_sin = np.sin(2 * np.pi * week_of_year / 52)
        week_cos = np.cos(2 * np.pi * week_of_year / 52)
        
        trend = initial_trend + i
        
        X_pred = pd.DataFrame({
            'sku_encoded': [sku_encoded],
            'week_of_year': [week_of_year],
            'month': [month],
            'week_sin': [week_sin],
            'week_cos': [week_cos],
            'trend': [trend],
            'lag_1': [lag_1],
            'lag_2': [lag_2],
            'lag_4': [lag_4],
            'rolling_mean_2': [rolling_2],
            'rolling_mean_4': [rolling_4],
            'sku_avg_demand': [sku_avg],
            'sku_std_demand': [sku_std],
            'sku_zero_ratio': [sku_zero]
        })
        
        p10 = model_p10.predict(X_pred)[0]
        p50 = model_p50.predict(X_pred)[0]
        p90 = model_p90.predict(X_pred)[0]
        
        recent_mean = np.mean(history[-4:]) if len(history) >= 4 else np.mean(history)
        p50 = min(p50, 3 * max(recent_mean, 1))
        
        p10 = max(0, p10)
        p50 = max(0, p50)
        p90 = max(0, p90)
        
        p10 = min(p10, p50)
        p90 = max(p90, p50)
        
        spread_lower = abs(p50 - p10)
        spread_upper = abs(p90 - p50)
        
        p10 = max(0, p50 - 1.2 * spread_lower)
        p90 = p50 + 1.2 * spread_upper
        
        p90 = min(p90, 5 * max(p50, 1))
        
        forecasts.append({
            'date': future_date,
            'forecast_p10': p10,
            'forecast_p50': p50,
            'forecast_p90': p90
        })
        
        history.append(p50)
    
    return pd.DataFrame(forecasts)

def forecast_intermittent_sku(sku_data, forecast_weeks):
    demand = sku_data['demand'].values
    
    non_zero_demand = demand[demand > 0]
    
    if len(non_zero_demand) == 0:
        return pd.DataFrame({
            'date': pd.date_range(sku_data['date'].max() + timedelta(weeks=1), 
                                 periods=forecast_weeks, freq='W'),
            'forecast_p10': 0,
            'forecast_p50': 0,
            'forecast_p90': 0
        })
    
    recent_non_zero = non_zero_demand[-10:] if len(non_zero_demand) > 10 else non_zero_demand
    avg_demand_size = np.mean(recent_non_zero)
    std_demand_size = np.std(recent_non_zero) if len(recent_non_zero) > 1 else avg_demand_size * 0.3
    
    recent_demand = demand[-10:] if len(demand) >= 10 else demand
    prob_demand = (recent_demand > 0).mean()
    
    sku_id = sku_data['sku_id'].iloc[0]
    np.random.seed(hash(sku_id) % 10000)
    
    n_simulations = 100
    
    last_date = sku_data['date'].max()
    forecasts = []
    
    for i in range(forecast_weeks):
        future_date = last_date + timedelta(weeks=i+1)
        
        simulated_demands = []
        for _ in range(n_simulations):
            if np.random.random() < prob_demand:
                demand_sample = max(0, np.random.normal(avg_demand_size, std_demand_size))
                simulated_demands.append(demand_sample)
            else:
                simulated_demands.append(0)
        
        p10 = np.percentile(simulated_demands, 10)
        p50 = np.percentile(simulated_demands, 50)
        p90 = np.percentile(simulated_demands, 90)
        
        forecasts.append({
            'date': future_date,
            'forecast_p10': p10,
            'forecast_p50': p50,
            'forecast_p90': p90
        })
    
    return pd.DataFrame(forecasts)

def forecast_all_skus(df, classification, sku_stats, sku_mapping, forecast_weeks):
    print("Training global models on ALL SKUs...")
    
    model_p10 = train_global_model(df, 0.1)
    model_p50 = train_global_model(df, 0.5)
    model_p90 = train_global_model(df, 0.9)
    
    if model_p50 is not None:
        feature_names = [
            'sku_encoded', 'week_of_year', 'month', 'week_sin', 'week_cos', 'trend',
            'lag_1', 'lag_2', 'lag_4',
            'rolling_mean_2', 'rolling_mean_4',
            'sku_avg_demand', 'sku_std_demand', 'sku_zero_ratio'
        ]
        importances = model_p50.feature_importances_
        print("\nFeature Importances (P50 Model):")
        for name, importance in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True):
            print(f"  {name:<20}: {importance:.4f}")
        print()
    
    if model_p10 is None or model_p50 is None or model_p90 is None:
        print("Warning: Insufficient data for global models. Using intermittent method for all SKUs.")
        use_global = False
    else:
        use_global = True
    
    all_forecasts = []
    
    for sku in df['sku_id'].unique():
        sku_data = df[df['sku_id'] == sku].copy()
        sku_type = classification[sku]
        
        print(f"Forecasting SKU {sku} ({sku_type})...")
        
        if sku_type == 'smooth' and use_global:
            sku_encoded = sku_mapping[sku]
            forecast_df = forecast_smooth_sku_global(
                sku_data, sku_stats, sku_encoded, forecast_weeks,
                model_p10, model_p50, model_p90
            )
        else:
            forecast_df = forecast_intermittent_sku(sku_data, forecast_weeks)
        
        if forecast_df is not None:
            forecast_df['sku_id'] = sku
            all_forecasts.append(forecast_df)
    
    result = pd.concat(all_forecasts, ignore_index=True)
    result = result[['date', 'sku_id', 'forecast_p10', 'forecast_p50', 'forecast_p90']]
    
    return result

def generate_naive_baseline(df, forecast_weeks):
    baseline_forecasts = []
    
    for sku in df['sku_id'].unique():
        sku_data = df[df['sku_id'] == sku].copy()
        last_demand = sku_data['demand'].iloc[-1]
        last_date = sku_data['date'].max()
        
        forecast_dates = pd.date_range(last_date + timedelta(weeks=1), 
                                       periods=forecast_weeks, freq='W')
        
        for date in forecast_dates:
            baseline_forecasts.append({
                'date': date,
                'sku_id': sku,
                'naive_forecast': last_demand
            })
    
    return pd.DataFrame(baseline_forecasts)

def evaluate_forecasts(forecast_df, actual_df, baseline_df=None):
    actual_df['date'] = pd.to_datetime(actual_df['date'], format='%d-%m-%Y')
    actual_weekly = actual_df.groupby(
        ['sku_id', pd.Grouper(key='date', freq='W')]
    )['demand'].sum().reset_index()
    
    merged = forecast_df.merge(actual_weekly, on=['date', 'sku_id'], how='inner')
    
    if len(merged) == 0:
        print("No matching data for evaluation!")
        return
    
    merged = merged.sort_values('date')
    split_idx = int(len(merged) * 0.8)
    train_eval = merged.iloc[:split_idx]
    test_eval = merged.iloc[split_idx:]
    
    print("\n" + "="*60)
    print("TIME-BASED EVALUATION SPLIT")
    print("="*60)
    print(f"Training period: {len(train_eval)} weeks")
    print(f"Testing period:  {len(test_eval)} weeks")
    print("="*60)
    
    if len(test_eval) == 0:
        print("No test data available. Using full data for evaluation.")
        test_eval = merged
    
    mae_model = mean_absolute_error(test_eval['demand'], test_eval['forecast_p50'])
    rmse_model = np.sqrt(mean_squared_error(test_eval['demand'], test_eval['forecast_p50']))
    
    denominator = np.sum(test_eval['demand'])
    if denominator == 0:
        wape_model = 0
    else:
        wape_model = np.sum(np.abs(test_eval['demand'] - test_eval['forecast_p50'])) / denominator * 100
    
    within_range = ((test_eval['demand'] >= test_eval['forecast_p10']) & 
                   (test_eval['demand'] <= test_eval['forecast_p90']))
    coverage = within_range.sum() / len(test_eval) * 100
    
    if baseline_df is not None:
        baseline_merged = baseline_df.merge(actual_weekly, on=['date', 'sku_id'], how='inner')
        baseline_test = baseline_merged[baseline_merged['date'].isin(test_eval['date'])]
        
        if len(baseline_test) > 0:
            mae_baseline = mean_absolute_error(baseline_test['demand'], baseline_test['naive_forecast'])
            rmse_baseline = np.sqrt(mean_squared_error(baseline_test['demand'], baseline_test['naive_forecast']))
            
            denom_baseline = np.sum(baseline_test['demand'])
            if denom_baseline == 0:
                wape_baseline = 0
            else:
                wape_baseline = np.sum(np.abs(baseline_test['demand'] - baseline_test['naive_forecast'])) / denom_baseline * 100
    
    print("\n" + "="*60)
    print("FORECAST EVALUATION METRICS (TEST SET)")
    print("="*60)
    print(f"Model MAE:      {mae_model:.2f}")
    print(f"Model RMSE:     {rmse_model:.2f}")
    print(f"Model WAPE:     {wape_model:.2f}%")
    print(f"Coverage:       {coverage:.2f}% (P10-P90)")
    
    if baseline_df is not None and len(baseline_test) > 0:
        print("\n" + "-"*60)
        print("BASELINE (NAIVE) COMPARISON")
        print("-"*60)
        print(f"Baseline MAE:   {mae_baseline:.2f}")
        print(f"Baseline RMSE:  {rmse_baseline:.2f}")
        print(f"Baseline WAPE:  {wape_baseline:.2f}%")
        print("\n" + "-"*60)
        print("IMPROVEMENT OVER BASELINE")
        print("-"*60)
        print(f"MAE Improvement:  {((mae_baseline - mae_model) / mae_baseline * 100):.2f}%")
        print(f"RMSE Improvement: {((rmse_baseline - rmse_model) / rmse_baseline * 100):.2f}%")
        print(f"WAPE Improvement: {((wape_baseline - wape_model) / wape_baseline * 100):.2f}%")
    
    print("="*60)
    
    print("\nPER-SKU METRICS (TEST SET):")
    print("-" * 60)
    print(f"{'SKU':<15} {'MAE':<10} {'RMSE':<10} {'WAPE%':<10} {'Coverage%':<12}")
    print("-" * 60)
    
    for sku in sorted(test_eval['sku_id'].unique()):
        sku_data = test_eval[test_eval['sku_id'] == sku]
        
        mae_sku = mean_absolute_error(sku_data['demand'], sku_data['forecast_p50'])
        rmse_sku = np.sqrt(mean_squared_error(sku_data['demand'], sku_data['forecast_p50']))
        
        denom_sku = np.sum(sku_data['demand'])
        if denom_sku == 0:
            wape_sku = 0
        else:
            wape_sku = np.sum(np.abs(sku_data['demand'] - sku_data['forecast_p50'])) / denom_sku * 100
        
        within_sku = ((sku_data['demand'] >= sku_data['forecast_p10']) & 
                     (sku_data['demand'] <= sku_data['forecast_p90']))
        coverage_sku = within_sku.sum() / len(sku_data) * 100
        
        print(f"{str(sku):<15} {mae_sku:<10.2f} {rmse_sku:<10.2f} {wape_sku:<10.2f} {coverage_sku:<12.2f}")
    
    print("-" * 60 + "\n")

def main():
    """Main pipeline with improved evaluation"""
    print("="*60)
    print("PHASE-1 GLOBAL DEMAND FORECASTING SYSTEM")
    print("Weekly Aggregation | Global Model | Uncertainty-Aware")
    print("="*60 + "\n")
    
    print("Loading data...")
    df = load_data(INPUT_FILE)
    
    df = aggregate_to_weekly(df)
    
    print("Preprocessing...")
    df = preprocess_data(df)
    
    print("Classifying SKUs...")
    classification, zero_ratios = classify_skus(df)
    smooth_count = sum(1 for v in classification.values() if v == 'smooth')
    intermittent_count = len(classification) - smooth_count
    print(f"  Smooth: {smooth_count}, Intermittent: {intermittent_count}")
    
    print("Calculating SKU statistics...")
    sku_stats = calculate_sku_stats(df)
    
    print("Creating features...")
    df, label_encoder, sku_mapping = create_features(df, sku_stats)
    
    print("Generating naive baseline...")
    baseline_forecast = generate_naive_baseline(df, FORECAST_WEEKS)
    
    print("Generating model forecasts...")
    forecasts = forecast_all_skus(df, classification, sku_stats, sku_mapping, FORECAST_WEEKS)
    
    print(f"Saving forecasts to {OUTPUT_FILE}...")
    forecasts.to_csv(OUTPUT_FILE, index=False)
    
    try:
        excel_file = OUTPUT_FILE.replace('.csv', '.xlsx')
        forecasts.to_excel(excel_file, index=False, engine='openpyxl')
        print(f"Forecast also saved as Excel → {excel_file}")
    except ImportError:
        print("Warning: Excel export failed. Install openpyxl: pip install openpyxl")
    except Exception as e:
        print(f"Warning: Excel export failed - {str(e)}")
    
    print("Evaluating forecasts...")
    try:
        actual_df = load_data(ACTUAL_FILE)
        evaluate_forecasts(forecasts, actual_df, baseline_forecast)
    except FileNotFoundError:
        print(f"Actual file {ACTUAL_FILE} not found. Skipping evaluation.")
    
    print("="*60)
    print("FORECASTING COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
