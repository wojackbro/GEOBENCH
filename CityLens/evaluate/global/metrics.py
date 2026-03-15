import json
import csv
import argparse
import os
import re
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    from path_config import results_path, summary_csv_path
    _USE_PATH_CONFIG = True
except ImportError:
    _USE_PATH_CONFIG = False

def compute_regression_metrics(pred_list, true_list):
    mse = mean_squared_error(true_list, pred_list)
    mae = mean_absolute_error(true_list, pred_list)
    r2 = r2_score(true_list, pred_list)
    rmse = mean_squared_error(true_list, pred_list, squared=False)
    return mse, mae, r2, rmse

def extract_float(value):
    if isinstance(value, (float, int)):
        return float(value)
    if isinstance(value, str):
        value_str = value.replace(",", "").replace("$", "").replace("£", "").strip()
        if len(value_str) <= 100:
            try:
                return float(value_str)
            except ValueError:
                pass
        # Fallback: extract first number from text (for long/verbose model responses)
        match = re.search(r"-?[\d,]+\.?\d*", value)
        if match:
            try:
                return float(match.group().replace(",", ""))
            except ValueError:
                pass
    return None


def process_json(json_path, prompt_type):
    with open(json_path, 'r') as f:
        data = json.load(f)

    true_vals = []
    pred_vals = []

    for item in data:
        if prompt_type == "simple":
            true_val = extract_float(item.get('reference'))
        elif prompt_type == "normalized":
            true_val = extract_float(item.get('reference_normalized') or item.get('reference'))
        else:
            true_val = extract_float(item.get('reference'))
        pred_val = extract_float(item.get('response'))
        if true_val is None or pred_val is None:
            print(f"Warning: Invalid data found in item: {item['response']}")
            continue

        true_vals.append(true_val)
        pred_vals.append(pred_val)
    print("length of vals: ", len(true_vals), len(pred_vals))
    return true_vals, pred_vals

def write_csv(output_path, city, model, prompt_type, mse, mae, r2, rmse):
    fieldnames = ['city', 'model', "prompt_type", 'MSE', 'MAE', 'R2', 'RMSE']
    write_header = not os.path.exists(output_path)

    with open(output_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({
            'city': city,
            'model': model,
            'prompt_type': prompt_type,
            'MSE': mse,
            'MAE': mae,
            'R2': r2,
            'RMSE': rmse
        })

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--city_name", default="NewYork", help="City name")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-VL-32B-Instruct", help="Model name")
    parser.add_argument('--task_name', type=str, default='gdp', choices=['gdp', "pop", "acc2health", "carbon", "build_height"], help='task name')
    parser.add_argument("--prompt_type", default="simple", help="Prompt type")

    args = parser.parse_args()
    model_name_full = args.model_name.replace("/", "_")
    if _USE_PATH_CONFIG:
        response_path = results_path(args.task_name, args.city_name, args.model_name, args.prompt_type)
        summary_path = summary_csv_path()
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    else:
        response_path = f''
        summary_path = f''
    if not response_path:
        data_root = os.environ.get("CITYLENS_DATA_ROOT", "")
        if data_root:
            response_path = os.path.join(data_root, "Results", f"{args.task_name}_{args.city_name}_{model_name_full}_{args.prompt_type}.json")
            summary_path = os.path.join(data_root, "Results", "summary.csv")
            os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    y_true, y_pred = process_json(response_path, args.prompt_type)
    if len(y_true) == 0 or len(y_pred) == 0:
        print("No valid data found.")
    else:
        mse, mae, r2, rmse = compute_regression_metrics(y_pred, y_true)
        print("r2:", r2)
        write_csv(summary_path, args.city_name, args.model_name, args.prompt_type, mse, mae, r2, rmse)
        print(f"Metrics written for city={args.city_name}, model={args.model_name} to {summary_path}")
