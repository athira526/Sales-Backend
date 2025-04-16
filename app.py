from flask import Flask, request, jsonify
import pandas as pd
import os
from datetime import datetime
import torch
import numpy as np

app = Flask(__name__)

UPLOAD_FOLDER = 'Uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the scripted TFT model
try:
    model = torch.jit.load("tft_traced_model.pt")
    model.eval()
    print("✅ TorchScript model loaded successfully.")
except Exception as e:
    print("❌ Failed to load model:", str(e))
    raise e

def validate_excel_data(df):
    required_columns = ['date', 'history', 'onpromotion', 'is_holiday', 'transactions', 'store_nbr', 'item_nbr']
    return all(col in df.columns for col in required_columns)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if file and file.filename.endswith('.xlsx'):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"upload_{timestamp}.xlsx"
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        try:
            df = pd.read_excel(file_path)
            if not validate_excel_data(df):
                return jsonify({"error": "Invalid Excel format"}), 400
            
            row_count = len(df)
            return jsonify({
                "message": "File uploaded and validated successfully",
                "filename": filename,
                "row_count": row_count
            }), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "Invalid file format, only .xlsx allowed"}), 400

@app.route('/forecast', methods=['POST'])
def forecast():
    try:
        # Get JSON data from request
        data = request.get_json()
        forecast_days = min(data.get('forecast_days', 7), 30)  # Cap at 30 days
        custom_is_holiday = data.get('is_holiday', None)
        custom_onpromotion = data.get('onpromotion', None)
        store_name = data.get('store_name', 'Store 1')
        item_name = data.get('item_name', 'Item 1')

        # Find latest uploaded file
        upload_dir = UPLOAD_FOLDER
        files = [f for f in os.listdir(upload_dir) if f.endswith('.xlsx')]
        if not files:
            return jsonify({"error": "No uploaded files found"}), 404
        
        latest_file = max(
            [os.path.join(upload_dir, f) for f in files],
            key=os.path.getctime
        )
        
        # Read Excel
        df = pd.read_excel(latest_file)
        if not validate_excel_data(df):
            return jsonify({"error": "Invalid data in uploaded file"}), 400
        
        # Sort by date
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # Handle NaN and ensure data types
        df['store_nbr'] = df['store_nbr'].fillna(1).astype(int)
        df['item_nbr'] = df['item_nbr'].fillna(1).astype(int)
        df['onpromotion'] = df['onpromotion'].fillna(0).astype(int)
        df['is_holiday'] = df['is_holiday'].fillna(0).astype(int)
        df['transactions'] = df['transactions'].fillna(0).astype(float)
        df['history'] = df['history'].fillna(0).astype(float)
        
        # Extract inputs
        store_nbr = df['store_nbr'].iloc[-1]
        item_nbr = df['item_nbr'].iloc[-1]
        history = df['history'].tail(30).tolist()
        transactions = df['transactions'].tail(37).tolist()
        
        # Use custom toggles if provided, else fall back to Excel
        is_holiday = custom_is_holiday if custom_is_holiday and len(custom_is_holiday) == 37 else df['is_holiday'].tail(37).tolist()
        onpromotion = custom_onpromotion if custom_onpromotion and len(custom_onpromotion) == 37 else df['onpromotion'].tail(37).tolist()

        # Validate lengths
        if len(history) < 30:
            return jsonify({"error": "Insufficient history data: need 30 days"}), 400
        history = history[-30:]
        
        # Extend arrays to support up to 30 days
        max_horizon = 30
        time_steps = 37 + max_horizon
        transactions = transactions[-37:] + [transactions[-1]] * max_horizon
        is_holiday = is_holiday[-37:] + [is_holiday[-1]] * max_horizon
        onpromotion = onpromotion[-37:] + [onpromotion[-1]] * max_horizon

        encoder_len = 30
        decoder_len = 1

        # Initialize predictions
        predictions = {
            "p10": [],
            "p50": [],
            "p90": []
        }
        current_history = history.copy()
        current_onpromotion = onpromotion.copy()
        current_is_holiday = is_holiday.copy()
        current_transactions = transactions.copy()

        for day in range(forecast_days):
            # Lengths
            encoder_lengths = torch.tensor([encoder_len], dtype=torch.long)
            decoder_lengths = torch.tensor([decoder_len], dtype=torch.long)

            # Continuous inputs
            history_tensor = torch.tensor(current_history[-encoder_len:], dtype=torch.float).unsqueeze(0)
            transactions_tensor = torch.tensor(current_transactions[day:37+day], dtype=torch.float).unsqueeze(0)
            time_idx = torch.arange(day, 37+day, dtype=torch.float).unsqueeze(0)
            day_of_week = torch.tensor([(i % 7) for i in range(day, 37+day)], dtype=torch.float).unsqueeze(0)
            month = torch.tensor([((i % 12) + 1) for i in range(day, 37+day)], dtype=torch.float).unsqueeze(0)
            dummy1 = torch.zeros(1, 37, dtype=torch.float)
            dummy2 = torch.zeros(1, 37, dtype=torch.float)
            dummy3 = torch.zeros(1, 37, dtype=torch.float)
            dummy4 = torch.zeros(1, 37, dtype=torch.float)
            dummy5 = torch.zeros(1, 37, dtype=torch.float)

            encoder_cont = torch.stack([
                history_tensor[:, :encoder_len],
                transactions_tensor[:, :encoder_len],
                time_idx[:, :encoder_len],
                day_of_week[:, :encoder_len],
                month[:, :encoder_len],
                dummy1[:, :encoder_len],
                dummy2[:, :encoder_len],
                dummy3[:, :encoder_len],
                dummy4[:, :encoder_len],
                dummy5[:, :encoder_len]
            ], dim=-1)

            decoder_cont = torch.stack([
                history_tensor[:, -1:],
                transactions_tensor[:, encoder_len:encoder_len+decoder_len],
                time_idx[:, encoder_len:encoder_len+decoder_len],
                day_of_week[:, encoder_len:encoder_len+decoder_len],
                month[:, encoder_len:encoder_len+decoder_len],
                dummy1[:, encoder_len:encoder_len+decoder_len],
                dummy2[:, encoder_len:encoder_len+decoder_len],
                dummy3[:, encoder_len:encoder_len+decoder_len],
                dummy4[:, encoder_len:encoder_len+decoder_len],
                dummy5[:, encoder_len:encoder_len+decoder_len]
            ], dim=-1)

            # Categorical inputs
            store_tensor = torch.tensor([[store_nbr] * 37], dtype=torch.long)
            item_tensor = torch.tensor([[item_nbr] * 37], dtype=torch.long)
            promo_tensor = torch.tensor(current_onpromotion[day:37+day], dtype=torch.long).unsqueeze(0)
            holiday_tensor = torch.tensor(current_is_holiday[day:37+day], dtype=torch.long).unsqueeze(0)

            encoder_cat = torch.stack([
                store_tensor[:, :encoder_len],
                item_tensor[:, :encoder_len],
                promo_tensor[:, :encoder_len],
                holiday_tensor[:, :encoder_len]
            ], dim=-1)

            decoder_cat = torch.stack([
                store_tensor[:, encoder_len:encoder_len+decoder_len],
                item_tensor[:, encoder_len:encoder_len+decoder_len],
                promo_tensor[:, encoder_len:encoder_len+decoder_len],
                holiday_tensor[:, encoder_len:encoder_len+decoder_len]
            ], dim=-1)

            # Target scale
            history_array = torch.tensor(current_history, dtype=torch.float)
            target_mean = history_array.mean()
            target_std = history_array.std() if history_array.std() > 0 else torch.tensor(1.0)
            target_scale = torch.tensor([[target_mean, target_std]], dtype=torch.float)

            # Model inputs
            model_input = {
                "encoder_cont": encoder_cont,
                "decoder_cont": decoder_cont,
                "encoder_cat": encoder_cat,
                "decoder_cat": decoder_cat,
                "encoder_lengths": encoder_lengths,
                "decoder_lengths": decoder_lengths,
                "target_scale": target_scale
            }

            # Run model
            with torch.no_grad():
                output = model(model_input)

            # Extract quantiles
            p10 = output[0][:, :, 1].squeeze().item()
            p50 = output[0][:, :, 3].squeeze().item()
            p90 = output[0][:, :, 5].squeeze().item()
            predictions["p10"].append(p10)
            predictions["p50"].append(p50)
            predictions["p90"].append(p90)

            # Update history with p50
            current_history.append(p50)
            current_history = current_history[1:]

        # Suggestions with store_name and item_name
        avg_forecast = np.mean(predictions["p50"])
        suggestions = [
            {
                "type": "stock_adjustment",
                "message": f"Prepare stock for ~{round(avg_forecast)} units/day of {item_name} at {store_name}.",
                "confidence": 0.9
            }
        ]

        return jsonify({
            "forecast": {
                "p10": predictions["p10"],
                "p50": predictions["p50"],
                "p90": predictions["p90"]
            },
            "suggestions": suggestions,
            "store_name": store_name,
            "item_name": item_name
        }), 200

    except Exception as e:
        print("❌ Error:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)