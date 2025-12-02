import numpy as np
import torch
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

def calculate_metrics(model, data_loader, device):
    model.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for batch in data_loader:
            features = batch['features'].to(device)
            prices = batch['prices'].to(device)

            pred_prices = get_model_prices_from_batch(model, batch, device)
            
            y_true.extend(prices.cpu().numpy())
            y_pred.extend(pred_prices.cpu().numpy())
            
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape}

def quantify_arbitrage_violations(model, config, device, grid_size=50):

    model.eval()
    moneyness = np.linspace(0.7, 1.3, grid_size)
    maturities = np.linspace(0.1, 2.0, grid_size)

    inputs.requires_grad = True
    
    sigma = model(inputs)

    grads_w = torch.autograd.grad(w.sum(), inputs, create_graph=True)[0]
    dw_dT = grads_w[:, 1]
    calendar_violations = (dw_dT < -1e-6).sum().item()
    calendar_ratio = calendar_violations / inputs.shape[0]
    
    return {
        "Calendar_Violation_Ratio": calendar_ratio,
    }


def get_model_prices_from_batch(model, batch, device):

    from black_scholes import bs_price
    iv = model(batch['features'].to(device)).flatten()
    S = batch['spots'].to(device)
    K = batch['strikes'].to(device)
    T = batch['maturities'].to(device)
    r = batch['rates'].to(device)
    q = batch['dividends'].to(device)
    flags = batch['cp_flags'].to(device)
    
    return bs_price(S, K, T, r, q, iv, flags)