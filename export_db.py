import sqlite3
import pandas as pd

def export():
    conn = sqlite3.connect("pcosense.db")
    
    # Export users
    try:
        users_df = pd.read_sql_query("SELECT id, username, name, created_at FROM users", conn)
        users_df.to_csv("users_export.csv", index=False)
        print("? Users exported successfully to users_export.csv")
    except Exception as e:
        print("Error exporting users:", e)
        
    # Export predictions
    try:
        preds_df = pd.read_sql_query("SELECT * FROM predictions", conn)
        preds_df.to_csv("predictions_export.csv", index=False)
        print("? Predictions history exported successfully to predictions_export.csv")
    except Exception as e:
        print("Error exporting predictions:", e)
        
    conn.close()

if __name__ == "__main__":
    export()

