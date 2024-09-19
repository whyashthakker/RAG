import random
from datetime import datetime, timedelta
import csv
import os
import sqlite3

def generate_tata_motors_dataset(num_records=5000):
    models = {
        "Tata Nexon": {"price_range": (700000, 1400000), "mileage_range": (16, 22)},
        "Tata Altroz": {"price_range": (600000, 950000), "mileage_range": (19, 25)},
        "Tata Harrier": {"price_range": (1400000, 2400000), "mileage_range": (14, 17)},
        "Tata Safari": {"price_range": (1500000, 2500000), "mileage_range": (14, 16.5)},
        "Tata Punch": {"price_range": (600000, 1000000), "mileage_range": (18, 24)},
        "Tata Tiago": {"price_range": (500000, 800000), "mileage_range": (19, 26)},
        "Tata Tigor": {"price_range": (600000, 950000), "mileage_range": (19, 25)},
    }
    colors = ["Red", "Blue", "White", "Black", "Silver", "Grey", "Orange", "Green", "Bronze", "Copper"]
    fuel_types = {"Petrol": 0.45, "Diesel": 0.35, "Electric": 0.15, "CNG": 0.05}
    transmission_types = {"Manual": 0.6, "Automatic": 0.4}
    states = [
        "Maharashtra", "Gujarat", "Delhi", "Tamil Nadu", "Karnataka", "Uttar Pradesh", "West Bengal",
        "Rajasthan", "Madhya Pradesh", "Telangana", "Kerala", "Bihar", "Punjab", "Haryana", "Odisha"
    ]
    variants = ["XE", "XM", "XZ", "XZ+", "XZ+ Lux"]
    
    dataset = []

    for _ in range(num_records):
        model = random.choice(list(models.keys()))
        color = random.choice(colors)
        fuel_type = random.choices(list(fuel_types.keys()), weights=list(fuel_types.values()))[0]
        transmission = random.choices(list(transmission_types.keys()), weights=list(transmission_types.values()))[0]
        variant = random.choice(variants)
        
        price_range = models[model]["price_range"]
        price = round(random.uniform(*price_range), 2)
        
        mileage_range = models[model]["mileage_range"]
        mileage = round(random.uniform(*mileage_range), 1)
        
        manufacture_date = datetime.now() - timedelta(days=random.randint(1, 730))  # Up to 2 years old
        sale_date = manufacture_date + timedelta(days=random.randint(1, 180))  # Up to 6 months to sell
        state = random.choice(states)
        
        # Additional nuanced fields
        engine_capacity = round(random.uniform(1.0, 2.0), 1) if fuel_type != "Electric" else 0
        battery_capacity = round(random.uniform(30, 50), 1) if fuel_type == "Electric" else 0
        charging_time = round(random.uniform(6, 10), 1) if fuel_type == "Electric" else 0
        seating_capacity = random.choice([5, 7]) if model in ["Tata Safari", "Tata Harrier"] else 5
        ground_clearance = round(random.uniform(165, 210), 1)
        boot_space = random.randint(250, 450)
        
        record = {
            "model": model,
            "variant": variant,
            "color": color,
            "fuel_type": fuel_type,
            "transmission": transmission,
            "price": price,
            "manufacture_date": manufacture_date.strftime("%Y-%m-%d"),
            "sale_date": sale_date.strftime("%Y-%m-%d"),
            "state": state,
            "mileage": mileage,
            "engine_capacity": engine_capacity,
            "battery_capacity": battery_capacity,
            "charging_time": charging_time,
            "seating_capacity": seating_capacity,
            "ground_clearance": ground_clearance,
            "boot_space": boot_space
        }
        
        dataset.append(record)
    
    return dataset

def save_to_csv(data, filename="tata_motors_data.csv"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, filename)
    
    with open(file_path, 'w', newline='') as csvfile:
        fieldnames = data[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in data:
            writer.writerow(row)
    
    return file_path

def save_to_sqlite(data, db_name="tata_motors_data.db"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(script_dir, db_name)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS tata_motors (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model TEXT,
        variant TEXT,
        color TEXT,
        fuel_type TEXT,
        transmission TEXT,
        price REAL,
        manufacture_date TEXT,
        sale_date TEXT,
        state TEXT,
        mileage REAL,
        engine_capacity REAL,
        battery_capacity REAL,
        charging_time REAL,
        seating_capacity INTEGER,
        ground_clearance REAL,
        boot_space INTEGER
    )
    ''')
    
    # Insert data
    for record in data:
        cursor.execute('''
        INSERT INTO tata_motors (
            model, variant, color, fuel_type, transmission, price, manufacture_date, sale_date,
            state, mileage, engine_capacity, battery_capacity, charging_time, seating_capacity,
            ground_clearance, boot_space
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', tuple(record.values()))
    
    conn.commit()
    conn.close()
    
    return db_path

# Generate the dataset
tata_motors_data = generate_tata_motors_dataset(5000)

# Save the dataset to a CSV file
csv_file_path = save_to_csv(tata_motors_data)

# Save the dataset to a SQLite database
db_file_path = save_to_sqlite(tata_motors_data)

print(f"Dataset has been saved to CSV: {csv_file_path}")
print(f"Dataset has been saved to SQLite database: {db_file_path}")

# Print the first 5 records as a sample
for i, record in enumerate(tata_motors_data[:5], 1):
    print(f"\nRecord {i}:")
    for key, value in record.items():
        print(f"  {key}: {value}")

print(f"\nTotal records generated and saved: {len(tata_motors_data)}")