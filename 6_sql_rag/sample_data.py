import random
from datetime import datetime, timedelta
import csv
import os
import sqlite3

def generate_tesla_motors_dataset(num_records=5000):
    models = {
        "Tesla Model 3": {"price_range": (38990, 53990), "mileage_range": (267, 333)},
        "Tesla Model Y": {"price_range": (43990, 69990), "mileage_range": (260, 330)},
        "Tesla Model S": {"price_range": (74990, 129990), "mileage_range": (320, 405)},
        "Tesla Model X": {"price_range": (79990, 139990), "mileage_range": (305, 348)},
        "Tesla Cybertruck": {"price_range": (49900, 79900), "mileage_range": (250, 500)},
        "Tesla Roadster": {"price_range": (200000, 250000), "mileage_range": (400, 620)}
    }
    colors = ["Pearl White", "Solid Black", "Deep Blue Metallic", "Midnight Silver Metallic", "Red Multi-Coat", "Ultra Red", "Stealth Grey"]
    fuel_types = {"Electric": 1.0}  # Tesla only makes electric vehicles
    transmission_types = {"Automatic": 1.0}  # Tesla only has automatic transmission
    states = [
        "California", "Texas", "Florida", "New York", "Illinois", "Washington", "Massachusetts",
        "New Jersey", "Colorado", "Virginia", "Arizona", "Nevada", "Oregon", "Connecticut", "Maryland"
    ]
    variants = ["Standard Range", "Long Range", "Performance", "Plaid"]
    
    dataset = []

    for _ in range(num_records):
        model = random.choice(list(models.keys()))
        color = random.choice(colors)
        fuel_type = "Electric"
        transmission = "Automatic"
        variant = random.choice(variants)
        
        price_range = models[model]["price_range"]
        price = round(random.uniform(*price_range), 2)
        
        mileage_range = models[model]["mileage_range"]
        mileage = round(random.uniform(*mileage_range), 1)
        
        manufacture_date = datetime.now() - timedelta(days=random.randint(1, 730))  # Up to 2 years old
        sale_date = manufacture_date + timedelta(days=random.randint(1, 90))  # Up to 3 months to sell
        state = random.choice(states)
        
        # Additional nuanced fields
        engine_capacity = 0  # Electric vehicles don't have engine capacity
        battery_capacity = round(random.uniform(60, 100), 1)  # kWh
        charging_time = round(random.uniform(0.5, 12), 1)  # Hours (varies by charger type)
        seating_capacity = 7 if model in ["Tesla Model X", "Tesla Model Y"] else 5
        ground_clearance = round(random.uniform(140, 170), 1)
        boot_space = random.randint(
            425, 900 if model in ["Tesla Model Y", "Tesla Model X"] else 650
        )
        
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

def save_to_csv(data, filename="tesla_motors_data.csv"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, filename)
    
    with open(file_path, 'w', newline='') as csvfile:
        fieldnames = data[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in data:
            writer.writerow(row)
    
    return file_path

def save_to_sqlite(data, db_name="tesla_motors_data.db"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(script_dir, db_name)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS tesla_motors (
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
        INSERT INTO tesla_motors (
            model, variant, color, fuel_type, transmission, price, manufacture_date, sale_date,
            state, mileage, engine_capacity, battery_capacity, charging_time, seating_capacity,
            ground_clearance, boot_space
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', tuple(record.values()))
    
    conn.commit()
    conn.close()
    
    return db_path

# Generate the dataset
tesla_motors_data = generate_tesla_motors_dataset(5000)

# Save the dataset to a CSV file
csv_file_path = save_to_csv(tesla_motors_data)

# Save the dataset to a SQLite database
db_file_path = save_to_sqlite(tesla_motors_data)

print(f"Dataset has been saved to CSV: {csv_file_path}")
print(f"Dataset has been saved to SQLite database: {db_file_path}")

# Print the first 5 records as a sample
for i, record in enumerate(tesla_motors_data[:5], 1):
    print(f"\nRecord {i}:")
    for key, value in record.items():
        print(f"  {key}: {value}")

print(f"\nTotal records generated and saved: {len(tesla_motors_data)}")