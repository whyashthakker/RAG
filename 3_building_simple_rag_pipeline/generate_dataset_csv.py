import random
from datetime import datetime, timedelta
import csv
import os

def generate_tata_motors_dataset(num_records=500):
    models = ["Tata Nexon", "Tata Altroz", "Tata Harrier", "Tata Safari", "Tata Punch", "Tata Tiago", "Tata Tigor"]
    colors = ["Red", "Blue", "White", "Black", "Silver", "Grey", "Orange"]
    fuel_types = ["Petrol", "Diesel", "Electric", "CNG"]
    transmission_types = ["Manual", "Automatic"]
    states = ["Maharashtra", "Gujarat", "Delhi", "Tamil Nadu", "Karnataka", "Uttar Pradesh", "West Bengal"]

    dataset = []

    for _ in range(num_records):
        model = random.choice(models)
        color = random.choice(colors)
        fuel_type = random.choice(fuel_types)
        transmission = random.choice(transmission_types)
        price = round(random.uniform(500000, 2500000), 2)
        manufacture_date = datetime.now() - timedelta(days=random.randint(1, 365))
        sale_date = manufacture_date + timedelta(days=random.randint(1, 90))
        state = random.choice(states)
        mileage = round(random.uniform(10, 25), 1)
        
        record = {
            "model": model,
            "color": color,
            "fuel_type": fuel_type,
            "transmission": transmission,
            "price": price,
            "manufacture_date": manufacture_date.strftime("%Y-%m-%d"),
            "sale_date": sale_date.strftime("%Y-%m-%d"),
            "state": state,
            "mileage": mileage
        }
        
        dataset.append(record)
    
    return dataset

def save_to_csv(data, filename="tata_motors_data.csv"):
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create the full path for the CSV file
    file_path = os.path.join(script_dir, filename)
    
    # Write the data to the CSV file
    with open(file_path, 'w', newline='') as csvfile:
        fieldnames = data[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in data:
            writer.writerow(row)
    
    return file_path

# Generate the dataset
tata_motors_data = generate_tata_motors_dataset()

# Save the dataset to a CSV file
csv_file_path = save_to_csv(tata_motors_data)

print(f"Dataset has been saved to: {csv_file_path}")

# Print the first 5 records as a sample
for i, record in enumerate(tata_motors_data[:5], 1):
    print(f"\nRecord {i}:")
    for key, value in record.items():
        print(f"  {key}: {value}")

print(f"\nTotal records generated and saved: {len(tata_motors_data)}")