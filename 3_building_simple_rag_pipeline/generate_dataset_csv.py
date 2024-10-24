import random
from datetime import datetime, timedelta
import csv
import os

def generate_tesla_motors_dataset(num_records=500):
    models = ["Tesla Model S", "Tesla Model 3", "Tesla Model X", "Tesla Model Y", "Tesla Cybertruck", "Tesla Roadster"]
    colors = ["Red", "Blue", "White", "Black", "Silver", "Grey", "Midnight Silver"]
    fuel_types = ["Electric"]  # Tesla only makes electric vehicles
    transmission_types = ["Automatic"]  # Tesla only has automatic transmission
    states = ["California", "Texas", "Florida", "New York", "Washington", "Illinois", "Massachusetts"]

    dataset = []

    for _ in range(num_records):
        model = random.choice(models)
        color = random.choice(colors)
        fuel_type = random.choice(fuel_types)
        transmission = random.choice(transmission_types)
        price = round(random.uniform(35000, 150000), 2)  # Adjusted price range for Tesla vehicles
        manufacture_date = datetime.now() - timedelta(days=random.randint(1, 365))
        sale_date = manufacture_date + timedelta(days=random.randint(1, 90))
        state = random.choice(states)
        mileage = round(random.uniform(250, 400), 1)  # Adjusted for electric vehicle range
        
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

def save_to_csv(data, filename="tesla_motors_data.csv"):
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
tesla_motors_data = generate_tesla_motors_dataset()

# Save the dataset to a CSV file
csv_file_path = save_to_csv(tesla_motors_data)

print(f"Dataset has been saved to: {csv_file_path}")

# Print the first 5 records as a sample
for i, record in enumerate(tesla_motors_data[:5], 1):
    print(f"\nRecord {i}:")
    for key, value in record.items():
        print(f"  {key}: {value}")

print(f"\nTotal records generated and saved: {len(tesla_motors_data)}")