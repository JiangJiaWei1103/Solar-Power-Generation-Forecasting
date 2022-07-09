echo "Start preparing data..."

# Process weather data
echo "Processing weather data...\n"
python -m data_preparation.proc_weather

# Process weather data
echo "Processing air quality data...\n"
python -m data_preparation.proc_aq

# Process pv data
echo "Sraping pv data...\n"
python -m data_preparation.scrape_pv

# Merge data
echo "Merge processed data with training and testing sets...\n"
python -m data_preparation.merge_data

# Handle missing issue
echo "Handling missing issue..."
python -m data_preparation.impute

echo "\nSuccess."
