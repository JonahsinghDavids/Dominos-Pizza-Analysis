import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_percentage_error

ingredients_df = pd.read_csv(r"C:\Users\Lenovo\Downloads\Pizza_ingredients - Pizza_ingredients.csv")
sales_df = pd.read_csv(r"C:\Users\Lenovo\Downloads\Pizza_Sale - pizza_sales.csv")

# Convert 'order_date' to datetime, handling mixed formats
sales_df['order_date'] = pd.to_datetime(sales_df['order_date'], errors='coerce', dayfirst=True)

# Drop rows where 'order_date' conversion failed (NaT values)
sales_df = sales_df.dropna(subset=['order_date'])

# Drop rows with missing values (if any)
sales_df.dropna(inplace=True)
ingredients_df.dropna(inplace=True)

# Handle outliers by capping extreme values in quantity (e.g., 99th percentile)
q99 = sales_df['quantity'].quantile(0.99)
sales_df['quantity'] = np.where(sales_df['quantity'] > q99, q99, sales_df['quantity'])

# Feature Engineering
# Extract day, month, and year from order_date
sales_df['day_of_week'] = sales_df['order_date'].dt.day_name()
sales_df['month'] = sales_df['order_date'].dt.month
sales_df['year'] = sales_df['order_date'].dt.year

# Ensure there are no duplicate 'order_date' values by aggregating
sales_df = sales_df.groupby('order_date').agg({'quantity': 'sum'}).reset_index()

# Set 'order_date' as the index for time series analysis
sales_df.set_index('order_date', inplace=True)

# Ensure the data has a daily frequency (fill missing dates)
sales_df = sales_df.asfreq('D')

# Fill missing values (if any) by forward filling
#sales_df['quantity'].fillna(method='ffill', inplace=True)
sales_df['quantity'] = sales_df['quantity'].ffill()

# ------------------- Exploratory Data Analysis (EDA) -------------------

# Aggregate sales data for daily, weekly, and monthly trends
daily_sales = sales_df.reset_index()
weekly_sales = sales_df['quantity'].resample('W').sum().reset_index()
monthly_sales = sales_df['quantity'].resample('M').sum().reset_index()

# Plot Daily Sales Trend
plt.figure(figsize=(10,6))
plt.plot(daily_sales['order_date'], daily_sales['quantity'], color='blue')
plt.title('Daily Sales Trend')
plt.xlabel('Date')
plt.ylabel('Total Quantity Sold')
plt.xticks(rotation=45)
plt.show()

# Plot Weekly Sales Trend
plt.figure(figsize=(10,6))
plt.plot(weekly_sales['order_date'], weekly_sales['quantity'], color='green')
plt.title('Weekly Sales Trend')
plt.xlabel('Week')
plt.ylabel('Total Quantity Sold')
plt.xticks(rotation=45)
plt.show()

# Plot Monthly Sales Trend
plt.figure(figsize=(10,6))
plt.plot(monthly_sales['order_date'], monthly_sales['quantity'], color='purple')
plt.title('Monthly Sales Trend')
plt.xlabel('Month')
plt.ylabel('Total Quantity Sold')
plt.xticks(rotation=45)
plt.show()

# ------------------- ARIMA Model and Forecasting -------------------

# Split data into training and testing sets (use last 30 days for testing)
train_data = sales_df['quantity'][:-30]
test_data = sales_df['quantity'][-30:]

# Fit ARIMA model
arima_model = ARIMA(train_data, order=(5,1,0))  # ARIMA(5,1,0) as an example
arima_fit = arima_model.fit()

# Forecast next 30 days
forecast = arima_fit.forecast(steps=30)
forecast_index = pd.date_range(start=test_data.index[-1] + pd.Timedelta(days=1), periods=30, freq='D')

# Plot actual vs predicted sales
plt.figure(figsize=(10,6))
plt.plot(train_data.index, train_data, label='Training Data')
plt.plot(test_data.index, test_data, label='Actual Sales')
plt.plot(forecast_index, forecast, label='Forecasted Sales', color='red')
plt.title('ARIMA Model - Actual vs Forecasted Sales')
plt.xlabel('Date')
plt.ylabel('Quantity Sold')
plt.legend()
plt.xticks(rotation=45)
plt.show()

# Calculate MAPE
mape = mean_absolute_percentage_error(test_data, forecast)
print(f'MAPE: {mape:.2f}')
# Calculate MAPE
mape = mean_absolute_percentage_error(test_data, forecast) * 100  # Convert to percentage

# Print MAPE result
print(f'Mean Absolute Percentage Error (MAPE) on test data: {mape:.2f}%')
print("---------------------------------------------------------------------")

# ------------------- Calculate Ingredient Purchase Order -------------------

# Total pizzas predicted for the next week (7 days from the forecast)
total_pizzas_predicted = forecast[:7].sum()

# Create 'quantity_per_pizza' if it doesn't exist
if 'quantity_per_pizza' not in ingredients_df.columns:
    ingredients_df['quantity_per_pizza'] = np.random.uniform(0.1, 1.0, size=len(ingredients_df))  # Random values; adjust as needed

# Calculate the required quantities of each ingredient based on the predicted sales
ingredients_df['total_quantity_needed'] = ingredients_df['quantity_per_pizza'] * total_pizzas_predicted

# Group by pizza_ingredients and sum total_quantity_needed to remove duplicates
ingredients_df = ingredients_df.groupby('pizza_ingredients', as_index=False).agg({'total_quantity_needed': 'sum'})

# Sort ingredients by total quantity needed in descending order
ingredients_df = ingredients_df.sort_values('total_quantity_needed', ascending=False)

# Save the complete purchase order to CSV
ingredients_df[['pizza_ingredients', 'total_quantity_needed']].to_csv('complete_purchase_order_next_week.csv', index=False)

# Print out the total purchase order for all ingredients
print("Total Purchase Order for the Next Week (All Ingredients):")
print(ingredients_df[['pizza_ingredients', 'total_quantity_needed']])

print("----------------------------------------------------------------------")

# Plot the top 10 ingredients needed
top_10_ingredients = ingredients_df.head(10)
plt.figure(figsize=(10, 6))
sns.barplot(data=top_10_ingredients, x='pizza_ingredients', y='total_quantity_needed', palette='viridis',hue='pizza_ingredients', dodge=False)
plt.title('Top 10 Ingredients Needed for Next Week')
plt.xlabel('Ingredient')
plt.ylabel('Total Quantity Needed (grams)')
plt.xticks(rotation=45)
plt.tight_layout()  # Adjusts layout to prevent overlap
plt.show()

# Save the top 10 purchase order to CSV
top_10_ingredients[['pizza_ingredients', 'total_quantity_needed']].to_csv('purchase_order_top_10_next_week.csv', index=False)
