from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
# Load Data
customers = pd.read_csv('Customers.csv')
products = pd.read_csv('Products.csv')
transactions = pd.read_csv('Transactions.csv')

# Merge datasets
transactions['TransactionDate'] = pd.to_datetime(transactions['TransactionDate'])
customers['SignupDate'] = pd.to_datetime(customers['SignupDate'])
merged_data = pd.merge(transactions, customers, on="CustomerID", how="inner")
merged_data = pd.merge(merged_data, products, on="ProductID", how="inner")

# PDF Creation
pdf = PdfPages("EDA_Report.pdf")

# 1. Title Page
plt.figure(figsize=(8, 6))
plt.text(0.5, 0.5, "Exploratory Data Analysis Report", fontsize=24, ha='center', va='center')
plt.text(0.5, 0.3, "E-Commerce Data Insights", fontsize=18, ha='center', va='center')
plt.axis('off')
pdf.savefig()
plt.close()

# 2. Dataset Overview
plt.figure(figsize=(10, 6))
sns.heatmap(merged_data.isnull(), cbar=False, cmap="viridis")
plt.title("Missing Values Heatmap")
pdf.savefig()
plt.close()

# 3. Revenue Trends
revenue_trends = merged_data.groupby(merged_data['TransactionDate'].dt.to_period("M"))['TotalValue'].sum().reset_index()
revenue_trends['TransactionDate'] = revenue_trends['TransactionDate'].dt.to_timestamp()

plt.figure(figsize=(12, 6))
sns.lineplot(data=revenue_trends, x="TransactionDate", y="TotalValue", marker="o")
plt.title("Revenue Trends Over Time")
plt.xticks(rotation=45)
plt.grid()
pdf.savefig()
plt.close()

# 4. Top Customers
top_customers = merged_data.groupby('CustomerName')['TotalValue'].sum().sort_values(ascending=False).head(10)
top_customers.plot(kind='bar', figsize=(12, 6), color="skyblue")
plt.title("Top Customers by Revenue")
plt.ylabel("Revenue")
plt.xlabel("Customer Name")
plt.grid(axis='y')
pdf.savefig()
plt.close()

# 5. Best-Selling Products
best_selling_products = merged_data.groupby('ProductName')['TotalValue'].sum().sort_values(ascending=False).head(10)
plt.figure(figsize=(10, 8))
best_selling_products.plot(kind='pie', autopct='%1.1f%%', startangle=90, colormap='coolwarm', textprops={'fontsize': 10}, wedgeprops={'edgecolor': '#fff'})
plt.title("Best-Selling Products (Revenue Share)")
pdf.savefig()
plt.close()

# 6. Monthly Trends
merged_data['Month'] = merged_data['TransactionDate'].dt.month
monthly_revenue = merged_data.groupby('Month')['TotalValue'].sum()

plt.figure(figsize=(12, 6))
sns.barplot(x=monthly_revenue.index, y=monthly_revenue.values, palette='husl')
plt.title("Monthly Revenue Trends")
plt.xticks(range(12), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.xlabel("Months")
plt.ylabel("Revenue")
plt.grid(axis='y')
pdf.savefig()
plt.close()

# 7. Concluding Slide
plt.figure(figsize=(8, 6))
plt.text(0.5, 0.5, "Thank You!", fontsize=24, ha='center', va='center')
plt.text(0.5, 0.4, "For further queries, contact us at xyz@example.com", fontsize=12, ha='center', va='center')
plt.axis('off')
pdf.savefig()
plt.close()

# Save PDF
pdf.close()
print("EDA Report saved as EDA_Report.pdf")
