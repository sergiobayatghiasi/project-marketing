#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# In[2]:


# Set the random seed
np.random.seed(42)

# Create a date range for the past 180 days
dates = pd.date_range(end=datetime.today(), periods=180).to_list()

# Competitor brands and platforms
brands = ['Trendora', 'ChicHive', 'ModaPulse']
platforms = ['Instagram', 'Twitter', 'TikTok', 'Facebook', 'YouTube']


# In[3]:


# Generate data
data = []

for date in dates:
    for brand in brands:
        for _ in range(2):  # simulate 2 entries per day per brand
            platform = random.choice(platforms)
            mentions = np.random.poisson(lam=150 if brand == 'Trendora' else 100)
            sentiment = np.clip(np.random.normal(loc=0.7 if brand == 'Trendora' else 0.6, scale=0.15), -1, 1)
            ad_spend = round(np.random.normal(loc=2000 if brand == 'ModaPulse' else 1500, scale=400), 2)
            engagement = round(np.random.uniform(30, 95), 2)
            traffic = round(np.random.normal(loc=100 if brand == 'ChicHive' else 120, scale=25), 2)

            data.append({
                'Date': date.strftime('%Y-%m-%d'),
                'Brand': brand,
                'Platform': platform,
                'Mentions': mentions,
                'Sentiment': sentiment,
                'Ad_Spend_USD': ad_spend,
                'Engagement_Score': engagement,
                'Traffic_k': traffic
            })


# In[4]:


# Convert to DataFrame and save to CSV
df = pd.DataFrame(data)
df.to_csv("competitor_trends_dataset.csv", index=False)
print("Dataset saved as competitor_trends_dataset.csv")


# In[5]:


df


# In[6]:


df.describe()


# In[7]:


# Convert 'Date' to datetime
df['Date'] = pd.to_datetime(df['Date'])


# In[8]:


# Set style
sns.set(style="whitegrid")

# 1. Mentions Over Time
plt.figure(figsize=(14,6))
sns.lineplot(data=df, x='Date', y='Mentions', hue='Brand')
plt.title('Daily Mentions Over Time')
plt.show()


# In[9]:


# 2. Sentiment Over Time
plt.figure(figsize=(14,6))
sns.lineplot(data=df, x='Date', y='Sentiment', hue='Brand')
plt.title('Sentiment Trend Over Time')
plt.show()


# In[10]:


# 3. Ad Spend vs Engagement
plt.figure(figsize=(10,6))
sns.scatterplot(data=df, x='Ad_Spend_USD', y='Engagement_Score', hue='Brand')
plt.title('Ad Spend vs Engagement Score')
plt.show()


# In[11]:


# 4. Average Traffic by Brand
plt.figure(figsize=(8,6))
sns.barplot(data=df, x='Brand', y='Traffic_k', estimator='mean')
plt.title('Average Website Traffic by Brand')
plt.show()


# In[12]:


# Load and prepare data
df = pd.read_csv("competitor_trends_dataset.csv")
df = df.dropna()


# In[13]:


# Feature selection
X = df[['Mentions', 'Sentiment', 'Ad_Spend_USD', 'Engagement_Score']]
y = df['Traffic_k']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LinearRegression()
model.fit(X_train, y_train)


# In[14]:


# Predictions
y_pred = model.predict(X_test)


# In[15]:


# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


# In[16]:


# Output
print("🔍 Model Coefficients:")
print(dict(zip(X.columns, model.coef_)))
print(f"📈 R² Score: {r2:.2f}")
print(f"📉 MSE: {mse:.2f}")


# In[17]:


#Scenario: Increase Ad Spend by 20%


# In[18]:


# Create a copy of X_test for simulation
X_test_simulated = X_test.copy()

# Increase Ad Spend by 20%
X_test_simulated['Ad_Spend_USD'] *= 1.2

# Predict traffic for original and simulated data
y_pred_original = model.predict(X_test)
y_pred_simulated = model.predict(X_test_simulated)

# Compare predictions
comparison = pd.DataFrame({
    'Original_Traffic': y_pred_original,
    'Simulated_Traffic': y_pred_simulated,
    'Change_in_Traffic': y_pred_simulated - y_pred_original
})

# Average impact
avg_increase = comparison['Change_in_Traffic'].mean()

print(f"📈 On average, increasing ad spend by 20% results in +{avg_increase:.2f}k more traffic per record.")
comparison.head()


# In[19]:


#Time Series Forecasting (Mentions)


# In[20]:


#We’ll forecast Mentions for a brand (e.g., Trendora) using Facebook Prophet—great for seasonality, trends, and holidays


# In[21]:


get_ipython().system('pip install prophet')


# In[22]:


import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("competitor_trends_dataset.csv")
df['Date'] = pd.to_datetime(df['Date'])

# Filter data: let's forecast for Trendora
trendora = df[df['Brand'] == 'Trendora']

# Aggregate mentions per day
daily_mentions = trendora.groupby('Date')['Mentions'].sum().reset_index()
daily_mentions.columns = ['ds', 'y']  # Prophet expects these names

# Build & train the model
model = Prophet()
model.fit(daily_mentions)

# Create future dates (next 30 days)
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

# Plot the forecast
fig = model.plot(forecast)
plt.title('📈 Forecast of Trendora Mentions (Next 30 Days)')
plt.show()


# In[23]:


#Clustering Brands by Behavior
#We'll use KMeans Clustering to group records based on:

 #Mentions

 #Sentiment

 #Ad Spend

 #Engagement Score


# In[24]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load and prepare
df = pd.read_csv("competitor_trends_dataset.csv")
features = ['Mentions', 'Sentiment', 'Ad_Spend_USD', 'Engagement_Score']
X = df[features]

# Normalize & cluster
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Assign manual colors to clusters
cluster_colors = {
    0: '#1f77b4',  # blue
    1: '#2ca02c',  # green
    2: '#ff7f0e',  # orange
}

# Add color column
df['Cluster_Color'] = df['Cluster'].map(cluster_colors)

# Plot with custom colors and markers
plt.figure(figsize=(12, 6))
brands = df['Brand'].unique()
markers = ['o', 's', '^', 'P', 'D', 'X', '*']  # enough for 7+ brands

for i, brand in enumerate(brands):
    subset = df[df['Brand'] == brand]
    plt.scatter(
        subset['Ad_Spend_USD'],
        subset['Engagement_Score'],
        c=subset['Cluster_Color'],
        label=brand,
        marker=markers[i % len(markers)],
        edgecolor='k',
        s=100
    )

plt.title("💡 Clustering Brands by Marketing Behavior")
plt.xlabel("Ad Spend (USD)")
plt.ylabel("Engagement Score")
plt.legend(title="Brand", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# In[27]:


# Save the dataframe with the 'Cluster' column
df.to_csv("competitor_with_clusters.csv", index=False)


# In[28]:


# Assuming 'forecast' is the dataframe with predictions from Prophet
forecast.to_csv("prophet_forecast_results.csv", index=False)


# In[ ]:




