#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# get_ipython().system('pip install dash')
# get_ipython().system('pip install pyxirr')


# In[1]:


import os
import re
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash
from pyxirr import xirr
from dash import dcc, html, dash_table
from datetime import datetime, timedelta
from dash.dependencies import Input, Output, State


# In[3]:


# Path to the folder containing the CSV files
folder_path = '../data'

# List all CSV files in the directory
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]  # Get only .csv files

# Initialize an empty list to store DataFrames
dfs = []

# Loop through each CSV file
for file in csv_files:
    file_path = os.path.join(folder_path, file)  # Get the full file path
    df = pd.read_csv(file_path)  # Read the CSV file into a DataFrame
    # Add a new column 'Type' which contains the base name of the file (removes the suffix after the first hyphen)
    df['Type'] = re.sub(r'-.*\.csv$', '', file)  
    dfs.append(df)  # Append the DataFrame to the list

# Concatenate all DataFrames in the list into a single DataFrame
data_df = pd.concat(dfs, axis=0)

# Trim leading and trailing spaces from column names
data_df.columns = data_df.columns.str.strip()

# Convert the 'Date' column to datetime format, ensuring it's interpreted correctly
data_df['Date'] = pd.to_datetime(data_df['Date'], format='%d-%b-%Y')

# Set the 'Date' column as the index of the DataFrame
data_df = data_df.set_index('Date')

# Sort the DataFrame by the index (dates) in ascending order
data_df = data_df.sort_index(ascending=True).reset_index(drop=False)

# Display the first few rows of the final DataFrame
# display(data_df.head(10))


# In[5]:


def create_sip_df(amount, freq, start_dt, end_dt, annual_increment=0.0):
    """
    Create a SIP DataFrame based on amount, frequency, start date, and end date.
    The SIP amount will increase annually by the given increment.

    Args:
    - amount (float): The SIP amount for each transaction.
    - freq (str): The frequency of SIP ('daily', 'weekly', 'monthly').
    - start_dt (str or datetime): The start date for the SIP transactions (in 'YYYY-MM-DD' format or datetime).
    - end_dt (str or datetime): The end date for the SIP transactions (in 'YYYY-MM-DD' format or datetime).
    - annual_increment (float): The annual increment in the SIP amount as a percentage (e.g., 0.1 for 10%).

    Returns:
    - pd.DataFrame: A DataFrame containing two columns: 'Date' and 'Amount'.
    """
    # Convert start and end dates to datetime objects
    start_dt = pd.to_datetime(start_dt)
    end_dt = pd.to_datetime(end_dt)

    # Generate dates based on the frequency
    if freq == 'daily':
        dates = pd.date_range(start=start_dt, end=end_dt, freq='D')
    elif freq == 'weekly':
        dates = pd.date_range(start=start_dt, end=end_dt, freq='W-MON')  # Weekly on Monday
    elif freq == 'monthly':
        dates = pd.date_range(start=start_dt, end=end_dt, freq='MS')  # Monthly on the start of each month
    else:
        raise ValueError("Frequency must be 'daily', 'weekly', or 'monthly'.")

    # Create a DataFrame with the generated dates
    sip_df = pd.DataFrame({'Date': dates})

    # Add the SIP amount and increment it yearly if annual_increment > 0
    sip_df['Amount'] = amount  # Set the initial SIP amount

    # Apply the annual increment logic
    if annual_increment > 0:
        # Add a column for year
        sip_df['Year'] = sip_df['Date'].dt.year
        
        # For each year, update the SIP amount based on the annual increment
        # This will increment the amount at the start of each new year
        sip_df['Amount'] = sip_df.apply(
            lambda row: amount * (1 + annual_increment) ** (row['Year'] - start_dt.year) if row['Year'] > start_dt.year else amount, axis=1
        )

    return sip_df[['Date', 'Amount']]


# In[7]:


def get_sip_units(sip_df, index, price_type, data_df):
    """
    Calculate the SIP units for the specified index and price type.
    
    Args:
    - sip_df (pd.DataFrame): DataFrame containing SIP transaction dates ('txn_dates') and amounts ('sip_amounts').
    - index (str): The specific 'Type' in `data_df` to filter by.
    - price_type (str): The price column in `data_df` to use for unit calculation.
    - data_df (pd.DataFrame): DataFrame containing financial data with 'Date', 'Type', and price columns.

    Returns:
    - pd.Series: A series of calculated SIP units.
    """
    # Ensure the relevant columns are present in both DataFrames
    required_columns_sip = ['Date', 'Amount']
    required_columns_data = ['Date', price_type]

    if not all(col in sip_df.columns for col in required_columns_sip):
        raise ValueError(f"SIP DataFrame must contain the following columns: {required_columns_sip}")
    if not all(col in data_df.columns for col in required_columns_data):
        raise ValueError(f"Data DataFrame must contain the following columns: {required_columns_data}")

    # Merge sip_df with data_df based on the 'Date' column (on the matching dates or next available date)
    sub_df = pd.merge_asof(
        sip_df, data_df.loc[data_df['Type'] == index, ['Date', price_type]], direction='forward'
    )

    # Calculate the units by dividing the sip_amounts by the price_type (price) value
    sub_df["Units"] = sub_df["Amount"] / sub_df[price_type]

    return sub_df["Units"]


# In[9]:


def get_price(index, price_type, data_df):
    """
    Get the price for a specific index and price_type from data_df.

    Args:
    - index (str): The value of the 'Type' column to filter by.
    - price_type (str): The price column to fetch from data_df.
    - data_df (pd.DataFrame): DataFrame containing financial data.

    Returns:
    - pd.DataFrame: A DataFrame with columns 'date' and 'price'.
    """
    # Ensure the required columns exist in the DataFrame
    if 'Type' not in data_df.columns or price_type not in data_df.columns:
        raise ValueError(f"DataFrame must contain 'Type' and '{price_type}' columns.")

    # Filter the data based on 'Type', and select 'Date' and the price_type column
    sub_df = data_df.loc[data_df['Type'] == index, ['Date', price_type]]

    # Rename the columns
    sub_df = sub_df.rename(columns={'Date': 'Date', price_type: 'Price'})

    return sub_df


# In[11]:


def get_next_monday(date):
    # Calculate the days to the next Monday (0 = Monday, 1 = Tuesday, ..., 6 = Sunday)
    days_to_next_monday = (7 - date.weekday()) % 7
    # If today is Monday, we want to return the same day
    return date + timedelta(days=days_to_next_monday) if days_to_next_monday != 0 else date


# In[13]:


def get_next_month(date):
    # If it's the first day of the month, return the same date
    if date.day == 1:
        return date
    
    # Calculate next month and year
    next_month = date.month % 12 + 1
    next_year = date.year if date.month < 12 else date.year + 1
    
    # Return the first day of the next month
    return datetime(next_year, next_month, 1)


# In[17]:


# Define the Dash app
app = dash.Dash(__name__)

# Define the layout of the Dash app
app.layout = html.Div([
    html.H1("Historical Index Performance"),
    dcc.Dropdown(
        id='price-type-dropdown',
        options=[
            {'label': 'Open', 'value': 'Open'},
            {'label': 'High', 'value': 'High'},
            {'label': 'Low', 'value': 'Low'},
            {'label': 'Close', 'value': 'Close'}
        ],
        value='Open',
        style={'width': '150px', 'font-size': '16px'}
    ),
    dcc.Graph(id='historical-line-plot'),

    html.H1("XIRR Calculator for SIP"),
    html.Div([
        html.Label("Index:", style={'padding-right': '8px'}),
        dcc.Dropdown(
            id='index-dropdown',
            options=[
                {'label': 'NIFTY 50', 'value': 'NIFTY 50'},
                {'label': 'NIFTY PRIVATE BANK', 'value': 'NIFTY PRIVATE BANK'},
                {'label': 'NIFTY BANK', 'value': 'NIFTY BANK'},
                {'label': 'NIFTY NEXT 50', 'value': 'NIFTY NEXT 50'},
                {'label': 'NIFTY MIDCAP 50', 'value': 'NIFTY MIDCAP 50'},
                {'label': 'NIFTY SMALLCAP 50', 'value': 'NIFTY SMALLCAP 50'}
            ],
            value='NIFTY 50',
            style={'width': '220px', 'padding-right': '8px'}
        ),
        html.Label("Amount: ", style={'padding-right': '8px'}),
        dcc.Input(id='sip-amount-input', type='number', value=10000, style={'width': '80px'}),
        html.Label("Frequency:", style={'padding-left': '8px', 'padding-right': '8px'}),
        dcc.Dropdown(
            id='sip-frequency-dropdown',
            options=[
                {'label': 'Daily', 'value': 'daily'},
                {'label': 'Weekly', 'value': 'weekly'},
                {'label': 'Monthly', 'value': 'monthly'}
            ],
            value='monthly',
            style={'width': '130px', 'padding-right': '8px'}
        ),
        html.Label("Start Date: ", style={'padding-right': '8px'}),
        dcc.DatePickerSingle(id='sip-start-date', date='2019-01-01'),
        html.Label("End Date: ", style={'padding-left': '8px', 'padding-right': '8px'}),
        dcc.DatePickerSingle(id='sip-end-date', date='2023-12-20'),
        html.Label("Increment (annual): ", style={'padding-left': '8px', 'padding-right': '8px'}),
        dcc.Input(id='sip-increment-input', type='number', value=0.05, style={'width': '50px'}),
        html.Button('Calculate', id='submit-button', n_clicks=0, style={'padding-left': '12px'}),
    ], style={'display': 'flex', 'align-items': 'center', 'margin-bottom': '10px', 'height': '30px', 'font-size': '16px'}),

    html.Div([html.P(id='output-message')]),
    html.Div([html.H4(id='output-xirr')]),

    # New Graph for displaying XIRR over time
    dcc.Graph(id='xirr-over-time-plot'),
])


@app.callback(
    Output('historical-line-plot', 'figure'),
    Input('price-type-dropdown', 'value')
)
def update_historical_plot(selected_price_type):
    fig = px.line(data_df, x='Date', y=selected_price_type, color='Type')
    
    # Customize the plot appearance
    fig.update_traces(line=dict(width=2), marker=dict(size=2))  # Set line width and marker size
    fig.update_traces(mode='lines+markers', hovertemplate='Date: %{x:%d-%m-%Y}<br>' + f'{selected_price_type}: ' +'%{y:,.0f}')

    # Set the layout properties for the plot
    fig.update_layout(
        height=700,
        yaxis_title=f'Daily {selected_price_type}',
    )
    return fig


@app.callback(
    Output('output-message', 'children'),
    Output('output-xirr', 'children'),
    Output('xirr-over-time-plot', 'figure'),
    Input('price-type-dropdown', 'value'),
    Input('index-dropdown', 'value'),
    Input('sip-frequency-dropdown', 'value'),
    Input('submit-button', 'n_clicks'),
    State('sip-amount-input', 'value'),
    State('sip-start-date', 'date'),
    State('sip-end-date', 'date'),
    State('sip-increment-input', 'value'),
)
def calculate_xirr(price_type, index_type, sip_frequency, submit_clicks, sip_amount, sip_start_date, sip_end_date, sip_increment):
    # Create the IRR over time chart
    xirr_fig = {
        'data': [go.Scatter(x=[], y=[], mode='lines', name='Cumulative IRR')],
        'layout': go.Layout(
            title='XIRR Over Investment Period',
            xaxis={'title': 'Date'},
            yaxis={'title': 'IRR (%)'},
        )
    }
    
    if submit_clicks == 0:
        return "Please enter details to calculate XIRR", "NA", xirr_fig

    # Check for missing inputs
    if not sip_amount or not sip_frequency or not sip_start_date or not sip_end_date:
        status_message = "Please enter SIP amount, frequency, and valid start/end dates"
        return status_message, "NA", xirr_fig

    status_message = f"Calculating XIRR for {sip_frequency} SIP of {sip_amount} from {sip_start_date} to {sip_end_date}..."

    # Convert input dates to datetime format
    start_date = datetime.strptime(sip_start_date, '%Y-%m-%d')
    end_date = datetime.strptime(sip_end_date, '%Y-%m-%d')

    if sip_frequency == 'weekly':
        start_date = get_next_monday(start_date)
    elif sip_frequency == 'monthly':
        start_date = get_next_month(start_date)

    # Assuming `create_sip_df` and `get_sip_units` are correctly defined
    sip_df = create_sip_df(sip_amount, sip_frequency, start_date, end_date, sip_increment)
    sip_df['Units'] = get_sip_units(sip_df, index_type, price_type, data_df)

    # Calculate cumulative SIP amounts and units
    sip_df['Cum_Amount'] = sip_df['Amount'].cumsum()
    sip_df['Cum_Units'] = sip_df['Units'].cumsum()

    # Assuming `get_price` function is defined
    price_df = get_price(index_type, price_type, data_df)

    cumulative_irr = []
    xirr_dates = []

    # Create an empty DataFrame with the required columns
    returns_df = pd.DataFrame(columns=['date', 'invested_amt', 'current_amt', 'xirr'])

    for date, price in price_df[(price_df['Date'] >= start_date)].values:
        dates_up_to_now = sip_df.loc[sip_df['Date'] <= date, 'Date'].tolist()
        sip_amounts_up_to_now = sip_df.loc[sip_df['Date'] <= date, 'Amount'].tolist()

        invested_amt = sum(sip_amounts_up_to_now)

        units_up_to_now = sip_df.loc[sip_df['Date'] <= date, 'Cum_Units'].iloc[-1]
        withdrawal_amt_up_to_now = units_up_to_now * price

        dates_up_to_now.append(date)
        sip_amounts_up_to_now.append(-withdrawal_amt_up_to_now)

        # Calculate the XIRR
        irr_value = xirr(dates_up_to_now, sip_amounts_up_to_now)
        cumulative_irr.append(irr_value)
        xirr_dates.append(date)

        new_row = pd.DataFrame(
            {
                'date': [date], 
                'invested_amt': [invested_amt], 
                'current_amt': [withdrawal_amt_up_to_now], 
                'xirr': [irr_value]
            }
        )

        # Use pd.concat to add the new row
        returns_df = pd.concat([returns_df, new_row], ignore_index=True)

    total_invested = sip_df['Cum_Amount'].iloc[-1]
    final_irr = cumulative_irr[-1]

    xirr_message = f"Total Amount Invested = {total_invested}\nIRR is {final_irr * 100:0.3f}%"

    xirr_fig = {
        'data': [
            # Invested Amount and Current Amount (Primary Y-axis)
            go.Scatter(
                x=returns_df['date'],
                y=returns_df['invested_amt'],
                mode='lines+markers',
                name='Invested Amount',
                line={'color': 'blue', 'width': 2},
                marker={'size': 2},
                yaxis='y1',
                hovertemplate=(
                    'Date: %{x|%Y-%m-%d}' +  # Format date as '2019-01-01'
                    '<br>Invested Amount: ₹%{y:,.0f}' +  # Format amount with commas
                    '<br><extra></extra>'  # Hide the trace name in hover
                )
            ),
            go.Scatter(
                x=returns_df['date'],
                y=returns_df['current_amt'],
                mode='lines+markers',
                name='Current Amount',
                line={'color': 'green', 'width': 2},
                marker={'size': 2},
                yaxis='y1',
                hovertemplate=(
                    'Date: %{x|%Y-%m-%d}' +  # Format date as '2019-01-01'
                    '<br>Current Amount: ₹%{y:,.0f}' +  # Format amount with commas
                    '<br><extra></extra>'  # Hide the trace name in hover
                )
            ),
            # XIRR (Secondary Y-axis)
            go.Scatter(
                x=returns_df['date'],
                y=returns_df['xirr'] * 100,  # Convert to percentage
                mode='lines+markers',
                name='XIRR',
                line={'color': 'red', 'width': 2},
                marker={'size': 2},
                yaxis='y2',
                hovertemplate=(
                    'Date: %{x|%Y-%m-%d}' +  # Format date as '2019-01-01'
                    '<br>XIRR: %{y:.2f}%' +  # Format XIRR as percentage with 2 decimals
                    '<br><extra></extra>'  # Hide the trace name in hover
                )
            ),
        ],
        'layout': go.Layout(
            title='Investment and XIRR Over Time',
            xaxis={'title': 'Date'},
            yaxis={
                'title': 'Amount',
                'titlefont': {'color': 'blue'},
                'tickfont': {'color': 'blue'}
            },
            yaxis2={
                'title': 'XIRR (%)',
                'titlefont': {'color': 'red'},
                'tickfont': {'color': 'red'},
                'overlaying': 'y',
                'side': 'right'
            },
            legend={'x': 0.1, 'y': 1.1, 'orientation': 'h'},
            height=700,
        )
    }

    return status_message, xirr_message, xirr_fig


if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)


