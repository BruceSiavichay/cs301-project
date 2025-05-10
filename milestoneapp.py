import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import base64
import io
import os
import plotly.express as px
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Set up Dash application
app = dash.Dash(
    __name__,
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)
server = app.server

# Shared data store for callbacks
df_store = {}

def preprocess_df(df):
    """
    Prepare dataframe: strip headers, remove duplicates,
    fill missing values, filter negatives, scale and encode data.
    """
    df = df.copy()
    df.columns = df.columns.str.strip()
    df.drop_duplicates(inplace=True)

    # Identify column types
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    if numeric_cols:
        # Impute numeric columns and remove negative rows
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        df = df[(df[numeric_cols] >= 0).all(axis=1)]
        # Scale features
        df[numeric_cols] = StandardScaler().fit_transform(df[numeric_cols])

    if categorical_cols:
        # Handle missing categories and encode labels
        df[categorical_cols] = df[categorical_cols].fillna('Unknown')
        for col in categorical_cols:
            df[col] = LabelEncoder().fit_transform(df[col])

    return df

# Define app layout
app.layout = html.Div([
    html.Div(
        html.H3('Upload Your CSV', style={'textAlign': 'center'}),
        style={'backgroundColor': '#d3d3d3', 'padding': '15px'}
    ),
    dcc.Upload(
        id='upload-data',
        children=html.Div(['Drag & drop or ', html.A('select a CSV file')]),
        style={
            'width': '98%', 'height': '60px', 'lineHeight': '60px',
            'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
            'textAlign': 'center', 'margin': '10px'
        },
        multiple=False
    ),
    html.Div(id='output-data-upload', style={'padding': '10px'}),
    html.Div(id='target-selector-container', style={'padding': '10px'}),
    html.Div(id='category-radio-container', style={'padding': '10px'}),
    html.Div([
        dcc.Graph(id='avg-target-bar', style={'width': '48%', 'display': 'inline-block'}),
        dcc.Graph(id='correlation-bar', style={'width': '48%', 'display': 'inline-block'})
    ], style={'display': 'flex', 'justifyContent': 'space-around', 'padding': '10px'}),
    html.Div(id='train-container', style={'padding': '10px'}),
    html.Div(id='predict-container', style={'padding': '10px'})
])

# Handle file upload and preprocessing
@app.callback(
    Output('output-data-upload', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_output(content, filename):
    if content:
        df_store.clear()
    if not content:
        return ''
    try:
        decoded = base64.b64decode(content.split(',')[1])
        df0 = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

        # Save raw column lists
        df_store['num_cols'] = df0.select_dtypes(include=[np.number]).columns.tolist()
        df_store['cat_cols'] = df0.select_dtypes(include=['object', 'category']).columns.tolist()

        # Preprocess
        df = preprocess_df(df0)
        df_store['df'] = df
        df_store['df_pre'] = df0

        rows, cols = df.shape
        return html.Div([
            html.H5(f"Loaded {filename}"),
            html.P(f"{rows:,} rows × {cols:,} columns processed.")
        ])
    except Exception as e:
        return html.Div(f"Error: {e}")

# Show target variable dropdown after upload
@app.callback(
    Output('target-selector-container', 'children'),
    Input('output-data-upload', 'children')
)
def display_target_selector(content):
    if 'df' not in df_store:
        return ''
    options = df_store['num_cols']
    return html.Div([
        html.Label('Choose target variable:'),
        dcc.Dropdown(id='target-dropdown', options=options, placeholder='Select a column')
    ])

# Build charts and feature selectors when target is set
@app.callback(
    Output('category-radio-container', 'children'),
    Output('avg-target-bar', 'figure'),
    Output('correlation-bar', 'figure'),
    Output('train-container', 'children'),
    Output('predict-container', 'children'),
    Input('target-dropdown', 'value')
)
def update_app(target_col):
    empty_fig = {'data': [], 'layout': {}}
    if not target_col or 'df' not in df_store:
        return '', empty_fig, empty_fig, '', ''

    df = df_store['df']
    df0 = df_store['df_pre']
    df_store['target'] = target_col

    # Small unique numeric cols as categories
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in num_cols if c != target_col and df[c].nunique() < 10]

    default_cat = cat_cols[0] if cat_cols else None

    # Category selector
    radio = html.Div([
        html.Label('Group by:'),
        dcc.RadioItems(id='category-radio', options=[{'label': c, 'value': c} for c in cat_cols], value=default_cat, inline=True)
    ])

    # Average target by category
    if target_col in df_store.get('num_cols', []):
        avg_df = df0.groupby(default_cat)[target_col].mean().reset_index()
        avg_fig = px.bar(avg_df, x=default_cat, y=target_col, title=f'Avg {target_col} by {default_cat}')
        avg_fig.update_traces(text=round(avg_df[target_col], 2), textposition='outside')
    else:
        avg_fig = empty_fig

    # Correlation with target
    corr_df = df0[df_store['num_cols']].corr()[target_col].drop(target_col).abs().reset_index()
    corr_df.columns = ['Feature', 'Corr']
    corr_fig = px.bar(corr_df, x='Feature', y='Corr', title=f'Correlation with {target_col}')
    corr_fig.update_traces(text=round(corr_df['Corr'], 2), textposition='outside')

    # Feature selection and train button
    features = [c for c in df.columns if c != target_col]
    df_store['features'] = features
    train_div = html.Div([
        html.Label('Pick features and train:'),
        dcc.Checklist(id='feature-checklist', options=[{'label': f, 'value': f} for f in features], value=features, inline=True),
        html.Button('Train', id='train-button'),
        html.Div(id='train-output')
    ])

    # Input for predictions
    predict_div = html.Div([
        html.Label('Input values (comma-separated):'),
        dcc.Input(id='predict-input', type='text', placeholder=','.join(features), style={'width': '50%'}),
        html.Button('Predict', id='predict-button'),
        html.Div(id='predict-output')
    ])

    return radio, avg_fig, corr_fig, train_div, predict_div

# Train linear regression model
@app.callback(
    Output('train-output', 'children'),
    Input('train-button', 'n_clicks'),
    State('feature-checklist', 'value'),
    prevent_initial_call=True
)
def train_model(n_clicks, selected_features):
    if not selected_features:
        return 'Select at least one feature.'
    df = df_store['df']
    X = df[selected_features]
    y = df[df_store['target']]
    model = LinearRegression().fit(X, y)
    df_store['model'] = model
    score = r2_score(y, model.predict(X))
    return f"R²: {score:.3f}"

# Predict new observations
@app.callback(
    Output('predict-output', 'children'),
    Input('predict-button', 'n_clicks'),
    State('predict-input', 'value'),
    State('feature-checklist', 'value'),
    prevent_initial_call=True
)
def predict_target(n_clicks, value, selected_features):
    model = df_store.get('model')
    if not model:
        return 'Train model first.'
    try:
        vals = [x for x in value.split(',')]
    except ValueError:
        return 'Invalid format.'

    if len(vals) != len(selected_features):
        return f'Expected {len(selected_features)} values.'
    
    input_df = pd.DataFrame([vals], columns=selected_features)

    prediction = model.predict(input_df)[0]
    return f"Predicted {df_store['target']}: {prediction:.3f}"

if __name__ == '__main__':
    app.run_server(debug=True, port=int(os.environ.get('PORT', 8050)), host='0.0.0.0')