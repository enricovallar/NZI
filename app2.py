import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import os
import pickle
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from photonic_crystal import PhotonicCrystal
import math
import meep as mp
from meep import mpb

# Path to pickle data folder
PICKLE_DIR = 'pickle_data'

# Helper function to list all pickled photonic crystals by their IDs
def get_photonic_crystal_ids():
    ids = []
    for file in os.listdir(PICKLE_DIR):
        if file.endswith('.pkl'):
            ids.append(file.split('.pkl')[0])  # Remove .pkl to get the ID
    return ids

# Initialize the Dash app
app = dash.Dash(__name__)

# App layout
app.layout = html.Div([
    # Dropdown to select the photonic crystal
    dcc.Dropdown(
        id='crystal-dropdown',
        options=[{'label': pc_id, 'value': pc_id} for pc_id in get_photonic_crystal_ids()],
        value=get_photonic_crystal_ids()[0]  # Default value is the first photonic crystal
    ),
    
    # Graphs for the dielectric distribution and band structure in the same row
    html.Div([
        html.Div([
            html.H3("Dielectric Distribution"),
            dcc.Graph(id='dielectric-plot')
        ], style={'display': 'inline-block', 'width': '50%'}),
        
        html.Div([
            html.H3("Band Structure"),
            dcc.Graph(id='band-structure-plot')
        ], style={'display': 'inline-block', 'width': '50%'})
    ]),
    
    # Graphs for electric field distribution in the same row
    html.Div([
        html.Div([
            html.H3("TE Field Distribution"),
            dcc.Graph(id='te-field-plot')
        ], style={'display': 'inline-block', 'width': '50%'}),

        html.Div([
            html.H3("TM Field Distribution"),
            dcc.Graph(id='tm-field-plot')
        ], style={'display': 'inline-block', 'width': '50%'})
        
    ])
])

# Callback to update the dielectric and band structure plots based on the selected photonic crystal
@app.callback(
    [Output('dielectric-plot', 'figure'),
     Output('band-structure-plot', 'figure')],
    [Input('crystal-dropdown', 'value')]
)
def update_plots(selected_id):
    # Load the selected photonic crystal from the pickle file
    pickle_file = os.path.join(PICKLE_DIR, f'{selected_id}.pkl')
    with open(pickle_file, 'rb') as f:
        pc = pickle.load(f)
    
    # Show a circular indicator when the simulation starts
    indicator = dcc.Loading(
        id="loading-indicator",
        type="circle",
        children=html.Div(id="loading-output")
    )
    
    # Run the simulation
    pc.run_simulation(type='both')
    pc.extract_data(periods=5)
    
    # Hide the circular indicator when the simulation ends
    indicator.children = html.Div(id="loading-output", children="Simulation complete")
    
    # Generate dielectric distribution plot interactively
    dielectric_fig = go.Figure(layout=dict(width=500, height=500))
    pc.plot_epsilon_interactive(fig=dielectric_fig)
    
    # Generate band structure plot interactively
    band_fig = go.Figure(layout=dict(width=500, height=500))
    pc.plot_bands_interactive(polarization='te', color="red", fig=band_fig)
    pc.plot_bands_interactive(polarization='tm', color="blue", fig=band_fig)
    
    return dielectric_fig, band_fig

# Callback to plot the electric field based on a clicked point in the band structure
@app.callback(
    [Output('te-field-plot', 'figure'),
     Output('tm-field-plot', 'figure')],
    [Input('band-structure-plot', 'clickData'),
     Input('crystal-dropdown', 'value')]
)
def update_field(clickData, selected_id):
    # Load the photonic crystal from the pickle file
    pickle_file = os.path.join(PICKLE_DIR, f'{selected_id}.pkl')
    with open(pickle_file, 'rb') as f:
        pc = pickle.load(f)
    
    # Run the simulation if not done already
    pc.run_simulation(type='both')
    pc.extract_data(periods=5)
    
    if clickData:
        # Extract the clicked k-point data (assuming it's available in the customdata)
        k_point_data = clickData['points'][0]['customdata']
        k_point = mp.Vector3(k_point_data[0], k_point_data[1], k_point_data[2])
        
        # Extract the corresponding frequency
        frequency = clickData['points'][0]['y']
        
        # Generate the electric field plot interactively
        field_fig = make_subplots(rows=2, cols=2, subplot_titles=("Bands Plot", "Eps Plot", "TE Field", "TM Field"))


        # Add the TE field plot to the bottom left
        te_field_fig = go.Figure()
        pc.plot_field_interactive(k_point=k_point, frequency=frequency, fig=te_field_fig, runner = "run_te")
        
        # Add the TM field plot to the bottom right
        tm_field_fig = go.Figure()
        pc.plot_field_interactive(k_point=k_point, frequency=frequency, fig=tm_field_fig, runner = "run_tm")
        

        return te_field_fig, tm_field_fig
    else:
        # Return an empty figure if no point is clicked
        return go.Figure()

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)


    