import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from photonic_crystal import PhotonicCrystal
import meep as mp

# Initialize the Dash app
app = dash.Dash(__name__)

# App layout
app.layout = html.Div([
    # Dropdown to select the photonic crystal type
    dcc.Dropdown(
        id='crystal-dropdown',
        options=[
            {'label': 'Square Lattice', 'value': 'square'},
            {'label': 'Triangular Lattice', 'value': 'triangular'}
        ],
        value='square'  # Default value
    ),
    
    # Slider to select the radius
    html.Div([
        html.Label("Select Radius:"),
        dcc.Slider(
            id='radius-slider',
            min=0,
            max=0.5,
            step=0.01,
            value=0.2,  # Default value
            marks={i/10: str(i/10) for i in range(11)}
        )
    ], style={'margin-top': '20px'}),
    
    # Button to run the simulation
    html.Button('Run Simulation', id='run-button', n_clicks=0),
    
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

# Callback to update the dielectric and band structure plots based on the selected photonic crystal and radius
@app.callback(
    [Output('dielectric-plot', 'figure'),
     Output('band-structure-plot', 'figure')],
    [Input('run-button', 'n_clicks')],
    [State('crystal-dropdown', 'value'),
     State('radius-slider', 'value')]
)
def update_plots(n_clicks, lattice_type, radius):
    if n_clicks == 0:
        # Return empty figures if the button has not been clicked
        return go.Figure(), go.Figure()
    
    # Create a new photonic crystal with the selected parameters
    pc = PhotonicCrystal(lattice_type=lattice_type, geometry=PhotonicCrystal.basic_geometry(radius=radius))
    
    # Run the simulation
    pc.run_simulation(type='both')
    pc.extract_data(periods=5)
    
    # Generate dielectric distribution plot interactively
    dielectric_fig = go.Figure(layout=dict(width=800, height=800))
    pc.plot_epsilon_interactive(fig=dielectric_fig)
    
    # Generate band structure plot interactively
    band_fig = go.Figure(layout=dict(width=1000, height=800))
    pc.plot_bands_interactive(polarization='te', color="red", fig=band_fig)
    pc.plot_bands_interactive(polarization='tm', color="blue", fig=band_fig)
    
    return dielectric_fig, band_fig

# Callback to plot the electric field based on a clicked point in the band structure and radius
@app.callback(
    [Output('te-field-plot', 'figure'),
     Output('tm-field-plot', 'figure')],
    [Input('band-structure-plot', 'clickData'),
     Input('crystal-dropdown', 'value'),
     Input('radius-slider', 'value')]
)
def update_field(clickData, lattice_type, radius):
    # Create a new photonic crystal with the selected parameters
    pc = PhotonicCrystal(lattice_type=lattice_type, geometry=PhotonicCrystal.basic_geometry(radius=radius))
    
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
        pc.plot_field_interactive(k_point=k_point, frequency=frequency, fig=te_field_fig, runner="run_te")
        
        # Add the TM field plot to the bottom right
        tm_field_fig = go.Figure()
        pc.plot_field_interactive(k_point=k_point, frequency=frequency, fig=tm_field_fig, runner="run_tm")
        
        return te_field_fig, tm_field_fig
    else:
        # Return an empty figure if no point is clicked
        return go.Figure(), go.Figure()

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)