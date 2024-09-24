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
    
    # Slider to select the radius_1
    html.Div([
        html.Label("Select Radius 1:"),
        dcc.Slider(
            id='radius-slider-1',
            min=0,
            max=0.5,
            step=0.01,
            value=0.2,  # Default value
            marks={i/10: str(i/10) for i in range(11)}
        )
    ], style={'margin-top': '20px'}),
    
    # Slider to select the radius_2
    html.Div([
        html.Label("Select Radius 2:"),
        dcc.Slider(
            id='radius-slider-2',
            min=0,
            max=0.5,
            step=0.01,
            value=0.2,  # Default value
            marks={i/10: str(i/10) for i in range(11)}
        )
    ], style={'margin-top': '20px'}),
    
    # Input to set epsilon_block
    html.Div([
        html.Label("Set Epsilon Block:"),
        dcc.Input(
            id='epsilon-block-input',
            type='number',
            value=1.0  # Default value
        )
    ], style={'margin-top': '20px'}),
    
    # Input to set epsilon_1
    html.Div([
        html.Label("Set Epsilon 1:"),
        dcc.Input(
            id='epsilon-1-input',
            type='number',
            value=12.0  # Default value
        )
    ], style={'margin-top': '20px'}),
    
    # Input to set height
    html.Div([
        html.Label("Set Height:"),
        dcc.Input(
            id='height-input',
            type='number',
            value=mp.inf  # Default value
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
     State('radius-slider-1', 'value'),
     State('radius-slider-2', 'value'),
     State('epsilon-block-input', 'value'),
     State('epsilon-1-input', 'value'),
     State('height-input', 'value')]
)
def update_plots(n_clicks, lattice_type, radius_1, radius_2, epsilon_block, epsilon_1, height):
    if n_clicks == 0:
        # Return empty figures if the button has not been clicked
        return go.Figure(), go.Figure()
    
    # Create a new photonic crystal with the selected parameters
    pc = PhotonicCrystal(
        lattice_type=lattice_type,
        geometry=PhotonicCrystal.slab_geometry(
            slab_h=height,
            supercell_h=4,  # You can adjust this value as needed
            eps_sub=epsilon_block,
            eps_atom_1=epsilon_1,
            eps_atom_2=epsilon_1,  # Assuming the same epsilon for both atoms
            eps_background=1,  # Assuming background epsilon is 1
            radius_1=radius_1,
            radius_2=radius_2
        ),
    )
    
    # Run the simulation
    pc.run_simulation(type='both')
    pc.extract_data(periods=5)
    
    # Generate dielectric distribution plot interactively
    dielectric_fig = go.Figure(layout=dict(width=800, height=800))
    converted_eps = pc.plot_epsilon_interactive(fig=dielectric_fig)
    print(f"Shape of converted epsilon: {converted_eps.shape}")
    
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
     Input('radius-slider-1', 'value'),
     Input('radius-slider-2', 'value'),
     Input('epsilon-block-input', 'value'),
     Input('epsilon-1-input', 'value'),
     Input('height-input', 'value')]
)
def update_field(clickData, lattice_type, radius_1, radius_2, epsilon_block, epsilon_1, height):
    # Create a new photonic crystal with the selected parameters
    pc = PhotonicCrystal(
        lattice_type=lattice_type,
        geometry=PhotonicCrystal.basic_geometry(radius_1=radius_1,
                                                 radius_2=radius_2,
                                                 eps_atom_1=epsilon_1,
                                                 eps_block=epsilon_block,
                                                 height=mp.inf),
    )
    
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